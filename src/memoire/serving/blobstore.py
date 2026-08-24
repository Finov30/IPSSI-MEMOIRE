"""Object store behind the claim-check transport (chap. 5.4).

Kafka carries a reference; the pixels live here. In production this is S3, in
the lab a directory on a shared volume — the service only ever sees the
:class:`BlobStore` protocol, so the tests use :class:`InMemoryBlobStore` and
touch neither disk nor network.

The error taxonomy is the important part, because it decides the fate of a
message:

- :class:`BlobNotFound` is PERMANENT (the object was never written, or its
  retention expired before the topic's — retention of the store must be >=
  retention of the topic, otherwise a replay breaks). Retrying forever would
  block the partition, so the message is dead-lettered.
- :class:`BlobUnavailable` is TRANSIENT (store unreachable, I/O error). It is
  retried with backoff, and if the budget runs out the service stops WITHOUT
  committing so the records are re-read after restart.
- :class:`InvalidBlobKey` is PERMANENT (a URI the filesystem itself refuses:
  embedded NUL, name too long). It comes from an untrusted message, so it must
  land in the DLQ rather than be retried forever.

The distinction between "this blob is missing" and "the store is not there"
is load-bearing: an unmounted volume answers *not found* for every message,
and a permanent verdict would drain the whole backlog into the DLQ, offsets
committed, without a single alarm. :meth:`FilesystemBlobStore.get` therefore
checks the root before pronouncing :class:`BlobNotFound`, and the loop caps
consecutive dead letters (``service.ServiceConfig.max_consecutive_dead_letters``)
for the case where the mount point exists but is empty.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

logger = logging.getLogger("memoire.serving.blobstore")

_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "application/octet-stream": ".bin"}

#: Errno values that mean "this key can never work", not "the disk is sick".
_PERMANENT_ERRNOS = frozenset({errno.ENAMETOOLONG, errno.EINVAL, errno.EISDIR})

#: Everything outside this class is replaced when slugifying an untrusted id.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

#: Path segments stay well under the 255-byte limit of ext4/overlayfs, with
#: room for the ``.png`` suffix and the ``.part`` staging suffix.
MAX_SEGMENT_LEN = 96


class BlobNotFound(KeyError):
    """Permanent: no object at that URI."""


class BlobUnavailable(OSError):
    """Transient: the store itself is failing."""


class InvalidBlobKey(ValueError):
    """Permanent: the key/URI is unusable, whatever the state of the store."""


def safe_key_segment(value: str) -> str:
    """Turn an untrusted identifier into ONE safe path segment.

    ``photo_id`` and ``inspection_id`` come off a topic: they may contain
    ``/`` (the ids used in the lab do), an embedded NUL (legal in JSON,
    rejected by ``Path.resolve``), or 300 characters (a base64 id). Building
    a blob key straight from them makes the *publish* stage fail identically
    on every replay of that offset — a permanent error dressed as a transient
    one, which blocks the partition forever.

    Slugifying instead of rejecting keeps a legitimate-but-verbose producer
    working; the short digest suffix restores the uniqueness the substitution
    would otherwise lose (two different ids never collapse onto one key).
    """
    slug = _UNSAFE_SEGMENT_CHARS.sub("-", value).strip("-.")
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    if not slug:
        return digest
    if slug == value and len(slug) <= MAX_SEGMENT_LEN:
        return slug
    return f"{slug[: MAX_SEGMENT_LEN - len(digest) - 1]}-{digest}"


class BlobStore(Protocol):
    def get(self, uri: str) -> bytes: ...

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        """Store ``data`` under ``key``; return the URI to publish."""
        ...


class FilesystemBlobStore:
    """``file://`` store rooted at a directory (the shared lab volume).

    URIs are resolved under ``root`` and any attempt to escape it (``..``,
    absolute path) is refused: a message coming off a topic is untrusted
    input, and the service must not be a file-read primitive for whoever can
    produce to the topic.
    """

    def __init__(self, root: Path | str, scheme: str = "file") -> None:
        self.root = Path(root).resolve()
        self.scheme = scheme

    def _resolve(self, uri: str) -> Path:
        # Everything here runs on an untrusted string: urlparse rejects some
        # inputs outright and Path.resolve raises ValueError on an embedded
        # NUL. Both are permanent (the same bytes fail identically on every
        # replay), so they must not surface as a retryable failure.
        try:
            parsed = urlparse(uri)
            if parsed.scheme and parsed.scheme != self.scheme:
                raise BlobNotFound(f"unsupported URI scheme {parsed.scheme!r} for {uri!r}")
            # file://host/path is unusual but legal; netloc is part of the path here.
            relative = f"{parsed.netloc}{parsed.path}" if parsed.scheme else uri
            candidate = (self.root / relative.lstrip("/")).resolve()
        except ValueError as exc:
            raise InvalidBlobKey(f"unusable blob key {uri!r}: {exc}") from exc
        except OSError as exc:
            if exc.errno in _PERMANENT_ERRNOS:
                raise InvalidBlobKey(f"unusable blob key {uri!r}: {exc}") from exc
            raise BlobUnavailable(f"cannot resolve {uri!r}: {exc}") from exc
        if not candidate.is_relative_to(self.root):
            raise BlobNotFound(f"{uri!r} resolves outside the blob root")
        return candidate

    def _check_root(self, uri: str) -> None:
        """Missing root = missing MOUNT, which is transient, not a 404.

        Without this, an unmounted volume answers "no such blob" for every
        record and the whole backlog is dead-lettered (and committed) in
        seconds instead of stopping the service.
        """
        if not self.root.is_dir():
            raise BlobUnavailable(
                f"blob root {self.root} is not a directory: the store is not mounted "
                f"(looking for {uri!r})"
            )

    def get(self, uri: str) -> bytes:
        path = self._resolve(uri)
        if not path.is_file():
            self._check_root(uri)
            raise BlobNotFound(f"no blob at {uri!r}")
        try:
            return path.read_bytes()
        except OSError as exc:  # unreadable while it exists: disk/mount problem
            raise BlobUnavailable(f"cannot read {uri!r}: {exc}") from exc

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        path = self._resolve(key)
        if path.suffix == "":
            path = path.with_suffix(_EXTENSIONS.get(content_type, ".bin"))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temporary sibling then rename: a reader that follows
            # the URI must never observe a half-written mask.
            tmp = path.with_name(path.name + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError as exc:
            if exc.errno in _PERMANENT_ERRNOS:
                # ENAMETOOLONG & co. depend on the key alone: retrying, and
                # retrying again after a restart, changes nothing.
                raise InvalidBlobKey(f"unusable blob key {key!r}: {exc}") from exc
            raise BlobUnavailable(f"cannot write {key!r}: {exc}") from exc
        return f"{self.scheme}://{path.relative_to(self.root).as_posix()}"


class InMemoryBlobStore:
    """Dict-backed store, for the tests and for a dry run without a volume."""

    def __init__(self, initial: dict[str, bytes] | None = None, scheme: str = "mem") -> None:
        self.blobs: dict[str, bytes] = dict(initial or {})
        self.scheme = scheme

    def get(self, uri: str) -> bytes:
        try:
            return self.blobs[uri]
        except KeyError as exc:
            raise BlobNotFound(f"no blob at {uri!r}") from exc

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        uri = key if "://" in key else f"{self.scheme}://{key}"
        self.blobs[uri] = data
        return uri


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
