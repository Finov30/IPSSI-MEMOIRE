"""Upstream producer: publishing inspection photos (chap. 5.4).

This is the agency-terminal side of the path — in production a mobile app, in
the lab this module plus ``scripts/publish_photos.py``, which is what makes
the whole chain demonstrable (publish a folder of photos, watch the masks
come out).

It writes the blob first and the message second: the claim-check must be
resolvable by the time a consumer reads the reference. The reverse order is a
race — the inference service would fetch a URI whose object does not exist
yet and (correctly, by its own error taxonomy) dead-letter a perfectly valid
photo.

Like the consumer, it only knows :class:`~memoire.serving.ports.MessageSink`,
so ``tests/test_serving_service.py`` publishes with a fake sink.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from memoire.serving import messages as msg
from memoire.serving.blobstore import BlobStore, sha256_hex
from memoire.serving.messages import MAX_INLINE_BYTES, PhotoPayload
from memoire.serving.ports import MessageSink

logger = logging.getLogger("memoire.serving.producer")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CONTENT_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".bmp": "image/bmp", ".webp": "image/webp",
}


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def publish_photo(
    sink: MessageSink,
    topic: str,
    photo_id: str,
    inspection_id: str,
    agency_id: str,
    data: bytes,
    content_type: str = "image/jpeg",
    store: BlobStore | None = None,
    inline: bool = False,
    captured_at: datetime | None = None,
    stage: str | None = None,
) -> bytes:
    """Publish one photo; return the message body that was sent.

    ``store=None`` forces the degraded inline mode (an agency terminal with no
    direct access to the object store), capped at
    :data:`~memoire.serving.messages.MAX_INLINE_BYTES`.
    """
    digest = sha256_hex(data)
    if inline or store is None:
        if len(data) > MAX_INLINE_BYTES:
            raise msg.MessageError(
                f"{photo_id}: {len(data)} bytes is too large for the inline mode "
                f"({MAX_INLINE_BYTES} max) — publish it by reference"
            )
        payload = PhotoPayload("inline", content_type, data=data, sha256=digest)
    else:
        name = photo_id if Path(photo_id).suffix else f"{photo_id}.jpg"
        uri = store.put(f"photos/{inspection_id}/{name}", data, content_type)
        payload = PhotoPayload(
            "reference", content_type, uri=uri, size_bytes=len(data), sha256=digest
        )
    body = msg.encode_photo(
        photo_id=photo_id,
        inspection_id=inspection_id,
        agency_id=agency_id,
        payload=payload,
        captured_at=captured_at or datetime.now(timezone.utc),
        stage=stage,
    )
    sink.send(
        topic,
        inspection_id.encode("utf-8"),  # same key rule as messages.partition_key
        body,
        msg.headers_for(msg.PHOTO_SCHEMA, str(uuid4())),
    )
    return body


def iter_images(root: Path) -> Iterator[Path]:
    """Every image file under ``root``, in a stable (sorted) order."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def publish_directory(
    sink: MessageSink,
    topic: str,
    images: Iterable[Path],
    inspection_id: str,
    agency_id: str,
    store: BlobStore | None = None,
    inline: bool = False,
    limit: int | None = None,
) -> int:
    """Publish a batch of image files; return how many were sent."""
    sent = 0
    for path in images:
        if limit is not None and sent >= limit:
            break
        publish_photo(
            sink,
            topic,
            photo_id=path.stem,
            inspection_id=inspection_id,
            agency_id=agency_id,
            data=path.read_bytes(),
            content_type=content_type_for(path),
            store=store,
            inline=inline,
        )
        sent += 1
        logger.info("published %s (%s)", path.name, inspection_id)
    sink.flush()
    return sent
