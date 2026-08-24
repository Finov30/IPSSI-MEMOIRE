"""Wire format of the three inference topics (chap. 5.4), as pure functions.

Topics and schemas:

- ``inspection.photos.v1``  -> :func:`decode_photo`   (upstream, one photo)
- ``inspection.masks.v1``   -> :func:`encode_mask`    (downstream, one mask)
- ``inspection.photos.dlq.v1`` -> :func:`encode_dlq`  (permanent failures)

Two decisions are materialised here and nowhere else.

*Claim-check transport.* A phone photo weighs 2-5 MiB; carrying the bytes
inline would force ``max.message.bytes`` up to ~10 MiB (broker memory,
replication cost, slower rebalances) and base64 would add another 33%. The
photo therefore travels as a reference (URI + sha256 + size) into an object
store, and the broker keeps its 1 MiB default — the configuration itself is
the proof of the design. The discriminated union ``payload.kind`` still
allows ``inline`` from v1 on (capped at :data:`MAX_INLINE_BYTES`), for an
agency terminal with no direct access to the object store; the degraded mode
is in the schema from the start rather than forcing a v2 later.

*Version in the topic name.* An incompatible schema change creates
``inspection.photos.v2`` and both coexist during the migration, so
:func:`decode_photo` can reject an unknown ``schema_version`` outright.

Every failure raised here is PERMANENT (:class:`MessageError`): a malformed
message will never become valid by being retried, so it goes to the DLQ and
its offset is committed. Blocking the partition on a poison pill is the
failure mode this rule exists to prevent.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memoire.serving.ports import InboundRecord, InferenceResult, ModelInfo

PHOTO_SCHEMA = "inspection.photo"
MASK_SCHEMA = "inspection.mask"
DLQ_SCHEMA = "inspection.photo.dlq"
SCHEMA_VERSION = 1

#: Hard ceiling on the degraded inline mode. Above this the message would
#: approach the broker's 1 MiB limit once base64-expanded: refused at decode
#: time (permanent), rather than letting the producer discover it as a
#: RecordTooLargeError.
MAX_INLINE_BYTES = 512 * 1024

#: Above this, a dead letter carries a *description* of the original message
#: rather than the message. base64 adds 33%, so a 512 KiB inline photo (which
#: :data:`MAX_INLINE_BYTES` allows) yields an ~800 KiB record whose dead
#: letter would be ~1.09 MiB — over the 1 MiB ``max.message.bytes`` of the
#: topics AND over kafka-python's ``max_request_size``, which is checked
#: synchronously and BEFORE compression. The service would then be minting
#: messages it is structurally unable to dead-letter: a permanent error with
#: no exit, i.e. a blocked partition.
DLQ_MAX_PAYLOAD_BYTES = 256 * 1024

#: How much of an over-sized payload is still worth keeping inline, to
#: identify it by eye in ``kafka-console-consumer``.
DLQ_PREVIEW_BYTES = 4 * 1024

PAYLOAD_KINDS = ("reference", "inline")


class MessageError(ValueError):
    """Permanent decoding/validation error: dead-letter, never retry."""


class ChecksumMismatch(MessageError):
    """The fetched blob does not match the sha256 announced in the message."""


@dataclass(frozen=True)
class PhotoPayload:
    """Where the pixels are: an object-store reference, or inline bytes."""

    kind: str
    content_type: str = "image/jpeg"
    uri: str | None = None
    data: bytes | None = None
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class PhotoMessage:
    photo_id: str
    inspection_id: str
    agency_id: str
    payload: PhotoPayload
    stage: str | None = None
    captured_at: str | None = None
    raw: dict[str, Any] | None = None


# -- decoding -----------------------------------------------------------------


def _require(raw: dict, key: str) -> Any:
    value = raw.get(key)
    if value is None or (isinstance(value, str) and not value):
        raise MessageError(f"missing required field '{key}'")
    return value


def _decode_payload(raw: Any) -> PhotoPayload:
    if not isinstance(raw, dict):
        raise MessageError("'payload' must be an object")
    kind = raw.get("kind")
    if kind not in PAYLOAD_KINDS:
        raise MessageError(f"unknown payload.kind {kind!r}, expected one of {PAYLOAD_KINDS}")
    content_type = raw.get("content_type") or "image/jpeg"
    sha256 = raw.get("sha256")
    size_bytes = raw.get("size_bytes")
    if kind == "reference":
        uri = _require(raw, "uri")
        if not isinstance(uri, str):
            raise MessageError("'payload.uri' must be a string")
        return PhotoPayload("reference", content_type, uri=uri, size_bytes=size_bytes,
                            sha256=sha256)
    encoded = _require(raw, "data_b64")
    if not isinstance(encoded, str):
        raise MessageError("'payload.data_b64' must be a string")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MessageError(f"invalid base64 in payload.data_b64: {exc}") from exc
    if len(data) > MAX_INLINE_BYTES:
        raise MessageError(
            f"inline payload is {len(data)} bytes, over the {MAX_INLINE_BYTES} limit "
            "(publish it by reference instead)"
        )
    return PhotoPayload("inline", content_type, data=data, size_bytes=size_bytes, sha256=sha256)


def decode_photo(value: bytes) -> PhotoMessage:
    """Parse and validate one ``inspection.photos.v1`` message body."""
    try:
        raw = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageError(f"message body is not UTF-8 JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise MessageError("message body must be a JSON object")
    schema = raw.get("schema")
    if schema != PHOTO_SCHEMA:
        raise MessageError(f"unexpected schema {schema!r}, expected {PHOTO_SCHEMA!r}")
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise MessageError(
            f"unsupported schema_version {version!r} (this service speaks v{SCHEMA_VERSION}; "
            "an incompatible schema gets its own topic)"
        )
    return PhotoMessage(
        photo_id=str(_require(raw, "photo_id")),
        inspection_id=str(_require(raw, "inspection_id")),
        agency_id=str(raw.get("agency_id") or "unknown"),
        payload=_decode_payload(_require(raw, "payload")),
        stage=raw.get("stage"),
        captured_at=raw.get("captured_at"),
        raw=raw,
    )


def partition_key(message: PhotoMessage) -> bytes:
    """Partition key = ``inspection_id``, never ``agency_id``.

    Two reasons, both load-bearing (chap. 5.4). Anti-skew: a handful of large
    agencies concentrate the traffic, so keying by agency creates hot
    partitions and caps parallelism at the number of agencies, while
    ``inspection_id`` has high cardinality and spreads evenly. Ordering: the
    business guarantee needed is per inspection, not per agency — every photo
    of one inspection lands on one partition, and the downstream topic is
    keyed identically so photos and masks are co-partitioned.
    """
    return message.inspection_id.encode("utf-8")


def verify_sha256(data: bytes, expected: str | None) -> None:
    """Raise :class:`ChecksumMismatch` if ``data`` is not what was announced.

    A blob that does not match its checksum is corrupt at the source: retrying
    the fetch would return the same bytes, so this is permanent.
    """
    if not expected:
        return
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected.removeprefix("sha256:"):
        raise ChecksumMismatch(f"blob sha256 is {digest}, message announced {expected}")


# -- encoding -----------------------------------------------------------------


def headers_for(schema: str, message_id: str, trace_id: str | None = None) -> list[tuple[str, bytes]]:
    """Kafka headers common to every message we publish."""
    headers = [
        ("schema", schema.encode("utf-8")),
        ("schema_version", str(SCHEMA_VERSION).encode("utf-8")),
        ("content-type", b"application/json"),
        ("message_id", message_id.encode("utf-8")),
    ]
    if trace_id:
        headers.append(("trace_id", trace_id.encode("utf-8")))
    return headers


def _isoformat(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def encode_photo(
    photo_id: str,
    inspection_id: str,
    agency_id: str,
    payload: PhotoPayload,
    captured_at: datetime | str | None = None,
    stage: str | None = None,
    capture: dict[str, Any] | None = None,
) -> bytes:
    """Build one ``inspection.photos.v1`` body (producer side, chap. 5.4)."""
    if payload.kind not in PAYLOAD_KINDS:
        raise MessageError(f"unknown payload.kind {payload.kind!r}")
    body: dict[str, Any] = {
        "schema": PHOTO_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "photo_id": photo_id,
        "inspection_id": inspection_id,
        "agency_id": agency_id,
        "stage": stage,
        "captured_at": (
            _isoformat(captured_at) if isinstance(captured_at, datetime) else captured_at
        ),
    }
    if payload.kind == "reference":
        if not payload.uri:
            raise MessageError("a reference payload needs a uri")
        body["payload"] = {
            "kind": "reference",
            "uri": payload.uri,
            "content_type": payload.content_type,
            "size_bytes": payload.size_bytes,
            "sha256": payload.sha256,
        }
    else:
        data = payload.data or b""
        if len(data) > MAX_INLINE_BYTES:
            raise MessageError(
                f"inline payload is {len(data)} bytes, over the {MAX_INLINE_BYTES} limit"
            )
        body["payload"] = {
            "kind": "inline",
            "data_b64": base64.b64encode(data).decode("ascii"),
            "content_type": payload.content_type,
            "size_bytes": len(data),
            "sha256": payload.sha256 or hashlib.sha256(data).hexdigest(),
        }
    if capture:
        body["capture"] = capture
    return _dumps(body)


def encode_mask(
    message: PhotoMessage,
    result: InferenceResult,
    model: ModelInfo,
    mask_uri: str,
    source: InboundRecord,
    received_at: datetime,
    produced_at: datetime,
) -> bytes:
    """Build one ``inspection.masks.v1`` body.

    The instance list is inlined (a few hundred bytes) so a dashboard or a
    billing rule can decide without fetching the mask blob; the mask itself
    travels by reference, like the photo. Boxes are in ORIGINAL pixels.
    """
    height, width = (int(result.mask.shape[0]), int(result.mask.shape[1]))
    instances = sorted(result.instances, key=lambda inst: inst.score, reverse=True)
    area_total = float(width * height) or 1.0
    body: dict[str, Any] = {
        "schema": MASK_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "photo_id": message.photo_id,
        "inspection_id": message.inspection_id,
        "agency_id": message.agency_id,
        "status": "ok",
        "model": {
            "name": model.name,
            "mode": model.mode,
            "input_size": model.input_size,
            "checkpoint_sha256": model.checkpoint_sha256,
            "run_id": model.run_id,
            "iteration": model.iteration,
        },
        "mask": {
            "kind": "reference",
            "uri": mask_uri,
            "encoding": "png-palette-u8",
            "width": width,
            "height": height,
            "class_values": {str(i): name for i, name in enumerate(model.class_names)},
        },
        "instances": [
            {
                "class": inst.class_name,
                "class_id": inst.class_id,
                "score": round(float(inst.score), 4),
                "bbox": list(inst.bbox),
                "area_px": int(inst.area_px),
                "area_ratio": round(int(inst.area_px) / area_total, 6),
            }
            for inst in instances
        ],
        "summary": {
            "n_instances": len(instances),
            "damage_pixel_ratio": round(float(result.damage_pixel_ratio), 6),
            "max_score": round(float(max((i.score for i in instances), default=0.0)), 4),
        },
        "timings": {
            "received_at": _isoformat(received_at),
            "inference_ms": round(float(result.inference_ms), 2),
            "produced_at": _isoformat(produced_at),
        },
        "source": {
            "topic": source.topic,
            "partition": source.partition,
            "offset": source.offset,
        },
    }
    return _dumps(body)


def encode_dlq(
    record: InboundRecord,
    error: BaseException,
    stage: str,
    attempts: int,
    now: datetime,
    max_payload_bytes: int = DLQ_MAX_PAYLOAD_BYTES,
) -> bytes:
    """Build one ``inspection.photos.dlq.v1`` body.

    The original key/value are kept verbatim (base64): once the bug is fixed,
    a replay script pushes them back onto ``inspection.photos.v1`` unchanged.
    ``stage`` is one of ``decode``/``fetch``/``decode_image``/``infer``/``publish``.

    Above ``max_payload_bytes`` the payload is replaced by its size, its
    sha256 and a :data:`DLQ_PREVIEW_BYTES` prefix. Without that cap the DLQ
    body of a large message is *bigger* than the message itself (base64 costs
    33%), so it is refused by ``max.message.bytes`` — and a permanent error
    that cannot be dead-lettered is a permanently blocked partition. The
    sha256 keeps the record identifiable in the source topic, where the bytes
    still are until retention expires.
    """
    body = {
        "schema": DLQ_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "failed_at": _isoformat(now),
        "attempts": attempts,
        "error": {
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error)[:2000],
        },
        "source": {
            "topic": record.topic,
            "partition": record.partition,
            "offset": record.offset,
        },
        "original_key_b64": _b64_capped(record.key, DLQ_PREVIEW_BYTES),
        "original_value_bytes": len(record.value),
        "original_value_sha256": hashlib.sha256(record.value).hexdigest(),
    }
    if len(record.value) <= max_payload_bytes:
        body["original_value_b64"] = base64.b64encode(record.value).decode("ascii")
        body["original_value_truncated"] = False
    else:
        preview = min(DLQ_PREVIEW_BYTES, max_payload_bytes)
        body["original_value_b64"] = base64.b64encode(record.value[:preview]).decode("ascii")
        body["original_value_truncated"] = True
    return _dumps(body)


def _b64_capped(data: bytes | None, limit: int) -> str | None:
    """base64 of ``data``, truncated to ``limit`` raw bytes (keys are small)."""
    if data is None:
        return None
    return base64.b64encode(data[:limit]).decode("ascii")


def decode_mask(value: bytes) -> dict[str, Any]:
    """Parse a mask message back (consumers downstream, and the tests)."""
    try:
        return json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MessageError(f"mask body is not UTF-8 JSON: {exc}") from exc


def _dumps(body: dict[str, Any]) -> bytes:
    # separators: no wasted whitespace on the wire; ensure_ascii=False keeps
    # accented agency names readable in kafka-console-consumer output.
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def header_value(headers: Sequence[tuple[str, bytes]], name: str) -> bytes | None:
    """First value of header ``name``, or None."""
    for key, value in headers:
        if key == name:
            return value
    return None
