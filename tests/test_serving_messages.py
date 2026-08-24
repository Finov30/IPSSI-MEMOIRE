"""Wire format of the inference topics (chap. 5.4): pure functions, no broker."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from memoire.serving import messages as msg
from memoire.serving.ports import DamageInstance, InboundRecord, InferenceResult, ModelInfo

NOW = datetime(2026, 8, 24, 9, 12, 3, tzinfo=timezone.utc)
INSPECTION = "NCE01/AB-123-CD/2026-08-24T09:12:03Z"


def photo_body(**overrides) -> bytes:
    body = {
        "schema": "inspection.photo",
        "schema_version": 1,
        "photo_id": "01J9ZC8X4K7Q2M5N",
        "inspection_id": INSPECTION,
        "agency_id": "NCE01",
        "stage": "checkout",
        "captured_at": "2026-08-24T09:12:03Z",
        "payload": {
            "kind": "reference",
            "uri": "file://photos/a.jpg",
            "content_type": "image/jpeg",
            "size_bytes": 3145728,
            "sha256": "e3b0c442",
        },
    }
    body.update(overrides)
    return json.dumps(body).encode("utf-8")


def test_decode_photo_reference() -> None:
    message = msg.decode_photo(photo_body())
    assert message.photo_id == "01J9ZC8X4K7Q2M5N"
    assert message.payload.kind == "reference"
    assert message.payload.uri == "file://photos/a.jpg"
    assert msg.partition_key(message) == INSPECTION.encode("utf-8")


def test_decode_photo_inline() -> None:
    data = b"\xff\xd8fake jpeg"
    message = msg.decode_photo(
        photo_body(payload={"kind": "inline", "data_b64": base64.b64encode(data).decode()})
    )
    assert message.payload.kind == "inline"
    assert message.payload.data == data


@pytest.mark.parametrize(
    "value",
    [
        b"{not json",
        b"[]",
        json.dumps({"schema": "other", "schema_version": 1}).encode(),
    ],
)
def test_decode_photo_rejects_malformed_bodies(value: bytes) -> None:
    with pytest.raises(msg.MessageError):
        msg.decode_photo(value)


def test_decode_photo_rejects_an_unsupported_schema_version() -> None:
    with pytest.raises(msg.MessageError, match="schema_version"):
        msg.decode_photo(photo_body(schema_version=2))


def test_decode_photo_rejects_a_missing_field() -> None:
    body = json.loads(photo_body())
    del body["photo_id"]
    with pytest.raises(msg.MessageError, match="photo_id"):
        msg.decode_photo(json.dumps(body).encode())


def test_decode_photo_rejects_an_unknown_payload_kind() -> None:
    with pytest.raises(msg.MessageError, match="payload.kind"):
        msg.decode_photo(photo_body(payload={"kind": "carrier-pigeon"}))


def test_decode_photo_rejects_an_oversized_inline_payload() -> None:
    """The degraded mode exists, but it must not approach the broker's 1 MiB."""
    data = b"x" * (msg.MAX_INLINE_BYTES + 1)
    with pytest.raises(msg.MessageError, match="inline payload"):
        msg.decode_photo(
            photo_body(payload={"kind": "inline", "data_b64": base64.b64encode(data).decode()})
        )


def test_verify_sha256() -> None:
    data = b"pixels"
    msg.verify_sha256(data, hashlib.sha256(data).hexdigest())
    msg.verify_sha256(data, "sha256:" + hashlib.sha256(data).hexdigest())
    msg.verify_sha256(data, None)
    with pytest.raises(msg.ChecksumMismatch):
        msg.verify_sha256(data, "deadbeef")


def test_encode_photo_round_trips_through_decode_photo() -> None:
    payload = msg.PhotoPayload("inline", "image/png", data=b"\x89PNG...")
    body = msg.encode_photo("p1", INSPECTION, "NCE01", payload, captured_at=NOW)
    message = msg.decode_photo(body)
    assert message.payload.data == b"\x89PNG..."
    assert message.captured_at.startswith("2026-08-24T09:12:03")


def _result() -> InferenceResult:
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:10, 5:15] = 1
    return InferenceResult(
        mask=mask,
        instances=[
            DamageInstance(1, "damage", 0.42, (5, 5, 10, 5), 50),
            DamageInstance(1, "damage", 0.91, (20, 20, 4, 4), 16),
        ],
        damage_pixel_ratio=50 / 1200,
        inference_ms=143.21,
    )


def test_encode_mask_publishes_native_geometry_and_sorted_instances() -> None:
    message = msg.decode_photo(photo_body())
    info = ModelInfo("unet", "binary", 512, 2, ("background", "damage"), "abc123", "runs/dev", 200)
    record = InboundRecord("inspection.photos.v1", 3, 918273, b"k", b"{}")
    body = json.loads(msg.encode_mask(message, _result(), info, "file://masks/a.png", record,
                                      NOW, NOW))
    assert body["schema"] == "inspection.mask"
    assert body["mask"]["width"] == 40 and body["mask"]["height"] == 30
    assert body["mask"]["class_values"] == {"0": "background", "1": "damage"}
    assert [inst["score"] for inst in body["instances"]] == [0.91, 0.42]
    assert body["summary"]["n_instances"] == 2
    assert body["source"] == {"topic": "inspection.photos.v1", "partition": 3, "offset": 918273}
    assert body["model"]["checkpoint_sha256"] == "abc123"


def test_encode_dlq_keeps_the_original_bytes_verbatim() -> None:
    """Replayability is the whole point of the DLQ: after a fix, the same bytes
    are pushed back onto the upstream topic unchanged."""
    record = InboundRecord("inspection.photos.v1", 3, 918274, b"key", b"\x00\x01garbage")
    body = json.loads(msg.encode_dlq(record, ValueError("boom"), "decode", 1, NOW))
    assert base64.b64decode(body["original_value_b64"]) == record.value
    assert base64.b64decode(body["original_key_b64"]) == record.key
    assert body["error"] == {"stage": "decode", "type": "ValueError", "message": "boom"}
    assert body["source"]["offset"] == 918274


def test_headers_carry_the_schema_and_version() -> None:
    headers = msg.headers_for(msg.MASK_SCHEMA, "mid-1", "trace-1")
    assert msg.header_value(headers, "schema") == b"inspection.mask"
    assert msg.header_value(headers, "schema_version") == b"1"
    assert msg.header_value(headers, "trace_id") == b"trace-1"
    assert msg.header_value(headers, "absent") is None


def test_encode_dlq_truncates_a_payload_the_broker_would_refuse() -> None:
    """The service must never mint a message it cannot dead-letter.

    The inline mode accepts MAX_INLINE_BYTES; base64 in the dead letter costs
    another 33%, and the topics cap records at 1 MiB. Without a cap here, the
    dead letter of a big-but-legal message is rejected by the producer, the
    record is never committed, and the partition is blocked forever.
    """
    value = b"x" * (msg.MAX_INLINE_BYTES + 1)
    record = InboundRecord("inspection.photos.v1", 0, 7, b"key", value)
    encoded = msg.encode_dlq(record, ValueError("too big"), "decode", 1, NOW)

    assert len(encoded) < 1024 * 1024
    body = json.loads(encoded)
    assert body["original_value_truncated"] is True
    assert body["original_value_bytes"] == len(value)
    # The bytes are still identifiable: the photo itself is in the source
    # topic until its retention expires.
    assert body["original_value_sha256"] == hashlib.sha256(value).hexdigest()
    assert base64.b64decode(body["original_value_b64"]) == value[: msg.DLQ_PREVIEW_BYTES]


def test_encode_dlq_can_drop_the_payload_entirely() -> None:
    """Last resort when even a truncated dead letter is refused."""
    record = InboundRecord("inspection.photos.v1", 0, 7, b"key", b"x" * 4096)
    body = json.loads(
        msg.encode_dlq(record, ValueError("no"), "decode", 1, NOW, max_payload_bytes=0)
    )
    assert body["original_value_b64"] == ""
    assert body["original_value_truncated"] is True
    assert body["original_value_bytes"] == 4096
