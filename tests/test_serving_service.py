"""The inference loop, exercised without a broker, without torch, without a GPU
(chap. 5.4).

Everything the loop touches is behind a protocol, so the fakes below replace
Kafka (``FakeSource``/``FakeSink``), the object store (``InMemoryBlobStore``)
and the model (``ConstantEngine``). What is actually asserted is the part that
keeps a partition alive in production: what gets committed, what gets
dead-lettered, and in which order the flush and the commit happen.
"""

from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from PIL import Image

from memoire.serving import messages as msg
from memoire.serving.blobstore import BlobUnavailable, InMemoryBlobStore
from memoire.serving.ports import DamageInstance, InboundRecord, InferenceResult, ModelInfo
from memoire.serving.producer import publish_photo
from memoire.serving.service import (
    Outcome,
    ServiceConfig,
    ServiceDeps,
    handle_record,
    run_service,
)

INSPECTION = "NCE01/AB-123-CD/2026-08-24T09:12:03Z"
MASKS = "inspection.masks.v1"
DLQ = "inspection.photos.dlq.v1"
NOW = datetime(2026, 8, 24, 9, 12, 3, tzinfo=timezone.utc)


# -- fakes --------------------------------------------------------------------


class FakeSource:
    def __init__(self, batches, journal=None) -> None:
        self.batches = list(batches)
        self.committed: list[tuple[int, int]] = []
        self.closed = 0
        self.journal = journal if journal is not None else []

    def poll(self, timeout_ms: int = 1000, max_records: int = 8):
        return self.batches.pop(0) if self.batches else []

    def commit(self, record: InboundRecord) -> None:
        self.committed.append((record.partition, record.offset))
        self.journal.append(("commit", record.offset))

    def close(self) -> None:
        self.closed += 1


class FakeSink:
    def __init__(self, journal=None, fail_times: int = 0) -> None:
        self.sent: list[tuple[str, bytes | None, bytes, tuple]] = []
        self.flushes = 0
        self.closed = 0
        self.fail_times = fail_times
        self.journal = journal if journal is not None else []

    def send(self, topic, key, value, headers=()) -> None:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("broker unreachable")
        self.sent.append((topic, key, value, tuple(headers)))
        self.journal.append(("send", topic))

    def flush(self, timeout_s: float = 30.0) -> None:
        self.flushes += 1
        self.journal.append(("flush", None))

    def close(self) -> None:
        self.closed += 1

    def bodies(self, topic: str) -> list[dict]:
        return [json.loads(value) for name, _, value, _ in self.sent if name == topic]


class ConstantEngine:
    """A SegmentationEngine that needs no checkpoint (and no torch)."""

    def __init__(self, ratio: float = 0.02, raises: BaseException | None = None) -> None:
        self.info = ModelInfo(
            "unet", "binary", 64, 2, ("background", "damage"), "ckpt-sha", "runs/dev", 200
        )
        self.raises = raises
        self.calls = 0
        self.ratio = ratio

    def predict(self, image) -> InferenceResult:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[2:6, 2:8] = 1
        return InferenceResult(
            mask=mask,
            instances=[DamageInstance(1, "damage", 0.87, (2, 2, 6, 4), 24)],
            damage_pixel_ratio=self.ratio,
            inference_ms=12.5,
        )


class BrokenStore:
    """Object store that is up but failing — the transient case."""

    def get(self, uri: str) -> bytes:
        raise BlobUnavailable("mount gone")

    def put(self, key: str, data: bytes, content_type: str = "image/png") -> str:
        raise BlobUnavailable("mount gone")


# -- helpers ------------------------------------------------------------------


def png_bytes(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (200, 40, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def make_deps(engine=None, photo_store=None, sink=None, journal=None, **config_kwargs):
    return ServiceDeps(
        engine=engine or ConstantEngine(),
        photo_store=photo_store if photo_store is not None else InMemoryBlobStore(),
        mask_store=InMemoryBlobStore(),
        sink=sink or FakeSink(journal),
        config=ServiceConfig(MASKS, DLQ, backoff_s=0.0, **config_kwargs),
        now=lambda: NOW,
        sleep=lambda _s: None,
        new_id=lambda: "message-id",
    )


def photo_record(deps: ServiceDeps, offset: int = 10, data: bytes | None = None) -> InboundRecord:
    """Publish a real photo through the producer, hand it back as a record.

    Going through ``publish_photo`` rather than a hand-written JSON blob is
    what makes this an end-to-end check of the two halves of the wire format.
    """
    outbox = FakeSink()
    body = publish_photo(
        outbox,
        "inspection.photos.v1",
        photo_id="01J9ZC8X4K7Q2M5N",
        inspection_id=INSPECTION,
        agency_id="NCE01",
        data=data if data is not None else png_bytes(),
        content_type="image/png",
        store=deps.photo_store,
    )
    return InboundRecord("inspection.photos.v1", 3, offset, INSPECTION.encode(), body)


# -- nominal path -------------------------------------------------------------


def test_nominal_publishes_a_mask_keyed_like_the_photo() -> None:
    deps = make_deps()
    record = photo_record(deps)
    assert handle_record(record, deps) is Outcome.PUBLISHED

    topic, key, value, headers = deps.sink.sent[0]
    assert topic == MASKS
    # Co-partitioning: masks carry the same key as the photos they answer.
    assert key == INSPECTION.encode("utf-8")
    body = json.loads(value)
    assert body["photo_id"] == "01J9ZC8X4K7Q2M5N"
    assert body["model"]["checkpoint_sha256"] == "ckpt-sha"
    assert body["instances"][0]["bbox"] == [2, 2, 6, 4]
    assert body["source"]["offset"] == record.offset
    assert msg.header_value(headers, "schema") == b"inspection.mask"
    # The mask itself travelled by reference, and the blob exists.
    assert deps.mask_store.get(body["mask"]["uri"])


def test_inline_payload_is_accepted_without_an_object_store() -> None:
    deps = make_deps(photo_store=BrokenStore())
    outbox = FakeSink()
    body = publish_photo(
        outbox, "inspection.photos.v1", "p1", INSPECTION, "NCE01",
        data=png_bytes(), content_type="image/png", store=None,
    )
    record = InboundRecord("inspection.photos.v1", 0, 1, INSPECTION.encode(), body)
    assert handle_record(record, deps) is Outcome.PUBLISHED


def test_run_service_commits_after_flushing() -> None:
    """At-least-once hinges on this order: the broker acknowledges the mask
    before we admit having consumed the photo."""
    journal: list[tuple[str, object]] = []
    deps = make_deps(journal=journal)
    source = FakeSource([[photo_record(deps, offset=10)]], journal=journal)

    summary = run_service(source, deps, max_iterations=1)

    assert summary["n_published"] == 1
    assert source.committed == [(3, 10)]
    assert journal == [("send", MASKS), ("flush", None), ("commit", 10), ("flush", None)]
    assert source.closed == 1 and deps.sink.closed == 1


def test_run_service_stops_on_the_stop_flag_after_finishing_the_record() -> None:
    stop = {"value": False}
    deps = make_deps()
    records = [photo_record(deps, offset=i) for i in (1, 2)]
    source = FakeSource([records])

    def should_stop() -> bool:
        return stop["value"]

    original = deps.engine.predict

    def predict_then_stop(image):
        stop["value"] = True
        return original(image)

    deps.engine.predict = predict_then_stop
    summary = run_service(source, deps, should_stop, max_iterations=5)

    # The in-flight record is finished and committed; the next one is not.
    assert summary["n_published"] == 1
    assert source.committed == [(3, 1)]
    assert summary["stopped"] == "signal"


# -- permanent failures: dead-letter, then commit ------------------------------


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("invalid_json", b"{not json at all"),
        ("unknown_schema", json.dumps({"schema": "other", "schema_version": 1}).encode()),
    ],
)
def test_undecodable_message_is_dead_lettered_and_committed(name: str, value: bytes) -> None:
    deps = make_deps()
    source = FakeSource([[InboundRecord("inspection.photos.v1", 3, 7, None, value)]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["n_dead_lettered"] == 1 and summary["n_published"] == 0
    assert deps.sink.bodies(MASKS) == []
    dead = deps.sink.bodies(DLQ)[0]
    assert dead["error"]["stage"] == "decode"
    assert base64.b64decode(dead["original_value_b64"]) == value
    # Committed: a poison pill must never block its partition.
    assert source.committed == [(3, 7)]


def test_missing_blob_is_permanent() -> None:
    deps = make_deps()
    record = photo_record(deps)
    deps.photo_store.blobs.clear()  # retention expired before the topic's

    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    assert deps.sink.bodies(DLQ)[0]["error"]["stage"] == "fetch"


def test_checksum_mismatch_is_permanent() -> None:
    deps = make_deps()
    record = photo_record(deps)
    uri = json.loads(record.value)["payload"]["uri"]
    deps.photo_store.blobs[uri] = png_bytes(8, 8)  # not what the message announced

    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    assert deps.sink.bodies(DLQ)[0]["error"]["type"] == "ChecksumMismatch"


def test_unreadable_image_is_dead_lettered() -> None:
    deps = make_deps()
    record = photo_record(deps, data=png_bytes()[:40])  # truncated PNG

    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    body = deps.sink.bodies(DLQ)[0]
    assert body["error"]["stage"] == "decode_image"
    assert body["error"]["type"] == "UnreadableImageError"


def test_a_model_that_keeps_failing_dead_letters_after_max_attempts() -> None:
    """A systematic bug on one image must not freeze the partition — but it
    must stay visible in the DLQ."""
    engine = ConstantEngine(raises=RuntimeError("kernel exploded"))
    deps = make_deps(engine=engine, max_attempts=3)
    record = photo_record(deps)

    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    assert engine.calls == 3
    body = deps.sink.bodies(DLQ)[0]
    assert body["error"]["stage"] == "infer" and body["attempts"] == 3


# -- transient failures: no commit ---------------------------------------------


def test_unreachable_object_store_stops_without_committing() -> None:
    deps = make_deps(photo_store=BrokenStore(), max_attempts=2)
    good = make_deps()
    record = photo_record(good, offset=5)
    record = InboundRecord(record.topic, record.partition, 5, record.key, record.value)
    source = FakeSource([[record]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "transient_failure"
    assert source.committed == []          # re-read after restart
    assert deps.sink.bodies(DLQ) == []     # a transient outage is not a poison pill


def test_a_sink_that_recovers_within_the_retry_budget_still_publishes() -> None:
    sink = FakeSink(fail_times=1)
    deps = make_deps(sink=sink, max_attempts=3)
    record = photo_record(deps)

    assert handle_record(record, deps) is Outcome.PUBLISHED
    assert len(sink.sent) == 1


def test_a_dead_letter_that_cannot_be_published_is_not_committed() -> None:
    """A poison pill during a broker outage: publishing the dead letter fails,
    so the record must NOT be committed — committing would lose it entirely
    (no mask, no dead letter)."""
    sink = FakeSink(fail_times=99)
    deps = make_deps(sink=sink, max_attempts=2)
    record = InboundRecord("inspection.photos.v1", 3, 12, None, b"{not json")
    source = FakeSource([[record]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "transient_failure"
    assert source.committed == []


# -- poison pills that used to kill the loop ----------------------------------


def test_a_decompression_bomb_is_dead_lettered_not_fatal() -> None:
    """A 68-byte PNG announcing 30000x30000 raises PIL's
    DecompressionBombError, which inherits from Exception: it used to escape
    open_image, handle_record and run_service, so the process died without
    committing and restarted on the same offset — a permanently blocked
    partition, the exact poison pill the taxonomy claims to prevent."""
    from test_serving_preprocess import _decompression_bomb_png

    deps = make_deps()
    record = photo_record(deps, offset=21, data=_decompression_bomb_png())
    source = FakeSource([[record]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["n_dead_lettered"] == 1
    body = deps.sink.bodies(DLQ)[0]
    assert body["error"]["stage"] == "decode_image"
    assert body["error"]["type"] == "UnreadableImageError"
    assert source.committed == [(3, 21)]  # the partition moves on


def test_no_exception_escapes_handle_record() -> None:
    """Last-resort net: whatever breaks, the record leaves with a verdict
    instead of an exception. Here the clock itself raises — a failure at a
    stage nobody thought to wrap, which is precisely the case that matters."""
    deps = make_deps()
    record = photo_record(deps, offset=3)
    calls = {"n": 0}

    def _clock_that_breaks_once():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("clock is on fire")
        return NOW

    deps.now = _clock_that_breaks_once
    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    assert deps.sink.bodies(DLQ)[0]["error"]["type"] == "RuntimeError"


def test_a_verdict_that_cannot_be_reached_is_not_a_commit() -> None:
    """And if even the dead letter cannot be built, the record is left
    uncommitted rather than the loop being killed."""
    deps = make_deps()
    record = photo_record(deps, offset=4)

    def _broken_clock():
        raise RuntimeError("clock is on fire")

    deps.now = _broken_clock
    assert handle_record(record, deps) is Outcome.RETRY


def test_the_loop_survives_a_handle_record_that_raises(monkeypatch) -> None:
    """Belt and braces around the belt and braces: even if handle_record ever
    lets something through, the loop stops without committing instead of
    dying on an uncommitted record."""
    from memoire.serving import service as service_module

    deps = make_deps()
    source = FakeSource([[photo_record(deps, offset=8)]])

    def _explode(record, deps):
        raise RuntimeError("should never happen")

    monkeypatch.setattr(service_module, "handle_record", _explode)
    summary = service_module.run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "transient_failure"
    assert source.committed == []
    assert source.closed == 1  # the consumer still left the group cleanly


# -- resource exhaustion is never a poison pill --------------------------------


class _CudaOutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError (same name, same base)."""

    __name__ = "OutOfMemoryError"


def _oom() -> BaseException:
    exc = _CudaOutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    exc.__class__.__name__ = "OutOfMemoryError"
    return exc


@pytest.mark.parametrize("exception", [MemoryError("no room"), _oom()])
def test_an_out_of_memory_model_is_retried_not_dead_lettered(exception) -> None:
    """A 60 s memory spike on the box would otherwise bury several hundred
    perfectly valid photos in the DLQ, committed and unreplayable — while a
    plain stop (exit 75) replays them intact."""
    deps = make_deps(engine=ConstantEngine(raises=exception), max_attempts=2)
    record = photo_record(deps)
    source = FakeSource([[record]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "transient_failure"
    assert deps.sink.bodies(DLQ) == []
    assert source.committed == []


# -- a publish stage that can never succeed ------------------------------------


def test_a_hostile_photo_id_still_publishes(tmp_path) -> None:
    """The mask key is built from message fields. A photo_id containing a NUL
    (legal JSON) or 300 characters used to make FilesystemBlobStore.put fail
    identically on every replay: reported transient, never committed, blocked
    partition. The ids are slugified instead."""
    from memoire.serving.blobstore import FilesystemBlobStore

    for photo_id in ("x\x00y", "b" * 300, "../../etc/passwd"):
        deps = make_deps()
        deps.mask_store = FilesystemBlobStore(tmp_path / "masks")
        outbox = FakeSink()
        body = publish_photo(
            outbox, "inspection.photos.v1", photo_id, INSPECTION, "NCE01",
            data=png_bytes(), content_type="image/png", store=None,
        )
        record = InboundRecord("inspection.photos.v1", 0, 1, INSPECTION.encode(), body)

        assert handle_record(record, deps) is Outcome.PUBLISHED, photo_id
        uri = json.loads(deps.sink.sent[0][2])["mask"]["uri"]
        assert deps.mask_store.get(uri)


class MessageSizeTooLargeError(Exception):
    """Same class name as kafka-python's, which is how the loop recognises it
    without importing kafka (tests/test_serving_isolation.py forbids that)."""


class RejectingSink(FakeSink):
    """A broker that refuses records over ``limit`` bytes, synchronously and
    before compression — kafka-python's own behaviour.

    ``topics`` restricts the refusal to some topics, the way a per-topic
    ``max.message.bytes`` does.
    """

    def __init__(self, limit: int, topics=None, journal=None) -> None:
        super().__init__(journal)
        self.limit = limit
        self.topics = topics
        self.refused = 0

    def send(self, topic, key, value, headers=()) -> None:
        if (self.topics is None or topic in self.topics) and len(value) > self.limit:
            self.refused += 1
            raise MessageSizeTooLargeError(f"{len(value)} bytes > {self.limit}")
        super().send(topic, key, value, headers)


def test_a_message_too_large_to_publish_is_dead_lettered() -> None:
    """A permanent rejection by the producer at the publish stage must not be
    reported transient: retrying, restarting and re-reading changes nothing,
    so the offset would never move again."""
    sink = RejectingSink(limit=200, topics={MASKS})
    deps = make_deps(sink=sink, max_attempts=3)
    record = photo_record(deps, offset=9)

    assert handle_record(record, deps) is Outcome.DEAD_LETTERED
    # Rejected once, not once per attempt: permanent means permanent.
    assert sink.refused == 1
    assert deps.sink.bodies(DLQ)[0]["error"]["stage"] == "publish"


def test_a_dead_letter_too_large_falls_back_to_a_payload_free_one() -> None:
    """The service could mint messages it was unable to dead-letter: 512 KiB
    inline is legal, its base64 dead letter is not. Truncation handles the
    normal case; this is the belt-and-braces one, with a broker refusing
    anything above the size of the description itself."""
    sink = RejectingSink(limit=700, topics={DLQ})
    deps = make_deps(sink=sink, max_attempts=2)
    value = b'{"schema":"nope","padding":"' + b"p" * 800 + b'"}'
    record = InboundRecord("inspection.photos.v1", 3, 12, None, value)
    source = FakeSource([[record]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["n_dead_lettered"] == 1
    dead = sink.bodies(DLQ)[0]
    assert dead["original_value_b64"] == ""
    assert dead["original_value_bytes"] == len(value)
    assert source.committed == [(3, 12)]


# -- the sink's promise, and the streak cap ------------------------------------


def test_a_flush_that_reports_a_lost_record_blocks_the_commit() -> None:
    """`flush` is what stands between 'published' and 'committed'. If it says
    a record was never acknowledged, committing would consume the photo with
    no mask and no dead letter, and no replay would be possible."""
    class LosingSink(FakeSink):
        def flush(self, timeout_s: float = 30.0) -> None:
            self.flushes += 1
            raise ConnectionError("2/2 records were not acknowledged")

    deps = make_deps(sink=LosingSink())
    source = FakeSource([[photo_record(deps, offset=6)]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "transient_failure"
    assert source.committed == []


def test_a_whole_batch_of_dead_letters_stops_the_service() -> None:
    """An unmounted volume answers 'not found' for every record. Dead-lettering
    them one by one drains the backlog in seconds, committed and silent; a
    streak is an outage and must be as loud as a crash."""
    deps = make_deps(max_consecutive_dead_letters=3)
    records = [
        InboundRecord("inspection.photos.v1", 3, offset, None, b"{not json")
        for offset in range(6)
    ]
    source = FakeSource([records])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "dead_letter_streak"
    assert summary["n_dead_lettered"] == 3        # not the whole backlog
    assert source.committed == [(3, 0), (3, 1), (3, 2)]


def test_the_streak_is_broken_by_any_success() -> None:
    deps = make_deps(max_consecutive_dead_letters=2)
    good = photo_record(deps, offset=1)
    bad = InboundRecord("inspection.photos.v1", 3, 0, None, b"{not json")
    source = FakeSource([[bad, good, bad]])

    summary = run_service(source, deps, max_iterations=1)

    assert summary["stopped"] == "max_iterations"
    assert summary["n_dead_lettered"] == 2 and summary["n_published"] == 1
