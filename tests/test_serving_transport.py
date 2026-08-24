"""The two adapters between the loop and the outside world (chap. 5.4).

What is tested here is not "does it store bytes" but the error taxonomy, since
that is what decides whether an offset is committed:

- the object store must tell "this blob is missing" (permanent, dead letter)
  apart from "the store is not mounted" (transient, stop and restart);
- the Kafka sink must tell "the broker acknowledged the mask" apart from "the
  broker never did", because a commit follows immediately.

No broker is involved: the producer is a fake reproducing the *contract* of
kafka-python's ``KafkaProducer`` — ``send`` returns a future, and ``flush``
does NOT raise when a batch has definitively failed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Through the module, never `from ... import`: tests/test_serving_isolation.py
# reloads kafka_client to prove it imports without the library, which rebinds
# the classes — a name captured at import time would no longer be the one the
# sink raises.
from memoire.serving import kafka_client
from memoire.serving.blobstore import (
    MAX_SEGMENT_LEN,
    BlobNotFound,
    BlobUnavailable,
    FilesystemBlobStore,
    InvalidBlobKey,
    safe_key_segment,
)

# -- object store: missing blob vs missing mount -------------------------------


def test_a_missing_blob_is_permanent(tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path)
    with pytest.raises(BlobNotFound):
        store.get("file://photos/absent.jpg")


def test_an_unmounted_root_is_transient_not_a_404(tmp_path: Path) -> None:
    """The failure mode this distinction exists for: with the volume missing,
    every record answers 'not found'. Permanent would drain a 10 000-photo
    backlog into the DLQ, offsets committed, in seconds and without an alarm."""
    store = FilesystemBlobStore(tmp_path / "photos")  # never created
    with pytest.raises(BlobUnavailable, match="not mounted"):
        store.get("file://photos/a.jpg")


def test_a_key_with_a_nul_byte_is_permanent(tmp_path: Path) -> None:
    """``Path.resolve`` raises ValueError on an embedded NUL — legal in JSON,
    so it comes straight off the topic. Left unclassified it was retried,
    reported transient, never committed: a blocked partition."""
    store = FilesystemBlobStore(tmp_path)
    with pytest.raises(InvalidBlobKey):
        store.get("file://photos/a\x00b.jpg")
    with pytest.raises(InvalidBlobKey):
        store.put("masks/a\x00b.png", b"x")


def test_a_key_that_is_too_long_is_permanent(tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path)
    with pytest.raises(InvalidBlobKey):
        store.put("masks/" + "n" * 400 + ".png", b"x")


def test_escaping_the_root_is_still_refused(tmp_path: Path) -> None:
    store = FilesystemBlobStore(tmp_path / "root")
    (tmp_path / "root").mkdir()
    (tmp_path / "secret.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(BlobNotFound, match="outside"):
        store.get("file://../secret.txt")


# -- key slugification ---------------------------------------------------------


def test_safe_key_segment_leaves_a_sane_id_alone() -> None:
    assert safe_key_segment("01J9ZC8X4K7Q2M5N") == "01J9ZC8X4K7Q2M5N"


@pytest.mark.parametrize(
    "hostile",
    ["a/b", "..", "x\x00y", "n" * 300, "", "photo id", "\ud800"],
)
def test_safe_key_segment_neutralises_hostile_ids(hostile: str) -> None:
    segment = safe_key_segment(hostile)
    assert segment
    assert "/" not in segment and "\x00" not in segment
    assert len(segment) <= MAX_SEGMENT_LEN
    assert segment not in (".", "..")


def test_safe_key_segment_does_not_collapse_two_different_ids() -> None:
    """Slugifying without a digest would map 'a/b' and 'a-b' onto one key, so
    one inspection's mask would overwrite another's."""
    assert safe_key_segment("a/b") != safe_key_segment("a-b")


# -- Kafka sink: a flush that actually verifies ---------------------------------


class _FakeFuture:
    def __init__(self, exception: BaseException | None = None, done: bool = True) -> None:
        self.is_done = done
        self.exception = exception

    def failed(self) -> bool:
        return self.is_done and self.exception is not None


class _FakeProducer:
    """Reproduces the contract that matters: flush() never raises on a failed
    batch — kafka-python 3.0.11 only logs a warning
    (``record_accumulator.await_flush_completion``)."""

    def __init__(self, outcomes=()) -> None:
        self.outcomes = list(outcomes)
        self.sent: list[tuple] = []
        self.flushes = 0
        self.closed = 0

    def send(self, topic, value=None, key=None, headers=None):
        self.sent.append((topic, key, value, tuple(headers or ())))
        return self.outcomes.pop(0) if self.outcomes else _FakeFuture()

    def flush(self, timeout=None):
        self.flushes += 1

    def close(self) -> None:
        self.closed += 1


def test_flush_returns_quietly_when_everything_was_acknowledged() -> None:
    producer = _FakeProducer()
    sink = kafka_client.KafkaMessageSink(producer)
    sink.send("inspection.masks.v1", b"key", b"body", [("schema", b"inspection.mask")])
    sink.flush(5.0)
    assert producer.flushes == 1 and producer.sent[0][0] == "inspection.masks.v1"


def test_flush_raises_when_the_broker_never_acknowledged() -> None:
    """The at-least-once hinge. The producer is asynchronous and flush() does
    not raise, so ignoring the futures means: mask lost, offset committed, no
    dead letter, no replay possible."""
    failed = _FakeFuture(exception=RuntimeError("NotLeaderForPartitionError"))
    sink = kafka_client.KafkaMessageSink(_FakeProducer([failed]))
    sink.send("inspection.masks.v1", b"key", b"body")
    with pytest.raises(kafka_client.SinkDeliveryError, match="NotLeaderForPartitionError"):
        sink.flush(5.0)


def test_a_record_still_in_flight_after_the_flush_is_a_failure() -> None:
    sink = kafka_client.KafkaMessageSink(_FakeProducer([_FakeFuture(done=False)]))
    sink.send("inspection.masks.v1", b"key", b"body")
    with pytest.raises(kafka_client.SinkDeliveryError, match="in flight"):
        sink.flush(1.0)


def test_a_failed_future_does_not_poison_the_next_flush() -> None:
    """The shutdown path flushes again; a retained failure would mask the
    real error and turn every later flush into a failure."""
    producer = _FakeProducer([_FakeFuture(exception=RuntimeError("boom"))])
    sink = kafka_client.KafkaMessageSink(producer)
    sink.send("inspection.masks.v1", b"key", b"body")
    with pytest.raises(kafka_client.SinkDeliveryError):
        sink.flush(1.0)
    sink.flush(1.0)  # nothing pending any more


def test_a_synchronous_rejection_propagates_from_send() -> None:
    """MessageSizeTooLargeError is raised by send() itself, before any I/O:
    the service classifies it as permanent and dead-letters."""

    class _TooBig(_FakeProducer):
        def send(self, topic, value=None, key=None, headers=None):
            raise ValueError("MessageSizeTooLargeError")

    sink = kafka_client.KafkaMessageSink(_TooBig())
    with pytest.raises(ValueError, match="MessageSizeTooLargeError"):
        sink.send("inspection.masks.v1", b"key", b"body")
