"""The ONLY module allowed to import kafka (chap. 5.4).

It contains adapters and no business logic: everything else in
``memoire.serving`` talks to :class:`~memoire.serving.ports.MessageSource` /
:class:`~memoire.serving.ports.MessageSink`. That is what lets the whole test
suite run with in-memory fakes — the CI never installs a broker, never opens a
socket, and ``tests/test_serving_isolation.py`` proves mechanically that the
training path never reaches this module.

**Library: kafka-python.** Pure Python (single ``py3-none-any`` wheel, zero
transitive dependencies), so it installs anywhere the project already runs.
``confluent-kafka`` was rejected: its wheels embed librdkafka and its sdist
needs ``librdkafka-dev`` headers. Two consequences are assumed rather than
hidden: no transactional producer (hence at-least-once, see
``memoire.serving.service``), and gzip as the only usable compression codec
(snappy/lz4/zstd would each pull a C extension).

The import is deferred into the factory functions so that
``import memoire.serving.kafka_client`` succeeds without kafka-python
installed — the dependency lives in the optional ``serve`` extra and must
never be dragged into training or CI.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from memoire.serving.ports import InboundRecord

logger = logging.getLogger("memoire.serving.kafka")

CONSUMER_DEFAULTS: dict[str, Any] = {
    # Offsets are committed by hand, after the producer's acknowledgement.
    "enable_auto_commit": False,
    "auto_offset_reset": "earliest",
    # A batch of inferences must fit comfortably inside max_poll_interval_ms,
    # otherwise the consumer is evicted from the group mid-batch.
    "max_poll_records": 8,
    "max_poll_interval_ms": 300_000,
    "session_timeout_ms": 45_000,
    "heartbeat_interval_ms": 3_000,
}

PRODUCER_DEFAULTS: dict[str, Any] = {
    "acks": "all",
    "retries": 5,
    # Ordering guarantee without depending on kafka-python's idempotence
    # support: at most one in-flight batch per connection.
    "max_in_flight_requests_per_connection": 1,
    "compression_type": "gzip",
    "linger_ms": 20,
}


def _kafka() -> Any:
    try:
        # Deferred on purpose: the optional extra must not be needed to import
        # this module (see the module docstring).
        import kafka
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "kafka-python is not installed. It lives in the optional extra: "
            "`uv pip install -e '.[serve]'` (or `pip install kafka-python`). "
            "Training and CI deliberately do not depend on it."
        ) from exc
    return kafka


def build_consumer(
    bootstrap_servers: str, topic: str, group_id: str, **options: Any
) -> Any:
    """Consumer on ``topic``, WITHOUT a value deserializer.

    Raw bytes on purpose: a deserialisation exception raised inside
    kafka-python's iterator cannot be caught cleanly per record, and above all
    the DLQ must carry the original bytes verbatim so a fixed build can replay
    them. Decoding is our own pure function (``messages.decode_photo``).
    """
    kafka = _kafka()
    settings = dict(CONSUMER_DEFAULTS)
    settings.update(options)
    logger.info("consuming %s as group %s via %s", topic, group_id, bootstrap_servers)
    return kafka.KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers.split(","),
        group_id=group_id,
        **settings,
    )


def build_producer(bootstrap_servers: str, **options: Any) -> Any:
    """Producer for the masks and DLQ topics (acks=all, ordered, gzip)."""
    kafka = _kafka()
    settings = dict(PRODUCER_DEFAULTS)
    settings.update(options)
    logger.info("producing via %s", bootstrap_servers)
    return kafka.KafkaProducer(bootstrap_servers=bootstrap_servers.split(","), **settings)


def _offset_and_metadata(offset: int) -> Any:
    """``OffsetAndMetadata`` for the next offset to read, across versions.

    kafka-python 2.x has ``(offset, metadata)``, 3.x added ``leader_epoch`` —
    building it positionally from ``_fields`` keeps both working.
    """
    structs = _kafka().structs
    cls = structs.OffsetAndMetadata
    fields = getattr(cls, "_fields", ("offset", "metadata"))
    values = {"offset": offset, "metadata": "", "leader_epoch": -1}
    return cls(*[values.get(name) for name in fields])


class KafkaMessageSource:
    """:class:`~memoire.serving.ports.MessageSource` over a ``KafkaConsumer``."""

    def __init__(self, consumer: Any) -> None:
        self._consumer = consumer

    def poll(self, timeout_ms: int = 1000, max_records: int = 8) -> list[InboundRecord]:
        batches = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        records: list[InboundRecord] = []
        for partition, entries in batches.items():
            for entry in entries:
                records.append(
                    InboundRecord(
                        topic=partition.topic,
                        partition=partition.partition,
                        offset=entry.offset,
                        key=entry.key,
                        value=entry.value,
                        headers=tuple(entry.headers or ()),
                        timestamp_ms=entry.timestamp,
                    )
                )
        # Deterministic order across partitions; within a partition Kafka's own
        # order is already offset order.
        records.sort(key=lambda rec: (rec.partition, rec.offset))
        return records

    def commit(self, record: InboundRecord) -> None:
        kafka = _kafka()
        topic_partition = kafka.TopicPartition(record.topic, record.partition)
        # Kafka commits the offset of the NEXT record to read, hence +1.
        self._consumer.commit({topic_partition: _offset_and_metadata(record.offset + 1)})

    def close(self) -> None:
        # Explicit close = explicit LeaveGroup: the partitions are reassigned
        # immediately instead of after session.timeout.ms.
        self._consumer.close()


class SinkDeliveryError(RuntimeError):
    """At least one record of the flushed batch was never acknowledged.

    Transient by nature (leader election, broker restart): the loop treats it
    as such and, crucially, does NOT commit the offset.
    """


class KafkaMessageSink:
    """:class:`~memoire.serving.ports.MessageSink` over a ``KafkaProducer``.

    The futures matter. ``KafkaProducer.send`` is asynchronous and
    ``KafkaProducer.flush`` does **not** raise when a batch failed for good:
    ``record_accumulator.await_flush_completion`` only does
    ``if batch.produce_future.failed(): log.warning(...)`` (kafka-python
    3.0.11). Dropping the future would therefore make "flush before commit"
    an empty ritual — a lost mask followed by a committed offset, which is
    exactly the at-least-once promise of chap. 5.4 being broken silently, and
    without any replay possible since the offset is gone.

    So every future is kept and inspected after the flush; one failure aborts
    the flush, the loop returns ``Outcome.RETRY``, nothing is committed and
    the restart replays the photo.
    """

    def __init__(self, producer: Any) -> None:
        self._producer = producer
        self._pending: list[Any] = []

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[tuple[str, bytes]] = (),
    ) -> None:
        # A synchronous raise here (MessageSizeTooLargeError on an oversized
        # record) is deliberately left to propagate: the service classifies it
        # as permanent and dead-letters.
        future = self._producer.send(topic, value=value, key=key, headers=list(headers))
        if future is not None:
            self._pending.append(future)

    def flush(self, timeout_s: float = 30.0) -> None:
        """Block until the broker has acknowledged every buffered record.

        Raises :class:`SinkDeliveryError` if any of them was not acknowledged
        — the caller must not commit after that.
        """
        try:
            self._producer.flush(timeout=timeout_s)
            failures = [
                future
                for future in self._pending
                if not getattr(future, "is_done", True) or future.failed()
            ]
        finally:
            # Cleared either way: a retained failed future would poison every
            # later flush, including the one in the shutdown path.
            pending, self._pending = self._pending, []
        if failures:
            first = next((f.exception for f in failures if f.exception is not None), None)
            detail = (
                f"{type(first).__name__}: {first}" if first is not None
                else f"still in flight after {timeout_s}s"
            )
            raise SinkDeliveryError(
                f"{len(failures)}/{len(pending)} records were not acknowledged ({detail})"
            )

    def close(self) -> None:
        self._pending.clear()
        self._producer.close()
