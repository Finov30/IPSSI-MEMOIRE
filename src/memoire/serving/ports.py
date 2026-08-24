"""Contracts of the streaming inference path (chap. 5.4): message transport,
segmentation engine, and the result types that travel between them.

This module is the seam that makes the whole path testable without a broker
and without a GPU: it declares *only* ``Protocol`` classes and frozen
dataclasses, and it imports **neither kafka nor torch**. The kafka adapters
live in ``memoire.serving.kafka_client`` (the single module allowed to import
``kafka``), the torch engine in ``memoire.serving.inference``; the service
loop (``memoire.serving.service``) sees nothing but the protocols below, so
the unit tests substitute in-memory fakes for all of them.

Why an explicit ``poll``/``commit`` pair rather than iterating a
``KafkaConsumer``: offsets are committed by hand, one record at a time, after
the producer has acknowledged the downstream message (at-least-once, see
``service.run_service``). An iterator with auto-commit would acknowledge
photos that were never published.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class InboundRecord:
    """One consumed message, kept as raw bytes.

    ``value`` is deliberately *not* deserialised by the transport: a decoding
    error must be catchable by our own code (it is a permanent error, routed
    to the dead-letter queue), and the DLQ has to carry the original bytes
    verbatim so a fixed build can replay them unchanged.
    """

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes
    headers: tuple[tuple[str, bytes], ...] = ()
    timestamp_ms: int | None = None


class MessageSource(Protocol):
    """Upstream side: hands out records and commits offsets explicitly."""

    def poll(self, timeout_ms: int = 1000, max_records: int = 8) -> list[InboundRecord]: ...

    def commit(self, record: InboundRecord) -> None:
        """Commit ``record.offset + 1`` for that record's partition."""
        ...

    def close(self) -> None: ...


class MessageSink(Protocol):
    """Downstream side: publishes masks and dead letters."""

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[tuple[str, bytes]] = (),
    ) -> None: ...

    def flush(self, timeout_s: float = 30.0) -> None:
        """Block until every buffered message is acknowledged by the broker."""
        ...

    def close(self) -> None: ...


# -- inference results --------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the served checkpoint, echoed in every mask message.

    ``checkpoint_sha256`` is what downstream consumers deduplicate on together
    with ``photo_id``: at-least-once delivery means the same photo can be
    inferred twice, and inference is deterministic, so republishing the same
    mask is idempotent as long as the model is identified.
    """

    name: str
    mode: str
    input_size: int
    num_classes: int
    class_names: tuple[str, ...]
    checkpoint_sha256: str
    run_id: str | None = None
    iteration: int | None = None


@dataclass(frozen=True)
class DamageInstance:
    """One predicted damage region, in the pixels of the ORIGINAL image.

    Never in the model's letterboxed 512x512 frame: a business consumer
    (dashboard, billing engine) must know nothing about the network's input
    geometry.
    """

    class_id: int
    class_name: str
    score: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: int


@dataclass(frozen=True)
class InferenceResult:
    """Everything one photo produces: mask at native resolution + instances."""

    mask: np.ndarray  # H x W uint8, ORIGINAL resolution, class ids
    instances: list[DamageInstance] = field(default_factory=list)
    damage_pixel_ratio: float = 0.0
    inference_ms: float = 0.0


@runtime_checkable
class SegmentationEngine(Protocol):
    """Anything that turns a PIL image into an :class:`InferenceResult`.

    The service depends on this, never on torch: the service tests run with a
    constant engine, no checkpoint and no GPU.
    """

    info: ModelInfo

    def predict(self, image: object) -> InferenceResult: ...
