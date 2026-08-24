"""The inference consumer loop (chap. 5.4): photo in, mask out, offsets by hand.

Nothing here imports kafka or torch. The loop talks to
:class:`~memoire.serving.ports.MessageSource`,
:class:`~memoire.serving.ports.MessageSink`,
:class:`~memoire.serving.blobstore.BlobStore` and
:class:`~memoire.serving.ports.SegmentationEngine`, so the whole business
logic is exercised in ``tests/test_serving_service.py`` with in-memory fakes:
no broker, no network, no GPU.

**Delivery semantics: at-least-once.** The order is consume -> infer ->
publish -> flush -> commit. kafka-python has no transactional producer, so
exactly-once is not on the table (that is *why* the mask message carries
``photo_id`` + ``checkpoint_sha256``: inference is deterministic, republishing
the same mask is idempotent, and downstream deduplicates). Committing before
the flush would lose photos on a crash; committing per record rather than per
batch costs a few milliseconds against a 100 ms inference, which buys a
precise restart point.

**Error taxonomy**, the part that keeps a partition alive:

- PERMANENT (invalid JSON, unknown schema, missing field, blob 404, checksum
  mismatch, undecodable image, unusable blob key, message refused by the
  broker for its size) -> DLQ **then commit**. Without this a poison pill
  blocks its partition forever.
- TRANSIENT (object store down, sink refusing) -> bounded retries with
  backoff; if the budget is exhausted the record is **not** committed and the
  loop stops cleanly, so a restart re-reads from the last commit.
- RESOURCE (``MemoryError``, ``torch.cuda.OutOfMemoryError``) -> always
  transient, **never** dead-lettered, whatever the stage. The photo is
  perfectly valid; the machine is momentarily too small. Burying a valid
  photo in the DLQ because a neighbouring process peaked is the one failure
  mode a restart cannot repair.
- An unexpected inference failure is retried, then dead-lettered: a
  systematic bug on one image must not freeze the partition, but must stay
  visible in the DLQ.

Two safety nets sit on top of that taxonomy:

- ``handle_record`` catches *everything*. An exception escaping it kills the
  loop mid-record — no commit, no dead letter — and the container comes back
  on the same offset: an infinite crash loop on one message.
- consecutive dead letters are capped (:attr:`ServiceConfig.max_consecutive_dead_letters`).
  A whole batch failing in a row is an infrastructure fault (an unmounted
  blob volume answers "not found" for every record), not a batch of poison
  pills, and draining a backlog into the DLQ at full speed must not be the
  silent default.
"""

from __future__ import annotations

import logging
import signal
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from memoire.serving import messages as msg
from memoire.serving.blobstore import (
    BlobNotFound,
    BlobStore,
    BlobUnavailable,
    InvalidBlobKey,
    safe_key_segment,
)
from memoire.serving.messages import MessageError
from memoire.serving.ports import InboundRecord, MessageSink, MessageSource, SegmentationEngine
from memoire.serving.preprocess import UnreadableImageError, encode_mask_png, open_image

logger = logging.getLogger("memoire.serving.service")

#: Exit code when the loop gives up on a transient failure. 75 = EX_TEMPFAIL:
#: `restart: unless-stopped` brings the container back and the uncommitted
#: records are simply re-read.
EXIT_TEMPFAIL = 75


class PermanentSinkError(RuntimeError):
    """The broker refuses this exact message and always will.

    kafka-python validates the serialized size *before* compression and
    raises ``MessageSizeTooLargeError`` synchronously from ``send()``: no
    amount of retrying shrinks the record. Treating it as transient is how a
    single oversized message blocks a partition forever.
    """


#: Permanent by construction: nothing about retrying makes them succeed.
PERMANENT_ERRORS = (
    MessageError,
    UnreadableImageError,
    BlobNotFound,
    InvalidBlobKey,
    PermanentSinkError,
)

#: Producer-side rejections that depend on the message alone. Matched by name
#: because this module must never import kafka (``tests/test_serving_isolation``
#: enforces that the adapters stay in ``memoire.serving.kafka_client``).
PERMANENT_SINK_ERROR_NAMES = frozenset(
    {"MessageSizeTooLargeError", "RecordTooLargeError", "RecordListTooLargeError"}
)

#: "The machine is short of memory", not "this message is bad". Never
#: dead-lettered. ``torch.cuda.OutOfMemoryError`` is matched by name for the
#: same reason as above: the loop stays importable without torch.
RESOURCE_ERRORS: tuple[type[BaseException], ...] = (MemoryError,)
RESOURCE_ERROR_NAMES = frozenset({"OutOfMemoryError", "CudaError", "CudaOutOfMemoryError"})


def _is_permanent_sink_error(exc: BaseException) -> bool:
    return type(exc).__name__ in PERMANENT_SINK_ERROR_NAMES


def _is_resource_error(exc: BaseException) -> bool:
    """True for an out-of-memory failure, torch's included.

    ``torch.cuda.OutOfMemoryError`` derives from ``RuntimeError``, so without
    this check a 60-second memory spike on the box would look exactly like a
    model bug and send every valid photo of that minute to the DLQ, committed.
    """
    if isinstance(exc, RESOURCE_ERRORS):
        return True
    if type(exc).__name__ in RESOURCE_ERROR_NAMES:
        return True
    return "out of memory" in str(exc).lower()


class Outcome(str, Enum):
    PUBLISHED = "published"
    DEAD_LETTERED = "dead_lettered"
    RETRY = "retry"


#: ``summary["stopped"]`` values that mean "come back later": the caller exits
#: EX_TEMPFAIL so the container restarts instead of reporting success.
TEMPFAIL_STOPS = frozenset({"transient_failure", "dead_letter_streak"})


@dataclass(frozen=True)
class ServiceConfig:
    masks_topic: str
    dlq_topic: str
    max_attempts: int = 3
    backoff_s: float = 1.0
    poll_timeout_ms: int = 1000
    max_poll_records: int = 8
    flush_timeout_s: float = 30.0
    #: A whole poll batch dead-lettered in a row without one success is an
    #: outage, not a run of poison pills: stop (EX_TEMPFAIL) instead of
    #: draining the backlog into the DLQ. 0 disables the cap.
    max_consecutive_dead_letters: int = 8


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid4_str() -> str:
    return str(uuid.uuid4())


@dataclass
class ServiceDeps:
    """Everything the loop needs, all substitutable in tests."""

    engine: SegmentationEngine
    photo_store: BlobStore
    mask_store: BlobStore
    sink: MessageSink
    config: ServiceConfig
    now: Callable[[], datetime] = _utcnow
    sleep: Callable[[float], None] = time.sleep
    new_id: Callable[[], str] = _uuid4_str
    counters: dict[str, int] = field(default_factory=dict)


class TransientError(RuntimeError):
    """Wraps a failure worth retrying, carrying the stage it happened at."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage = stage
        self.cause = cause


def _retrying(deps: ServiceDeps, stage: str, action: Callable[[], Any]) -> Any:
    """Run ``action`` with bounded retries; raise TransientError when spent.

    Permanent errors are re-raised on the first attempt — retrying a malformed
    message is pure latency.
    """
    attempts = max(1, deps.config.max_attempts)
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except PERMANENT_ERRORS:
            raise
        except Exception as exc:  # deliberately broad: see the docstring
            if _is_permanent_sink_error(exc):
                # Rejected on its size, synchronously, before any network I/O.
                raise PermanentSinkError(f"{type(exc).__name__}: {exc}") from exc
            last = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s", stage, attempt, attempts, exc, exc_info=True
            )
            if attempt < attempts:
                deps.sleep(deps.config.backoff_s * attempt)
    raise TransientError(stage, last if last is not None else RuntimeError("unknown failure"))


def _fetch_photo(message: msg.PhotoMessage, deps: ServiceDeps) -> bytes:
    """Resolve the claim-check (or take the inline bytes) and check sha256."""
    payload = message.payload
    if payload.kind == "inline":
        data = payload.data or b""
    else:
        if not payload.uri:  # unreachable via decode_photo, cheap to keep honest
            raise MessageError("reference payload without uri")
        data = deps.photo_store.get(payload.uri)
    msg.verify_sha256(data, payload.sha256)
    return data


def _send_dlq(record: InboundRecord, deps: ServiceDeps, body: bytes, stage: str) -> None:
    headers = msg.headers_for(msg.DLQ_SCHEMA, deps.new_id())
    _retrying(
        deps,
        f"publish_dlq.{stage}",
        lambda: deps.sink.send(deps.config.dlq_topic, record.key, body, headers),
    )


def _dead_letter(
    record: InboundRecord, deps: ServiceDeps, error: BaseException, stage: str, attempts: int
) -> Outcome:
    """Publish a dead letter for ``record``. NEVER raises.

    It is the last verdict available to :func:`handle_record`; if it could
    raise, the record would leave the loop with no verdict at all.
    """
    try:
        now = deps.now()
        body = msg.encode_dlq(record, error, stage, attempts, now)
    except Exception:
        logger.exception(
            "cannot even build the dead letter for %s[%d]@%d: not committing",
            record.topic, record.partition, record.offset,
        )
        return Outcome.RETRY
    try:
        _send_dlq(record, deps, body, stage)
    except PermanentSinkError as exc:
        # The dead letter itself is too big for the broker. This is not
        # hypothetical: the inline mode accepts 512 KiB, the DLQ carries the
        # original value base64-encoded (x1.334) and the topic caps records at
        # 1 MiB, so the service can mint the very messages it cannot
        # dead-letter. encode_dlq truncates the payload above
        # msg.DLQ_MAX_PAYLOAD_BYTES; if the body is *still* refused, fall back
        # to a payload-free dead letter rather than loop on the offset forever.
        logger.error("dead letter refused by the broker (%s), retrying without payload", exc)
        minimal = msg.encode_dlq(record, error, stage, attempts, now, max_payload_bytes=0)
        try:
            _send_dlq(record, deps, minimal, stage)
        except (TransientError, PermanentSinkError):
            logger.error(
                "even a payload-free dead letter is refused for %s[%d]@%d: not committing",
                record.topic, record.partition, record.offset,
            )
            return Outcome.RETRY
        deps.counters["dlq.truncated"] = deps.counters.get("dlq.truncated", 0) + 1
    except TransientError:
        # A permanent error whose dead letter cannot be published is NOT a
        # permanent outcome: committing here would drop the message entirely
        # (no mask, no dead letter). Degrade to RETRY so nothing is committed
        # and the restart replays the record. Found by
        # tests/test_serve_inference_cli.py, where a broken sink met a missing
        # blob and the exception escaped the loop.
        logger.error(
            "cannot publish the dead letter for %s[%d]@%d: not committing",
            record.topic, record.partition, record.offset,
        )
        return Outcome.RETRY
    deps.counters[f"dlq.{stage}"] = deps.counters.get(f"dlq.{stage}", 0) + 1
    logger.error(
        "dead-lettered %s[%d]@%d at stage %s: %s (%s)",
        record.topic, record.partition, record.offset, stage, error, type(error).__name__,
    )
    return Outcome.DEAD_LETTERED


def handle_record(record: InboundRecord, deps: ServiceDeps) -> Outcome:
    """Process ONE record end to end. Never commits, never polls.

    Returns the outcome and leaves the offset decision to :func:`run_service`,
    which is what keeps "commit only after flush" in a single place.
    """
    stage = "decode"
    try:
        # Inside the try, like everything else: a clock that raises must be a
        # dead letter, not an exception escaping into the poll loop.
        received_at = deps.now()
        message = msg.decode_photo(record.value)

        stage = "fetch"
        data = _retrying(deps, stage, lambda: _fetch_photo(message, deps))

        stage = "decode_image"
        # Wrapped like every other stage, not called bare: open_image can
        # still raise a resource error, and an unwrapped call here made the
        # `decode_image` branch below unreachable.
        image = _retrying(deps, stage, lambda: open_image(data))

        stage = "infer"
        result = _retrying(deps, stage, lambda: deps.engine.predict(image))

        stage = "publish"

        def _publish() -> None:
            # Both ids come off the topic: slugified, never interpolated raw.
            # A photo_id containing a NUL or 300 characters would otherwise
            # make this stage fail identically on every replay of the offset.
            mask_key = (
                f"masks/{safe_key_segment(message.inspection_id)}"
                f"/{safe_key_segment(message.photo_id)}.png"
            )
            mask_uri = deps.mask_store.put(mask_key, encode_mask_png(result.mask), "image/png")
            body = msg.encode_mask(
                message,
                result,
                deps.engine.info,
                mask_uri,
                record,
                received_at,
                deps.now(),
            )
            headers = msg.headers_for(msg.MASK_SCHEMA, deps.new_id())
            # Same key as the incoming photo: masks stay co-partitioned with
            # the photos of the inspection they belong to.
            deps.sink.send(
                deps.config.masks_topic, msg.partition_key(message), body, headers
            )

        _retrying(deps, stage, _publish)
    except PERMANENT_ERRORS as exc:
        return _dead_letter(record, deps, exc, stage, 1)
    except TransientError as exc:
        if _is_resource_error(exc.cause):
            # Out of memory is about the machine, not about this photo:
            # dead-lettering it would bury a perfectly valid image (and a few
            # hundred of its neighbours) for the duration of a memory spike.
            logger.error(
                "resource exhaustion at %s (%s): stopping instead of dead-lettering",
                exc.stage, type(exc.cause).__name__,
            )
            deps.counters["resource_failures"] = deps.counters.get("resource_failures", 0) + 1
            return Outcome.RETRY
        if exc.stage in ("infer", "decode_image"):
            # A model that keeps failing on this one image is a bug, not an
            # outage: dead-letter it so the partition moves on and the case
            # stays inspectable. The consecutive-dead-letter cap in
            # run_service catches the case where it is not one image but all
            # of them.
            return _dead_letter(record, deps, exc.cause, exc.stage, deps.config.max_attempts)
        logger.error("giving up on %s after %d attempts", exc.stage, deps.config.max_attempts)
        return Outcome.RETRY
    except BlobUnavailable as exc:  # raised outside _retrying (mask store put)
        logger.error("blob store unavailable: %s", exc)
        return Outcome.RETRY
    except RESOURCE_ERRORS as exc:  # raised outside _retrying
        logger.error("resource exhaustion at %s: %s", stage, exc)
        return Outcome.RETRY
    except Exception as exc:  # last resort: nothing may escape this function
        # An exception leaving handle_record kills the loop mid-record: no
        # commit, no dead letter, and `restart: unless-stopped` brings the
        # container back on the same offset — an infinite crash loop on one
        # message. Whatever the bug is, the partition must keep moving, and
        # the DLQ is where the evidence goes.
        if _is_resource_error(exc):
            logger.error("resource exhaustion at %s: %s", stage, exc)
            return Outcome.RETRY
        logger.exception("unexpected failure at stage %s, dead-lettering", stage)
        return _dead_letter(record, deps, exc, stage, 1)

    deps.counters["published"] = deps.counters.get("published", 0) + 1
    return Outcome.PUBLISHED


def _never() -> bool:
    return False


def run_service(
    source: MessageSource,
    deps: ServiceDeps,
    should_stop: Callable[[], bool] = _never,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Poll -> handle -> flush -> commit, until stopped.

    ``max_iterations`` bounds the number of poll rounds so tests terminate;
    in production it is None and the loop runs until SIGTERM.
    """
    summary = {
        "n_polls": 0,
        "n_records": 0,
        "n_published": 0,
        "n_dead_lettered": 0,
        "stopped": "signal",
    }
    cap = deps.config.max_consecutive_dead_letters
    streak = 0
    try:
        while not should_stop():
            if max_iterations is not None and summary["n_polls"] >= max_iterations:
                summary["stopped"] = "max_iterations"
                break
            records = source.poll(
                timeout_ms=deps.config.poll_timeout_ms,
                max_records=deps.config.max_poll_records,
            )
            summary["n_polls"] += 1
            for record in records:
                try:
                    outcome = handle_record(record, deps)
                except Exception:
                    # handle_record is written never to raise; if it ever
                    # does, the loop still must not die holding an
                    # uncommitted record — stop cleanly so the restart
                    # replays it instead of crashing on the same offset.
                    logger.exception(
                        "unhandled failure on %s[%d]@%d: stopping without committing",
                        record.topic, record.partition, record.offset,
                    )
                    summary["stopped"] = "transient_failure"
                    return summary
                if outcome is Outcome.RETRY:
                    # No commit: the record (and everything after it on this
                    # partition) is re-read after restart.
                    summary["stopped"] = "transient_failure"
                    return summary
                # Flush BEFORE commit: the broker has acknowledged the mask
                # (or the dead letter) before we admit having consumed it. A
                # flush that reports a lost record must abort the commit —
                # committing there would leave the photo consumed with no
                # mask and no dead letter, unrecoverably.
                try:
                    deps.sink.flush(deps.config.flush_timeout_s)
                    source.commit(record)
                except Exception:
                    # A failed commit is survivable (the record is replayed,
                    # inference is idempotent); a failed flush must never be
                    # followed by one. Both stop the loop without committing.
                    logger.exception(
                        "flush/commit failed on %s[%d]@%d: not committing",
                        record.topic, record.partition, record.offset,
                    )
                    summary["stopped"] = "transient_failure"
                    return summary
                summary["n_records"] += 1
                if outcome is Outcome.PUBLISHED:
                    summary["n_published"] += 1
                    streak = 0
                else:
                    summary["n_dead_lettered"] += 1
                    streak += 1
                    deps.counters["dlq_streak_max"] = max(
                        deps.counters.get("dlq_streak_max", 0), streak
                    )
                    if 0 < cap <= streak:
                        # Every record of a batch failing in a row is an
                        # outage (an unmounted blob volume answers "not
                        # found" for all of them), not a run of poison pills.
                        # Stop: a crash loop is visible, a DLQ quietly
                        # swallowing a 10 000-photo backlog is not.
                        logger.error(
                            "%d consecutive dead letters: stopping, this looks like an "
                            "infrastructure failure rather than %d bad messages",
                            streak, streak,
                        )
                        summary["stopped"] = "dead_letter_streak"
                        return summary
                if should_stop():
                    summary["stopped"] = "signal"
                    break
    finally:
        _shutdown(source, deps)
        # Per-stage counters (how many dead letters, and for which reason):
        # the DLQ volume is one of the two metrics the chapter reports, the
        # other being the consumer lag read from the broker.
        summary["counters"] = dict(deps.counters)
    return summary


def _shutdown(source: MessageSource, deps: ServiceDeps) -> None:
    """Best-effort drain: nothing here may mask the original failure."""
    for label, action in (
        ("flush", lambda: deps.sink.flush(deps.config.flush_timeout_s)),
        ("sink.close", deps.sink.close),
        ("source.close", source.close),
    ):
        try:
            action()
        except Exception:  # shutdown must never raise over the real failure
            logger.warning("%s failed during shutdown", label, exc_info=True)


def install_signal_handlers() -> Callable[[], bool]:
    """Arm SIGTERM/SIGINT and return the ``should_stop`` predicate.

    A flag, not an exception: the record being processed finishes, its mask is
    flushed and its offset committed, *then* the loop exits and the consumer
    leaves the group explicitly (immediate rebalance rather than waiting for
    ``session.timeout.ms``). ``stop_grace_period: 60s`` in compose gives it
    room.
    """
    stopping = {"value": False}

    def _handler(signum: int, _frame: Any) -> None:
        logger.info("received signal %d, finishing the current record", signum)
        stopping["value"] = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)

    return lambda: stopping["value"]
