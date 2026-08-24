"""Configuration of the streaming path (``configs/streaming.yaml``, chap. 5.4).

Same contract as ``memoire.training.train.resolve_config``: a defaults dict
merged with the user's YAML, so a partial config is legal and every key has a
documented value in one place. The merge is one level deeper than training's
(the sections here are nested), and unknown top-level keys are rejected — a
typo in a broker setting must not silently keep the default.

Environment overrides exist for the two values a container must be able to set
without a rebuilt config file: ``MEMOIRE_KAFKA_BOOTSTRAP`` and
``MEMOIRE_CHECKPOINT``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from memoire.serving.service import ServiceConfig

DEFAULTS: dict[str, Any] = {
    "bootstrap_servers": "localhost:29092",
    "photos_topic": "inspection.photos.v1",
    "masks_topic": "inspection.masks.v1",
    "dlq_topic": "inspection.photos.dlq.v1",
    "group_id": "memoire-inference-v1",
    "poll_timeout_ms": 1000,
    "consumer": {},
    "producer": {},
    "blob": {
        "photos_root": "data/streaming/photos",
        "masks_root": "data/streaming/masks",
        "scheme": "file",
    },
    "model": {
        "checkpoint": "runs/serving/best.pt",
        # cpu, never "auto": the training machine's GPU belongs to the
        # campaign, and a service must not silently take it (chap. 5.4).
        "device": "cpu",
        "score_threshold": 0.5,
        "min_area_px": 64,
    },
    "retry": {
        "max_attempts": 3,
        "backoff_s": 1.0,
        # A full poll batch dead-lettered in a row = outage, not poison
        # pills: the loop stops (EX_TEMPFAIL) instead of draining the
        # backlog into the DLQ. 0 disables the cap.
        "max_consecutive_dead_letters": 8,
    },
}

_NESTED = ("consumer", "producer", "blob", "model", "retry")


def resolve_streaming_config(config: dict | None) -> dict[str, Any]:
    """Merge a user config over :data:`DEFAULTS` (nested sections merged)."""
    config = dict(config or {})
    unknown = sorted(set(config) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown streaming config key(s): {unknown} (expected {sorted(DEFAULTS)})"
        )
    resolved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for key, value in config.items():
        if key in _NESTED:
            resolved[key].update(value or {})
        else:
            resolved[key] = value

    bootstrap = os.environ.get("MEMOIRE_KAFKA_BOOTSTRAP")
    if bootstrap:
        resolved["bootstrap_servers"] = bootstrap
    checkpoint = os.environ.get("MEMOIRE_CHECKPOINT")
    if checkpoint:
        resolved["model"]["checkpoint"] = checkpoint
    return resolved


def load_streaming_config(path: Path | str | None) -> dict[str, Any]:
    """Read a YAML config (or none at all) and resolve it."""
    if path is None:
        return resolve_streaming_config({})
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
    return resolve_streaming_config(raw)


def service_config(config: dict[str, Any]) -> ServiceConfig:
    """Extract the loop's own settings from a resolved config."""
    consumer = config["consumer"]
    return ServiceConfig(
        masks_topic=config["masks_topic"],
        dlq_topic=config["dlq_topic"],
        max_attempts=int(config["retry"]["max_attempts"]),
        backoff_s=float(config["retry"]["backoff_s"]),
        poll_timeout_ms=int(config["poll_timeout_ms"]),
        max_poll_records=int(consumer.get("max_poll_records", 8)),
        max_consecutive_dead_letters=int(
            config["retry"].get("max_consecutive_dead_letters", 8)
        ),
    )
