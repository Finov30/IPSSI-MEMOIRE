#!/usr/bin/env python
"""Kafka inference consumer: photos in, masks out (chap. 5.4).

Usage:
    uv run python scripts/serve_inference.py --config configs/streaming.yaml \
        [--checkpoint runs/dev/best.pt] [--device cpu] [--max-iterations N]

Reads ``inspection.photos.v1``, segments each photo with a trained checkpoint,
writes the mask to the blob store and publishes ``inspection.masks.v1``.
Permanent failures go to ``inspection.photos.dlq.v1``; a transient failure
stops the loop WITHOUT committing (exit code 75, EX_TEMPFAIL) so the container
restarts and re-reads.

Same shape as ``scripts/evaluate_map.py`` (argparse + ``--config`` +
``--checkpoint``), with one deliberate difference: the architecture is NOT
read from the YAML. ``--checkpoint`` alone is enough — ``train()`` stores the
already-resolved config inside the checkpoint, and rebuilding a served model
from a YAML edited after training is exactly how a service ends up serving a
different network than the one that was evaluated.

Nothing in this script talks to Kafka directly: it wires the adapters of
``memoire.serving.kafka_client`` into ``memoire.serving.service.run_service``,
which is why the loop itself is covered by tests without a broker.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from memoire.serving.blobstore import FilesystemBlobStore
from memoire.serving.config import load_streaming_config, service_config
from memoire.serving.kafka_client import (
    KafkaMessageSink,
    KafkaMessageSource,
    build_consumer,
    build_producer,
)
from memoire.serving.service import (
    EXIT_TEMPFAIL,
    TEMPFAIL_STOPS,
    ServiceDeps,
    install_signal_handlers,
    run_service,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="configs/streaming.yaml")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="overrides model.checkpoint"
    )
    parser.add_argument(
        "--device", default=None,
        help="cpu (default) | cuda | auto — 'auto' would take the training GPU",
    )
    parser.add_argument(
        "--bootstrap-servers", default=None, help="overrides bootstrap_servers"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="stop after N poll rounds (smoke test); unbounded by default",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_deps(config: dict, engine, sink) -> ServiceDeps:
    """Wire a resolved config + an engine + a sink into the loop's deps."""
    blob = config["blob"]
    # A missing photos_root usually means the volume is not mounted. It is not
    # fatal here (the directory may legitimately be populated after start, and
    # masks_root is created on first write), but it must be said out loud at
    # startup: without this line the only symptom is a service that fetches
    # nothing. FilesystemBlobStore.get then reports the missing root as
    # TRANSIENT, so the backlog is not drained into the DLQ.
    for name in ("photos_root", "masks_root"):
        if not Path(blob[name]).is_dir():
            logging.getLogger("serve_inference").warning(
                "blob %s %s does not exist: is the volume mounted?", name, blob[name]
            )
    return ServiceDeps(
        engine=engine,
        photo_store=FilesystemBlobStore(blob["photos_root"], blob.get("scheme", "file")),
        mask_store=FilesystemBlobStore(blob["masks_root"], blob.get("scheme", "file")),
        sink=sink,
        config=service_config(config),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_streaming_config(args.config)
    if args.bootstrap_servers:
        config["bootstrap_servers"] = args.bootstrap_servers
    if args.checkpoint:
        config["model"]["checkpoint"] = str(args.checkpoint)
    if args.device:
        config["model"]["device"] = args.device

    # Imported here, not at module level: loading the engine pulls torch, and
    # `--help` must work in an environment that only has the serve extra.
    from memoire.serving.inference import load_engine

    model_cfg = config["model"]
    engine = load_engine(
        model_cfg["checkpoint"],
        device=model_cfg.get("device", "cpu"),
        score_threshold=float(model_cfg.get("score_threshold", 0.5)),
        min_area_px=int(model_cfg.get("min_area_px", 64)),
    )

    consumer = build_consumer(
        config["bootstrap_servers"],
        config["photos_topic"],
        config["group_id"],
        **config["consumer"],
    )
    producer = build_producer(config["bootstrap_servers"], **config["producer"])
    source = KafkaMessageSource(consumer)
    deps = build_deps(config, engine, KafkaMessageSink(producer))

    should_stop = install_signal_handlers()
    summary = run_service(source, deps, should_stop, max_iterations=args.max_iterations)
    print(json.dumps(summary, ensure_ascii=False))
    if summary["stopped"] in TEMPFAIL_STOPS:
        # No offset was committed for the failing record: exiting non-zero lets
        # `restart: unless-stopped` replay it instead of silently dropping it.
        # A dead-letter streak lands here too: a whole batch failing in a row
        # is an outage, and an outage must be as loud as a crash.
        return EXIT_TEMPFAIL
    return 0


if __name__ == "__main__":
    sys.exit(main())
