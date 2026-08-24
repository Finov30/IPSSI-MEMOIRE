#!/usr/bin/env python
"""Publish photos onto the upstream topic (chap. 5.4) — the producer side.

Usage:
    uv run python scripts/publish_photos.py --config configs/streaming.yaml \
        --images data-raw/demo --inspection-id NCE01/AB-123-CD/2026-08-24T09:12:03Z \
        [--agency-id NCE01] [--limit 20] [--inline]

Stands in for the agency terminal: it writes each image into the blob store
and publishes a claim-check message keyed by ``inspection_id``. ``--inline``
switches to the degraded mode (bytes in the message, capped at 512 KiB) for a
terminal with no access to the object store.

Useful for the measurement that backs the chapter: publish a burst, then watch
``kafka-consumer-groups.sh --describe --group memoire-inference-v1`` — the lag
rises during the burst and drains afterwards, with nothing refused upstream.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from memoire.serving.blobstore import FilesystemBlobStore
from memoire.serving.config import load_streaming_config
from memoire.serving.kafka_client import KafkaMessageSink, build_producer
from memoire.serving.producer import iter_images, publish_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=None, help="configs/streaming.yaml")
    parser.add_argument("--images", type=Path, required=True, help="directory of image files")
    parser.add_argument("--inspection-id", required=True, help="partition key")
    parser.add_argument("--agency-id", default="unknown")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--inline", action="store_true",
        help="degraded mode: bytes inside the message instead of a reference",
    )
    parser.add_argument("--bootstrap-servers", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = load_streaming_config(args.config)
    if args.bootstrap_servers:
        config["bootstrap_servers"] = args.bootstrap_servers

    blob = config["blob"]
    store = None if args.inline else FilesystemBlobStore(blob["photos_root"], blob["scheme"])
    sink = KafkaMessageSink(build_producer(config["bootstrap_servers"], **config["producer"]))
    try:
        sent = publish_directory(
            sink,
            config["photos_topic"],
            iter_images(args.images),
            inspection_id=args.inspection_id,
            agency_id=args.agency_id,
            store=store,
            inline=args.inline,
            limit=args.limit,
        )
    finally:
        sink.close()
    print(f"published {sent} photo(s) on {config['photos_topic']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
