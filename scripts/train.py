#!/usr/bin/env python
"""Train the damage segmentation U-Net from a YAML config.

Usage:
    uv run python scripts/train.py --config configs/train.yaml \
        [--override key=value] [--override section.key=value] ...

Overrides are type-aware: values are parsed as int, then float, then YAML
(true/false/null/lists), and kept as strings otherwise. Dotted keys reach
into nested sections (e.g. ``mlflow.tracking_uri=file:./mlruns``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from memoire.training.train import train


def parse_value(raw: str) -> Any:
    """Best-effort typed parsing of an override value."""
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_override(config: dict, dotted_key: str, value: Any) -> None:
    """Set ``value`` at ``dotted_key`` in ``config``, creating nested dicts."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="YAML training config (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry (repeatable, dotted keys for nesting)",
    )
    return parser.parse_args(argv)


def load_config(config_path: Path, overrides: list[str]) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"config root must be a mapping: {config_path}")
    for override in overrides:
        key, sep, raw_value = override.partition("=")
        if not sep or not key:
            raise SystemExit(f"invalid --override {override!r}, expected KEY=VALUE")
        apply_override(config, key.strip(), parse_value(raw_value))
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, args.override)
    summary = train(config)
    printable = {k: v for k, v in summary.items() if k != "train_losses"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
