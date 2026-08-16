#!/usr/bin/env python
"""Instance-level mAP for a trained checkpoint (chap. 7.4/8.4).

Usage:
    uv run python scripts/evaluate_map.py --config configs/train.yaml \
        --checkpoint runs/dev/best.pt [--override key=value] ...

Loads the same YAML config as ``scripts/train.py`` (so corpus/mode/model
architecture match what the checkpoint was trained with), builds the val
split, and reports mAP@0.5 / mAP@[.5:.95] plus per-class AP. This is a
separate, one-shot pass — expensive per-image instance matching, meant to run
once on a finished checkpoint, not every ``val_every`` like the streaming
IoU/Dice/ECE/Brier in ``evaluate()``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from memoire.training.train import (
    build_dataset,
    build_model,
    evaluate_map,
    load_checkpoint,
    num_classes_for,
    resolve_config,
    resolve_device,
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config entry (repeatable, dotted keys for nesting)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = resolve_config(load_config(args.config, args.override))
    device = resolve_device(config["device"])

    num_classes = num_classes_for(config)
    model = build_model(config, num_classes).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(checkpoint["model_state"], strict=True)

    val_dataset = build_dataset(config, "val")
    result = evaluate_map(model, val_dataset, device, num_classes)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
