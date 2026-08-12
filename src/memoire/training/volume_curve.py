"""Data-volume curve campaign: one seeded training run per (subset size, seed).

The central experiment of the thesis (how much annotated data is needed) is a
grid of independent :func:`memoire.training.train.train` calls over
``subset_n_images`` values and seeds. This module holds the orchestration
logic; ``scripts/run_volume_curve.py`` is the thin CLI wrapper (same split as
``memoire.training.train`` / ``scripts/train.py``).

Runs are resumable: a ``(point, seed)`` whose ``output_dir`` already has a
``summary.json`` is skipped unless ``force=True`` — safe to relaunch after a
Colab disconnect without losing already-completed runs.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memoire.training.train import train

OnEvent = Callable[[int | None, int, Path, bool], None]


def parse_points(raw: str) -> list[int | None]:
    """Parse a comma-separated points spec (e.g. ``"50,100,full"``).

    ``"full"`` maps to ``None`` (no subsampling, the entire train split).
    """
    points: list[int | None] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        points.append(None if token.lower() == "full" else int(token))
    return points


def parse_seeds(raw: str) -> list[int]:
    """Parse a comma-separated seeds spec (e.g. ``"42,43,44"``)."""
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def point_label(point: int | None) -> str:
    return "full" if point is None else str(point)


def build_run_config(
    base_config: dict,
    point: int | None,
    seed: int,
    output_root: Path,
    iterations: int | None,
) -> tuple[dict, Path]:
    """One training config for ``(point, seed)``, its own ``output_dir``."""
    config = dict(base_config)
    config["subset_n_images"] = point
    config["seed"] = seed
    if iterations is not None:
        config["iterations"] = iterations
    run_dir = output_root / f"n{point_label(point)}_seed{seed}"
    config["output_dir"] = str(run_dir)
    return config, run_dir


def run_campaign(
    base_config: dict,
    points: list[int | None],
    seeds: list[int],
    output_root: Path,
    iterations: int | None = None,
    force: bool = False,
    on_event: OnEvent | None = None,
) -> list[dict[str, Any]]:
    """Train one model per ``(point, seed)`` pair; return one summary row each."""
    if not points or not seeds:
        raise ValueError("points and seeds must both be non-empty")

    rows: list[dict[str, Any]] = []
    for point in points:
        for seed in seeds:
            config, run_dir = build_run_config(base_config, point, seed, output_root, iterations)
            summary_path = run_dir / "summary.json"
            skipped = summary_path.exists() and not force
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8")) if skipped else train(config)
            )
            if on_event:
                on_event(point, seed, run_dir, skipped)
            rows.append(
                {
                    "subset_n_images": point_label(point),
                    "seed": seed,
                    "num_train_images": summary["num_train_images"],
                    "best_val_iou": summary["best_val_iou"],
                    "best_iteration": summary["best_iteration"],
                    "elapsed_seconds": summary["elapsed_seconds"],
                    "output_dir": summary["output_dir"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(
    base_config: dict,
    point: int | None,
    seed: int,
    output_root: Path,
    calibrate_iterations: int,
) -> dict[str, Any]:
    """Run one short job and return the measured seconds/iteration.

    Used to size a campaign (points x seeds x iterations) on the actual
    machine before committing to it — GPU throughput cannot be estimated from
    the CPU smoke-run (``data/processed/SMOKE-RUN.md``).
    """
    if calibrate_iterations <= 0:
        raise ValueError(f"calibrate_iterations must be positive, got {calibrate_iterations}")
    config, run_dir = build_run_config(
        base_config, point, seed, output_root / "_calibration", calibrate_iterations
    )
    summary = train(config)
    return {
        "seconds_per_iteration": summary["elapsed_seconds"] / summary["iterations"],
        "run_dir": str(run_dir),
        "device": summary["device"],
    }
