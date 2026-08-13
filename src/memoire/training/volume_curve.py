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

OnEvent = Callable[[Any, int, Path, bool], None]
DensityBucket = tuple[int, int | None]


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


def parse_density_buckets(raw: str) -> list[DensityBucket]:
    """Parse a comma-separated density-bucket spec (e.g. ``"1,2,3,4+"``).

    A bare integer is an exact-count bucket (``"2"`` -> ``(2, 2)``); a
    trailing ``+`` is an open-ended bucket (``"4+"`` -> ``(4, None)``), the
    thesis's "4+ instances" stratum (chap. 7.1).
    """
    buckets: list[DensityBucket] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("+"):
            buckets.append((int(token[:-1]), None))
        else:
            n = int(token)
            buckets.append((n, n))
    return buckets


def density_bucket_label(bucket: DensityBucket) -> str:
    lo, hi = bucket
    if hi is None:
        return f"{lo}+"
    return str(lo) if lo == hi else f"{lo}-{hi}"


def build_run_config(
    base_config: dict,
    point: int | None,
    seed: int,
    output_root: Path,
    iterations: int | None,
    density_bucket: DensityBucket | None = None,
) -> tuple[dict, Path]:
    """One training config for ``(point, seed)`` — or ``(point, density_bucket,
    seed)`` on the density axis (chap. 7.1: volume held constant via
    ``point``, density varied via ``density_bucket``) — with its own
    ``output_dir``.
    """
    config = dict(base_config)
    config["subset_n_images"] = point
    config["seed"] = seed
    if iterations is not None:
        config["iterations"] = iterations
    if density_bucket is not None:
        config["density_bucket"] = list(density_bucket)
        run_dir = (
            output_root
            / f"n{point_label(point)}_density{density_bucket_label(density_bucket)}_seed{seed}"
        )
    else:
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


def run_density_campaign(
    base_config: dict,
    volume: int,
    buckets: list[DensityBucket],
    seeds: list[int],
    output_root: Path,
    iterations: int | None = None,
    force: bool = False,
    on_event: OnEvent | None = None,
) -> list[dict[str, Any]]:
    """Train one model per ``(density_bucket, seed)`` pair at a fixed ``volume``.

    The density axis of chap. 7.1: image count held constant (``volume``,
    the He et al. confound this decouples), which density stratum the
    training images are drawn from varied instead — tests whether the
    from-scratch breakdown tracks density rather than volume (H2), the
    hypothesis He et al. raise but never test (chap. 2.4).
    """
    if not buckets or not seeds:
        raise ValueError("buckets and seeds must both be non-empty")

    rows: list[dict[str, Any]] = []
    for bucket in buckets:
        for seed in seeds:
            config, run_dir = build_run_config(
                base_config, volume, seed, output_root, iterations, density_bucket=bucket
            )
            summary_path = run_dir / "summary.json"
            skipped = summary_path.exists() and not force
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8")) if skipped else train(config)
            )
            if on_event:
                on_event(bucket, seed, run_dir, skipped)
            rows.append(
                {
                    "density_bucket": density_bucket_label(bucket),
                    "subset_n_images": point_label(volume),
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
