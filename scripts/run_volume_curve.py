#!/usr/bin/env python
"""Run the data-volume or data-density curve campaign (chap. 7.1: the two
decoupled axes of the thesis's central experiment).

Volume axis (default) — one training per (subset size, seed):
    uv run python scripts/run_volume_curve.py --config configs/train.yaml \
        [--points 50,100,250,500,1000,full] [--seeds 42,43,44] \
        [--iterations 20000] [--output-root runs/volume-curve]

Density axis — image count held constant, density stratum varied instead;
one training per (density bucket, seed):
    uv run python scripts/run_volume_curve.py --config configs/train.yaml \
        --density-buckets 1,2,3,4+ --volume-for-density 500 --seeds 42,43,44 \
        [--iterations 20000] [--output-root runs/density-curve]

Resumable on both axes: a run whose output_dir already has a summary.json is
skipped unless --force is passed — safe to relaunch after a Colab disconnect.

Calibrate first, on the target GPU, before committing to the full campaign
(seconds/iteration cannot be estimated from the CPU smoke-run; the estimate
applies to either axis, since density-based subsetting doesn't change
per-iteration compute cost, only which images are selected):
    uv run python scripts/run_volume_curve.py --config configs/train.yaml \
        --calibrate --calibrate-iterations 100
prints seconds/iteration and the estimated total campaign wall-clock time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from memoire.training.volume_curve import (
    calibrate,
    density_bucket_label,
    parse_density_buckets,
    parse_points,
    parse_seeds,
    run_campaign,
    run_density_campaign,
    write_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="base YAML training config (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--points", default="50,100,250,500,1000,full",
        help="comma-separated subset_n_images values; 'full' = no subsampling",
    )
    parser.add_argument(
        "--seeds", default="42,43,44",
        help="comma-separated seeds; one run per point x seed",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="override the config's iterations for every run",
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/volume-curve"),
        help="root directory for per-run output_dir and the aggregated CSV",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-run even if a run's output_dir already has a summary.json",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="run one short job (largest point, first seed) and print a time estimate, then exit",
    )
    parser.add_argument("--calibrate-iterations", type=int, default=100)
    parser.add_argument(
        "--density-buckets", default=None,
        help="density axis instead of volume: comma-separated instances/image buckets "
        "(e.g. '1,2,3,4+'); requires --volume-for-density",
    )
    parser.add_argument(
        "--volume-for-density", type=int, default=None,
        help="fixed subset_n_images for every density-axis run (image count held constant)",
    )
    return parser.parse_args(argv)


def load_base_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"config root must be a mapping: {config_path}")
    return config


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_config = load_base_config(args.config)
    points = parse_points(args.points)
    seeds = parse_seeds(args.seeds)
    if not points or not seeds:
        raise SystemExit("--points and --seeds must both be non-empty")

    if args.calibrate:
        if args.calibrate_iterations <= 0:
            raise SystemExit(f"--calibrate-iterations must be positive, got {args.calibrate_iterations}")
        result = calibrate(base_config, points[-1], seeds[0], args.output_root, args.calibrate_iterations)
        n_runs = len(points) * len(seeds)
        target_iterations = (
            args.iterations if args.iterations is not None else base_config.get("iterations", 200)
        )
        estimated_hours = result["seconds_per_iteration"] * target_iterations * n_runs / 3600
        print(json.dumps(
            {
                **result,
                "planned_runs": n_runs,
                "iterations_per_run": target_iterations,
                "estimated_campaign_hours": round(estimated_hours, 2),
            },
            indent=2,
        ))
        return

    if args.density_buckets:
        if args.volume_for_density is None:
            raise SystemExit("--density-buckets requires --volume-for-density")
        buckets = parse_density_buckets(args.density_buckets)
        if not buckets:
            raise SystemExit("--density-buckets must be non-empty")

        def on_event_density(bucket, seed: int, run_dir: Path, skipped: bool) -> None:
            label = "skip" if skipped else "run"
            print(
                f"[{label}] density_bucket={density_bucket_label(bucket)} "
                f"volume={args.volume_for_density} seed={seed} -> {run_dir}"
            )

        rows = run_density_campaign(
            base_config, args.volume_for_density, buckets, seeds, args.output_root,
            iterations=args.iterations, force=args.force, on_event=on_event_density,
        )
        csv_path = args.output_root / "density_curve.csv"
        write_csv(rows, csv_path)
        print(f"[done] {len(rows)} runs -> {csv_path}")
        return

    def on_event(point: int | None, seed: int, run_dir: Path, skipped: bool) -> None:
        label = "skip" if skipped else "run"
        point_str = "full" if point is None else str(point)
        print(f"[{label}] subset_n_images={point_str} seed={seed} -> {run_dir}")

    rows = run_campaign(
        base_config, points, seeds, args.output_root,
        iterations=args.iterations, force=args.force, on_event=on_event,
    )
    csv_path = args.output_root / "volume_curve.csv"
    write_csv(rows, csv_path)
    print(f"[done] {len(rows)} runs -> {csv_path}")


if __name__ == "__main__":
    main()
