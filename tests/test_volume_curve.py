"""Tests for the data-volume curve orchestration (memoire.training.volume_curve)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from memoire.training.volume_curve import (
    build_run_config,
    calibrate,
    density_bucket_label,
    parse_density_buckets,
    parse_points,
    parse_seeds,
    point_label,
    run_campaign,
    run_density_campaign,
    write_csv,
)

IMG_SIZE = 64
N_TRAIN = 8
N_VAL = 4


def _make_corpus(root: Path) -> tuple[Path, Path]:
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    images, annotations = [], []
    for i in range(N_TRAIN + N_VAL):
        split = "train" if i < N_TRAIN else "val"
        x0 = 6 + (i * 5) % 20
        y0 = 6 + (i * 7) % 20
        w, h = 24, 18
        pixels = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
        pixels[y0 : y0 + h, x0 : x0 + w] = 220
        file_name = f"img_{i:03d}.png"
        Image.fromarray(pixels).save(images_dir / file_name)

        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "split": split,
                "source": "synthetic",
                "memoire_image_id": f"synthetic/img_{i:03d}",
                "group_id": f"synthetic/img_{i:03d}",
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [
                    [
                        float(x0), float(y0),
                        float(x0 + w), float(y0),
                        float(x0 + w), float(y0 + h),
                        float(x0), float(y0 + h),
                    ]
                ],
                "bbox": [float(x0), float(y0), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )
    coco = {
        "info": {"description": "synthetic smoke corpus"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "scratch", "supercategory": "damage"}],
    }
    coco_path = root / "corpus.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path, images_dir


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, Path]:
    return _make_corpus(tmp_path_factory.mktemp("synthetic-corpus"))


def _make_density_corpus(root: Path) -> tuple[Path, Path]:
    """8 train images with varying instance counts (1,1,2,2,3,3,4,4) + 4 val images."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    counts = [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4]  # last 4 are val
    images, annotations = [], []
    ann_id = 1
    for i, n in enumerate(counts):
        split = "train" if i < 8 else "val"
        pixels = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
        file_name = f"dimg_{i:03d}.png"
        Image.fromarray(pixels).save(images_dir / file_name)
        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "split": split,
                "source": "synthetic",
                "memoire_image_id": f"synthetic/dimg_{i:03d}",
                "group_id": f"synthetic/dimg_{i:03d}",
            }
        )
        for k in range(n):
            x0, y0, w, h = 4 + k * 4, 4, 6, 6
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [
                        [
                            float(x0), float(y0),
                            float(x0 + w), float(y0),
                            float(x0 + w), float(y0 + h),
                            float(x0), float(y0 + h),
                        ]
                    ],
                    "bbox": [float(x0), float(y0), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    coco = {
        "info": {"description": "density corpus"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "scratch", "supercategory": "damage"}],
    }
    coco_path = root / "density_corpus.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path, images_dir


@pytest.fixture(scope="module")
def density_corpus(tmp_path_factory) -> tuple[Path, Path]:
    return _make_density_corpus(tmp_path_factory.mktemp("density-corpus"))


def _base_config(corpus: tuple[Path, Path]) -> dict:
    coco_path, images_dir = corpus
    return {
        "corpus": [str(coco_path)],
        "images_root": str(images_dir),
        "mode": "binary",
        "input_size": IMG_SIZE,
        "base_channels": 8,
        "depth": 2,
        "gn_groups": 4,
        "batch_size": 2,
        "num_workers": 0,
        "augment": False,
        "lr": 3.0e-3,
        "weight_decay": 0.0,
        "iterations": 4,
        "warmup_iterations": 1,
        "val_every": 4,
        "device": "cpu",
        "mlflow": {"tracking_uri": None},
    }


def test_parse_points():
    assert parse_points("50,100,full") == [50, 100, None]
    assert parse_points(" 10 , full ,") == [10, None]


def test_parse_seeds():
    assert parse_seeds("42,43, 44") == [42, 43, 44]


def test_point_label():
    assert point_label(None) == "full"
    assert point_label(50) == "50"


def test_build_run_config_sets_output_dir_and_overrides(corpus):
    base_config = _base_config(corpus)
    config, run_dir = build_run_config(base_config, 3, 7, Path("runs/vc"), iterations=99)
    assert config["subset_n_images"] == 3
    assert config["seed"] == 7
    assert config["iterations"] == 99
    assert run_dir == Path("runs/vc/n3_seed7")
    assert config["output_dir"] == str(run_dir)

    full_config, full_dir = build_run_config(base_config, None, 7, Path("runs/vc"), iterations=None)
    assert full_config["subset_n_images"] is None
    assert full_config["iterations"] == base_config["iterations"]
    assert full_dir == Path("runs/vc/nfull_seed7")


def test_run_campaign_produces_one_row_per_point_and_seed_and_resumes(corpus, tmp_path):
    base_config = _base_config(corpus)
    output_root = tmp_path / "campaign"
    events = []
    rows = run_campaign(
        base_config,
        points=[2, None],
        seeds=[1, 2],
        output_root=output_root,
        on_event=lambda point, seed, run_dir, skipped: events.append((point, seed, skipped)),
    )
    assert len(rows) == 4
    assert {(e[0], e[1]) for e in events} == {(2, 1), (2, 2), (None, 1), (None, 2)}
    assert all(not skipped for _, _, skipped in events)

    for row in rows:
        assert (Path(row["output_dir"]) / "summary.json").exists()
    two_image_rows = [r for r in rows if r["subset_n_images"] == "2"]
    full_rows = [r for r in rows if r["subset_n_images"] == "full"]
    assert all(r["num_train_images"] == 2 for r in two_image_rows)
    assert all(r["num_train_images"] == N_TRAIN for r in full_rows)

    # Re-running without --force skips every (point, seed): resumable after a
    # disconnect, no wasted GPU time re-doing completed runs.
    events.clear()
    rows_again = run_campaign(
        base_config,
        points=[2, None],
        seeds=[1, 2],
        output_root=output_root,
        on_event=lambda point, seed, run_dir, skipped: events.append(skipped),
    )
    assert all(events)
    assert rows_again == rows


def test_run_campaign_rejects_empty_points_or_seeds(corpus, tmp_path):
    base_config = _base_config(corpus)
    with pytest.raises(ValueError):
        run_campaign(base_config, points=[], seeds=[1], output_root=tmp_path)
    with pytest.raises(ValueError):
        run_campaign(base_config, points=[1], seeds=[], output_root=tmp_path)


def test_write_csv(tmp_path):
    rows = [
        {"subset_n_images": "50", "seed": 42, "best_val_iou": 0.1},
        {"subset_n_images": "full", "seed": 42, "best_val_iou": 0.3},
    ]
    csv_path = tmp_path / "out" / "volume_curve.csv"
    write_csv(rows, csv_path)
    with csv_path.open(encoding="utf-8") as fh:
        read_rows = list(csv.DictReader(fh))
    assert [r["subset_n_images"] for r in read_rows] == ["50", "full"]
    assert read_rows[0]["best_val_iou"] == "0.1"


def test_calibrate_reports_seconds_per_iteration(corpus, tmp_path):
    base_config = _base_config(corpus)
    result = calibrate(base_config, point=None, seed=1, output_root=tmp_path, calibrate_iterations=2)
    assert result["seconds_per_iteration"] > 0
    assert result["device"] == "cpu"
    assert (Path(result["run_dir"]) / "summary.json").exists()


def test_calibrate_rejects_non_positive_iterations(corpus, tmp_path):
    base_config = _base_config(corpus)
    with pytest.raises(ValueError):
        calibrate(base_config, point=None, seed=1, output_root=tmp_path, calibrate_iterations=0)


# --- density axis (chap. 7.1) ---


def test_parse_density_buckets():
    assert parse_density_buckets("1,2,3,4+") == [(1, 1), (2, 2), (3, 3), (4, None)]
    assert parse_density_buckets(" 1 , 4+ ,") == [(1, 1), (4, None)]


def test_density_bucket_label():
    assert density_bucket_label((2, 2)) == "2"
    assert density_bucket_label((4, None)) == "4+"
    assert density_bucket_label((2, 3)) == "2-3"


def test_build_run_config_with_density_bucket(corpus):
    base_config = _base_config(corpus)
    config, run_dir = build_run_config(
        base_config, 5, 7, Path("runs/dc"), iterations=None, density_bucket=(2, 2)
    )
    assert config["subset_n_images"] == 5
    assert config["density_bucket"] == [2, 2]
    assert run_dir == Path("runs/dc/n5_density2_seed7")
    assert config["output_dir"] == str(run_dir)

    # Without density_bucket, naming/config are unchanged (backward compatible).
    plain_config, plain_dir = build_run_config(base_config, 5, 7, Path("runs/dc"), iterations=None)
    assert "density_bucket" not in plain_config
    assert plain_dir == Path("runs/dc/n5_seed7")


def test_run_density_campaign_holds_volume_constant_and_varies_bucket(density_corpus, tmp_path):
    base_config = _base_config(density_corpus)
    output_root = tmp_path / "density-campaign"
    events = []
    rows = run_density_campaign(
        base_config,
        volume=2,
        buckets=[(1, 1), (2, 2)],
        seeds=[1, 2],
        output_root=output_root,
        on_event=lambda bucket, seed, run_dir, skipped: events.append((bucket, seed, skipped)),
    )
    assert len(rows) == 4
    assert {(e[0], e[1]) for e in events} == {((1, 1), 1), ((1, 1), 2), ((2, 2), 1), ((2, 2), 2)}
    assert all(not skipped for _, _, skipped in events)
    # Volume (image count) is the same across every run; only the density bucket varies.
    assert all(r["num_train_images"] == 2 for r in rows)
    assert {r["density_bucket"] for r in rows} == {"1", "2"}

    # Resumable, same contract as the volume axis.
    events.clear()
    rows_again = run_density_campaign(
        base_config,
        volume=2,
        buckets=[(1, 1), (2, 2)],
        seeds=[1, 2],
        output_root=output_root,
        on_event=lambda bucket, seed, run_dir, skipped: events.append(skipped),
    )
    assert all(events)
    assert rows_again == rows


def test_run_density_campaign_rejects_empty_buckets_or_seeds(density_corpus, tmp_path):
    base_config = _base_config(density_corpus)
    with pytest.raises(ValueError):
        run_density_campaign(base_config, volume=2, buckets=[], seeds=[1], output_root=tmp_path)
    with pytest.raises(ValueError):
        run_density_campaign(
            base_config, volume=2, buckets=[(1, 1)], seeds=[], output_root=tmp_path
        )
