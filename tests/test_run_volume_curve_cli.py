"""Tests for scripts/run_volume_curve.py's actual CLI entry point (argparse ->
library calls), not just the underlying memoire.training.volume_curve
functions — this is the exact surface a real campaign launch (Colab or local)
invokes.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

IMG_SIZE = 64
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_volume_curve.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_volume_curve_cli", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _make_corpus(root: Path) -> tuple[Path, Path]:
    """8 train images, densities 1,1,2,2,3,3,4,4 (usable by both axes) + 4 val."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    counts = [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4]
    images, annotations = [], []
    ann_id = 1
    for i, n in enumerate(counts):
        split = "train" if i < 8 else "val"
        pixels = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
        file_name = f"img_{i:03d}.png"
        Image.fromarray(pixels).save(images_dir / file_name)
        image_id = i + 1
        images.append(
            {
                "id": image_id, "file_name": file_name, "width": IMG_SIZE, "height": IMG_SIZE,
                "split": split, "source": "synthetic", "memoire_image_id": f"synthetic/img_{i:03d}",
                "group_id": f"synthetic/img_{i:03d}",
            }
        )
        for k in range(n):
            x0, y0, w, h = 4 + k * 4, 4, 6, 6
            annotations.append(
                {
                    "id": ann_id, "image_id": image_id, "category_id": 1,
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
        "info": {"description": "cli test corpus"}, "licenses": [], "images": images,
        "annotations": annotations, "categories": [{"id": 1, "name": "scratch", "supercategory": "damage"}],
    }
    coco_path = root / "corpus.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path, images_dir


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, Path]:
    return _make_corpus(tmp_path_factory.mktemp("cli-corpus"))


@pytest.fixture(scope="module")
def config_path(corpus, tmp_path_factory) -> Path:
    coco_path, images_dir = corpus
    config = {
        "seed": 42,
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
    path = tmp_path_factory.mktemp("cli-config") / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_volume_axis_runs_and_resumes(script, config_path, tmp_path):
    output_root = tmp_path / "campaign"
    script.main([
        "--config", str(config_path),
        "--points", "2,4",
        "--seeds", "1",
        "--output-root", str(output_root),
    ])
    csv_path = output_root / "volume_curve.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert {r["subset_n_images"] for r in rows} == {"2", "4"}
    for run_dir in output_root.glob("n*_seed1"):
        assert (run_dir / "summary.json").exists()

    # Re-invoking the CLI (not the library function) must skip both runs.
    script.main([
        "--config", str(config_path),
        "--points", "2,4",
        "--seeds", "1",
        "--output-root", str(output_root),
    ])
    with csv_path.open(encoding="utf-8") as fh:
        rows_again = list(csv.DictReader(fh))
    assert rows_again == rows


def test_calibrate_prints_estimate(script, config_path, tmp_path, capsys):
    output_root = tmp_path / "calib"
    script.main([
        "--config", str(config_path),
        "--calibrate", "--calibrate-iterations", "2",
        "--output-root", str(output_root),
    ])
    out = json.loads(capsys.readouterr().out)
    assert out["seconds_per_iteration"] > 0
    assert "estimated_campaign_hours" in out


def test_density_axis_runs_via_cli(script, config_path, tmp_path):
    output_root = tmp_path / "density-campaign"
    script.main([
        "--config", str(config_path),
        "--density-buckets", "1,2",
        "--volume-for-density", "2",
        "--seeds", "1,2",
        "--output-root", str(output_root),
    ])
    csv_path = output_root / "density_curve.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {r["density_bucket"] for r in rows} == {"1", "2"}
    assert all(r["num_train_images"] == "2" for r in rows)


def test_density_axis_requires_volume_for_density(script, config_path, tmp_path):
    with pytest.raises(SystemExit):
        script.main([
            "--config", str(config_path),
            "--density-buckets", "1,2",
            "--output-root", str(tmp_path / "bad"),
        ])
