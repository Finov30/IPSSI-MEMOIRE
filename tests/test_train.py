"""Smoke tests for the training loop on a synthetic COCO corpus.

The corpus mimics the ``data/processed`` exports (``split`` field per image,
``memoire_image_id``/``group_id`` extras, polygon segmentations): bright
rectangles on a dark background, so 30 CPU iterations at 64x64 are enough for
the loss to drop measurably. This is a plumbing test, not a performance test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from memoire.model.unet import UNet
from memoire.training.train import load_checkpoint, subset_indices, train

IMG_SIZE = 64
N_TRAIN = 8
N_VAL = 4


def _make_corpus(root: Path) -> tuple[Path, Path]:
    """Write a tiny COCO corpus (images + JSON) under ``root``."""
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


def _config(corpus: Path, images_root: Path, output_dir: Path, **overrides) -> dict:
    config = {
        "seed": 123,
        "corpus": [str(corpus)],
        "images_root": str(images_root),
        "mode": "binary",
        "input_size": IMG_SIZE,
        "base_channels": 8,
        "depth": 2,
        "gn_groups": 4,
        "batch_size": 4,
        "num_workers": 0,
        "augment": False,
        "lr": 3.0e-3,
        "weight_decay": 0.0,
        "iterations": 30,
        "warmup_iterations": 5,
        "val_every": 15,
        "device": "cpu",
        "subset_n_images": None,
        "output_dir": str(output_dir),
        "mlflow": {"tracking_uri": None},
    }
    config.update(overrides)
    return config


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, Path]:
    return _make_corpus(tmp_path_factory.mktemp("synthetic-corpus"))


def test_smoke_loss_decreases_and_checkpoints_reload(corpus, tmp_path):
    coco_path, images_dir = corpus
    config = _config(coco_path, images_dir, tmp_path / "run")
    summary = train(config)

    # Loss goes down over 30 iterations on an easy synthetic task.
    losses = summary["train_losses"]
    assert len(losses) == 30
    assert all(np.isfinite(losses))
    assert np.mean(losses[-10:]) < np.mean(losses[:10])

    # Split sizes and validation metrics are reported.
    assert summary["num_train_images"] == N_TRAIN
    assert summary["num_val_images"] == N_VAL
    assert summary["last_val"] is not None
    assert set(summary["last_val"]["iou_per_class"]) == {0, 1}
    assert set(summary["last_val"]["dice_per_class"]) == {0, 1}
    assert summary["best_val_iou"] is not None

    # Checkpoints exist, carry state_dict + config + iteration, and reload
    # strictly into a freshly built model.
    output_dir = Path(summary["output_dir"])
    for name, expected_final in (("last.pt", True), ("best.pt", False)):
        path = output_dir / name
        assert path.exists(), name
        checkpoint = load_checkpoint(path)
        assert set(checkpoint) >= {"model_state", "config", "iteration"}
        assert checkpoint["config"]["seed"] == config["seed"]
        assert 1 <= checkpoint["iteration"] <= 30
        if expected_final:
            assert checkpoint["iteration"] == 30
        model = UNet(
            in_channels=3,
            num_classes=2,
            base_channels=config["base_channels"],
            depth=config["depth"],
            gn_groups=config["gn_groups"],
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)

    # JSONL fallback history is written even without MLflow.
    history_path = output_dir / "history.jsonl"
    assert history_path.exists()
    events = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert sum(event["type"] == "train" for event in events) == 30
    assert sum(event["type"] == "val" for event in events) == 2  # iterations 15 and 30


def test_same_seed_same_losses(corpus, tmp_path):
    coco_path, images_dir = corpus
    summary_a = train(_config(coco_path, images_dir, tmp_path / "run-a"))
    summary_b = train(_config(coco_path, images_dir, tmp_path / "run-b"))
    assert summary_a["train_losses"] == pytest.approx(summary_b["train_losses"], abs=1e-7)
    assert summary_a["final_train_loss"] == pytest.approx(
        summary_b["final_train_loss"], abs=1e-7
    )


def test_subset_n_images_is_deterministic(corpus, tmp_path):
    coco_path, images_dir = corpus
    config = _config(
        coco_path,
        images_dir,
        tmp_path / "run-subset",
        subset_n_images=4,
        iterations=4,
        warmup_iterations=2,
        val_every=4,
    )
    summary = train(config)
    assert summary["num_train_images"] == 4
    assert summary["num_val_images"] == N_VAL

    # The subsampling itself is a pure function of (n_total, n_keep, seed).
    assert subset_indices(N_TRAIN, 4, seed=123) == subset_indices(N_TRAIN, 4, seed=123)
    assert subset_indices(N_TRAIN, N_TRAIN, seed=123) == list(range(N_TRAIN))
