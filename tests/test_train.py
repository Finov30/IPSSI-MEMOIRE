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
from torch.utils.data import ConcatDataset, Subset

from memoire.model.unet import UNet
from memoire.training.train import (
    _iter_damage_datasets,
    build_dataset,
    density_indices,
    flat_instance_counts,
    load_checkpoint,
    subset_indices,
    train,
)

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


def _make_density_corpus(root: Path) -> tuple[Path, Path]:
    """6 train images with varying instance counts (1, 1, 2, 2, 4, 4) + 2 val images."""
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    counts = [1, 1, 2, 2, 4, 4, 1, 2]  # last two are val, count irrelevant there
    images, annotations = [], []
    ann_id = 1
    for i, n in enumerate(counts):
        split = "train" if i < 6 else "val"
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


@pytest.fixture(scope="module")
def density_corpus(tmp_path_factory) -> tuple[Path, Path]:
    return _make_density_corpus(tmp_path_factory.mktemp("density-corpus"))


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


def test_subset_n_images_zero_is_not_treated_as_no_subsampling(corpus, tmp_path):
    # subset_n_images=0 is a real request for an empty train split, not the
    # None sentinel for "no subsampling" — both are falsy, so `if subset_n:`
    # used to silently fall back to the full dataset instead of raising.
    coco_path, images_dir = corpus
    config = _config(coco_path, images_dir, tmp_path / "run-subset-zero", subset_n_images=0)
    with pytest.raises(ValueError, match="empty train split"):
        train(config)


def test_iter_damage_datasets_finds_nested_instances(corpus, tmp_path):
    coco_path, images_dir = corpus
    config = _config(coco_path, images_dir, tmp_path / "run-nested")
    dataset = build_dataset(config, "train")

    assert list(_iter_damage_datasets(dataset)) == [dataset]
    assert list(_iter_damage_datasets(Subset(dataset, [0, 1]))) == [dataset]
    assert list(_iter_damage_datasets(ConcatDataset([dataset, dataset]))) == [dataset, dataset]


def test_multi_worker_augmented_run_completes(corpus, tmp_path):
    # Regression test for the shared-generator bug: with num_workers>0 (the
    # setting docs/COLAB.md actually uses for real GPU runs) and augment=True,
    # every worker used to inherit an identical, unadvanced flip generator.
    # This exercises that path end-to-end (previously untested entirely).
    coco_path, images_dir = corpus
    config = _config(
        coco_path,
        images_dir,
        tmp_path / "run-multiworker",
        augment=True,
        num_workers=2,
        iterations=6,
        warmup_iterations=1,
        val_every=6,
    )
    summary = train(config)
    assert len(summary["train_losses"]) == 6
    assert all(np.isfinite(summary["train_losses"]))


# --- density axis (chap. 7.1: volume held constant, density stratum varied) ---


def test_flat_instance_counts_matches_dataset(density_corpus):
    coco_path, images_dir = density_corpus
    config = {
        "corpus": [str(coco_path)],
        "images_root": str(images_dir),
        "mode": "binary",
        "input_size": IMG_SIZE,
        "augment": False,
    }
    ds = build_dataset(config, "train")
    assert flat_instance_counts(ds) == [1, 1, 2, 2, 4, 4]


def test_flat_instance_counts_concat_dataset_preserves_order(density_corpus):
    coco_path, images_dir = density_corpus
    config = {
        "corpus": [str(coco_path)],
        "images_root": str(images_dir),
        "mode": "binary",
        "input_size": IMG_SIZE,
        "augment": False,
    }
    ds = build_dataset(config, "train")
    combined = ConcatDataset([ds, ds])
    assert flat_instance_counts(combined) == [1, 1, 2, 2, 4, 4, 1, 1, 2, 2, 4, 4]


def test_density_indices_filters_to_the_bucket():
    counts = [1, 1, 2, 2, 4, 4]
    indices = density_indices(counts, n_keep=2, seed=1, density_min=2, density_max=2)
    assert indices == sorted(indices)
    assert set(indices) <= {2, 3}


def test_density_indices_open_ended_bucket():
    counts = [1, 2, 3, 4, 5]
    indices = density_indices(counts, n_keep=2, seed=0, density_min=4, density_max=None)
    assert set(indices) <= {3, 4}


def test_density_indices_deterministic_at_fixed_seed():
    counts = [1, 1, 2, 2, 4, 4]
    a = density_indices(counts, n_keep=2, seed=7, density_min=1, density_max=1)
    b = density_indices(counts, n_keep=2, seed=7, density_min=1, density_max=1)
    assert a == b


def test_density_indices_raises_when_bucket_too_small():
    counts = [1, 1, 1]
    with pytest.raises(ValueError, match="density bucket"):
        density_indices(counts, n_keep=5, seed=0, density_min=1, density_max=1)


def test_train_with_density_bucket_selects_only_matching_images(density_corpus, tmp_path):
    coco_path, images_dir = density_corpus
    config = _config(
        coco_path,
        images_dir,
        tmp_path / "run-density",
        subset_n_images=2,
        density_bucket=[2, 2],
        iterations=2,
        warmup_iterations=1,
        val_every=2,
    )
    summary = train(config)
    assert summary["num_train_images"] == 2


def test_train_density_bucket_without_subset_n_images_raises(density_corpus, tmp_path):
    coco_path, images_dir = density_corpus
    config = _config(
        coco_path,
        images_dir,
        tmp_path / "run-density-bad",
        subset_n_images=None,
        density_bucket=[2, 2],
    )
    with pytest.raises(ValueError, match="density_bucket requires subset_n_images"):
        train(config)


# --- multi-corpus combination (binary mode) ---


def _make_standalone_corpus(root: Path, source: str, n_train: int, n_val: int) -> Path:
    """A COCO JSON using ``memoire_file_path`` (like real build_corpus.py output),
    so it needs no shared ``images_root`` — safe to combine with another corpus
    living under a completely different directory.
    """
    images_dir = root / "images"
    images_dir.mkdir(parents=True)
    images, annotations = [], []
    for i in range(n_train + n_val):
        split = "train" if i < n_train else "val"
        pixels = np.full((IMG_SIZE, IMG_SIZE, 3), 30, dtype=np.uint8)
        x0, y0, w, h = 6, 6, 24, 18
        pixels[y0 : y0 + h, x0 : x0 + w] = 220
        file_name = f"{source}_{i:03d}.png"
        file_path = images_dir / file_name
        Image.fromarray(pixels).save(file_path)
        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": IMG_SIZE,
                "height": IMG_SIZE,
                "split": split,
                "source": source,
                "memoire_image_id": f"{source}/{i:03d}",
                "group_id": f"{source}/{i:03d}",
                "memoire_file_path": str(file_path),
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
        "info": {"description": f"standalone corpus {source}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "scratch", "supercategory": "damage"}],
    }
    coco_path = root / f"{source}.json"
    coco_path.write_text(json.dumps(coco), encoding="utf-8")
    return coco_path


def test_binary_training_combines_multiple_corpora(tmp_path):
    # Two independent corpora (own images, own directories, no shared
    # images_root) mirroring a real corpus=[cardd.json, vehide.json, hitl.json]
    # config. Binary mode never touches per-corpus class ids (unlike
    # multiclass), so combining sources is safe.
    corpus_a = _make_standalone_corpus(tmp_path / "source_a", "sourcea", n_train=5, n_val=2)
    corpus_b = _make_standalone_corpus(tmp_path / "source_b", "sourceb", n_train=3, n_val=1)

    config = {
        "seed": 123,
        "corpus": [str(corpus_a), str(corpus_b)],
        "images_root": None,
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
        "iterations": 4,
        "warmup_iterations": 1,
        "val_every": 4,
        "device": "cpu",
        "subset_n_images": None,
        "output_dir": str(tmp_path / "run-multi-corpus"),
        "mlflow": {"tracking_uri": None},
    }
    summary = train(config)
    assert summary["num_train_images"] == 5 + 3
    assert summary["num_val_images"] == 2 + 1
    assert all(np.isfinite(summary["train_losses"]))


# --- copy-paste augmentation config wiring (chap. 7.3) ---


def test_train_with_copy_paste_enabled_completes(corpus, tmp_path):
    coco_path, images_dir = corpus
    config = _config(
        coco_path,
        images_dir,
        tmp_path / "run-copy-paste",
        augment=True,
        copy_paste=True,
        copy_paste_prob=1.0,
        iterations=6,
        warmup_iterations=1,
        val_every=6,
    )
    summary = train(config)
    assert len(summary["train_losses"]) == 6
    assert all(np.isfinite(summary["train_losses"]))
