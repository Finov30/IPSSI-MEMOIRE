"""Tests for ``DamageSegDataset`` (synthetic COCO fixtures + real CarDD data)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from memoire.training.dataset import DamageSegDataset

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
REAL_ROOT = Path(os.environ.get("MEMOIRE_DATASETS", "/home/hdcc5629/memoire-datasets")) / "cardd"

SIZE = 64  # input_size used by the synthetic tests


def _save_png(path: Path, width: int, height: int, columns: list[tuple[int, int, int]]) -> None:
    """Write a PNG whose columns are painted per horizontal band.

    ``columns`` maps each horizontal band ``[x0, x1)`` (given as fractions of
    the full width via equal bands) to a grayscale value; here we simply take
    a list of (x0, x1, value) triples in pixels.
    """
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for x0, x1, value in columns:
        array[:, x0:x1, :] = value
    Image.fromarray(array).save(path)


def _rect_polygon(x0: float, y0: float, x1: float, y1: float) -> list[float]:
    return [x0, y0, x1, y0, x1, y1, x0, y1]


@pytest.fixture()
def synthetic(tmp_path: Path) -> dict:
    """Small COCO corpus: 3 train images, 2 val images, 2 categories.

    - a.png (40x40, train): left half white, right half black; one 'dent'
      instance on the left half (image/mask coherence for the flip test).
    - b.png (80x40, train): all white, rectangular; one full-image 'scratch'
      instance (letterbox geometry test).
    - c.png (40x40, train): all white; 'dent' on the left half, 'scratch' on
      the right half (multiclass test).
    - d.png (40x40, val): uniform gray, no annotation (empty mask test).
    - e.png (40x40, val): all white, full-image 'dent'; the entry carries a
      wrong file_name but a valid 'memoire_file_path' extra.
    """
    images_root = tmp_path / "images"
    images_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    _save_png(images_root / "a.png", 40, 40, [(0, 20, 255), (20, 40, 0)])
    _save_png(images_root / "b.png", 80, 40, [(0, 80, 255)])
    _save_png(images_root / "c.png", 40, 40, [(0, 40, 255)])
    _save_png(images_root / "d.png", 40, 40, [(0, 40, 128)])
    _save_png(elsewhere / "real_e.png", 40, 40, [(0, 40, 255)])

    coco = {
        "info": {"description": "synthetic corpus for DamageSegDataset tests"},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "a.png", "width": 40, "height": 40, "split": "train",
             "memoire_image_id": "synthetic/a"},
            {"id": 2, "file_name": "b.png", "width": 80, "height": 40, "split": "train",
             "memoire_image_id": "synthetic/b"},
            {"id": 3, "file_name": "c.png", "width": 40, "height": 40, "split": "train",
             "memoire_image_id": "synthetic/c"},
            {"id": 4, "file_name": "d.png", "width": 40, "height": 40, "split": "val",
             "memoire_image_id": "synthetic/d"},
            {"id": 5, "file_name": "missing_e.png", "width": 40, "height": 40, "split": "val",
             "memoire_image_id": "synthetic/e",
             "memoire_file_path": str(elsewhere / "real_e.png")},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1,
             "segmentation": [_rect_polygon(0, 0, 20, 40)],
             "bbox": [0, 0, 20, 40], "area": 800.0, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 2,
             "segmentation": [_rect_polygon(0, 0, 80, 40)],
             "bbox": [0, 0, 80, 40], "area": 3200.0, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1,
             "segmentation": [_rect_polygon(0, 0, 20, 40)],
             "bbox": [0, 0, 20, 40], "area": 800.0, "iscrowd": 0},
            {"id": 4, "image_id": 3, "category_id": 2,
             "segmentation": [_rect_polygon(20, 0, 40, 40)],
             "bbox": [20, 0, 20, 40], "area": 800.0, "iscrowd": 0},
            {"id": 5, "image_id": 5, "category_id": 1,
             "segmentation": [_rect_polygon(0, 0, 40, 40)],
             "bbox": [0, 0, 40, 40], "area": 1600.0, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "dent", "supercategory": "damage"},
            {"id": 2, "name": "scratch", "supercategory": "damage"},
        ],
    }
    coco_json = tmp_path / "synthetic.json"
    coco_json.write_text(json.dumps(coco), encoding="utf-8")
    return {"coco_json": coco_json, "images_root": images_root}


def _make(synthetic: dict, **kwargs) -> DamageSegDataset:
    defaults: dict = {"split": "train", "mode": "binary", "input_size": SIZE}
    defaults.update(kwargs)
    return DamageSegDataset(synthetic["coco_json"], synthetic["images_root"], **defaults)


# --- construction / split filtering ---


def test_split_filtering(synthetic):
    assert len(_make(synthetic)) == 3
    assert len(_make(synthetic, split="val")) == 2


def test_unknown_split_raises(synthetic):
    with pytest.raises(ValueError, match="split"):
        _make(synthetic, split="test")


def test_invalid_mode_raises(synthetic):
    with pytest.raises(ValueError, match="mode"):
        _make(synthetic, mode="panoptic")


def test_class_names(synthetic):
    assert _make(synthetic).class_names == ["background", "damage"]
    ds = _make(synthetic, mode="multiclass")
    assert ds.class_names == ["background", "dent", "scratch"]
    assert ds.num_classes == 3


# --- shapes, dtypes, values ---


def test_shapes_dtypes_and_ranges(synthetic):
    image, mask = _make(synthetic)[0]
    assert image.shape == (3, SIZE, SIZE) and image.dtype == torch.float32
    assert mask.shape == (SIZE, SIZE) and mask.dtype == torch.int64
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert set(mask.unique().tolist()) <= {0, 1}


def test_binary_mask_matches_annotation(synthetic):
    # a.png: instance on the left half -> after 40->64 scaling the boundary
    # lands around column 32 (one-pixel tolerance on each side).
    _, mask = _make(synthetic)[0]
    assert (mask[:, :31] == 1).all()
    assert (mask[:, 33:] == 0).all()


def test_multiclass_mask_follows_category_order(synthetic):
    _, mask = _make(synthetic, mode="multiclass")[2]  # c.png
    assert set(mask.unique().tolist()) == {1, 2}  # full-frame annotations, no background
    assert mask[32, 10] == 1  # 'dent' (category id 1) on the left
    assert mask[32, 54] == 2  # 'scratch' (category id 2) on the right


def test_empty_annotations_give_empty_mask(synthetic):
    image, mask = _make(synthetic, split="val")[0]  # d.png, no annotation
    assert int(mask.sum()) == 0
    assert abs(image.mean().item() - 128 / 255) < 1e-3  # uniform gray, no padding


def test_memoire_file_path_extra_wins_over_images_root(synthetic):
    # e.png: file_name does not exist under images_root, only the extra does.
    image, mask = _make(synthetic, split="val")[1]
    assert image.max() > 0.99
    assert (mask == 1).all()


def test_missing_images_root_raises_when_no_extra(synthetic):
    with pytest.raises(ValueError, match="images_root"):
        DamageSegDataset(synthetic["coco_json"], None, split="train", input_size=SIZE)


# --- letterbox geometry ---


def test_letterbox_rectangular_image(synthetic):
    # b.png is 80x40, all white, full-image instance. With input_size=64 the
    # content becomes 64x32 centered vertically: rows [16, 48) hold content,
    # rows [0, 16) and [48, 64) are background padding (0) in image AND mask.
    image, mask = _make(synthetic)[1]
    assert image.shape == (3, SIZE, SIZE)
    # padding bands
    assert image[:, :16, :].abs().max() == 0
    assert image[:, 48:, :].abs().max() == 0
    assert (mask[:16, :] == 0).all() and (mask[48:, :] == 0).all()
    # content band
    assert image[:, 17:47, :].min() > 0.99
    assert (mask[17:47, :] == 1).all()


def test_letterbox_square_image_has_no_padding(synthetic):
    image, _ = _make(synthetic)[2]  # c.png all white, square
    assert image.min() > 0.99


# --- augmentation ---


def test_no_augment_is_deterministic(synthetic):
    ds = _make(synthetic)
    img1, mask1 = ds[0]
    img2, mask2 = ds[0]
    assert torch.equal(img1, img2) and torch.equal(mask1, mask2)


def _flip_expected(seed: int) -> bool:
    gen = torch.Generator().manual_seed(seed)
    return bool(torch.rand((), generator=gen) < 0.5)


def test_flip_is_coherent_and_seedable(synthetic):
    base_img, base_mask = _make(synthetic)[0]
    seeds_flip = [s for s in range(50) if _flip_expected(s)]
    seeds_keep = [s for s in range(50) if not _flip_expected(s)]
    assert seeds_flip and seeds_keep  # both behaviours are exercised

    for seed in (seeds_flip[0], seeds_keep[0]):
        gen = torch.Generator().manual_seed(seed)
        image, mask = _make(synthetic, augment=True, generator=gen)[0]
        if _flip_expected(seed):
            assert torch.equal(image, torch.flip(base_img, dims=[-1]))
            assert torch.equal(mask, torch.flip(base_mask, dims=[-1]))
        else:
            assert torch.equal(image, base_img)
            assert torch.equal(mask, base_mask)


def test_flip_keeps_image_and_mask_aligned(synthetic):
    # a.png: bright pixels and damage pixels are the same left half, so after
    # any flip the mask must still sit exactly on the bright region.
    gen = torch.Generator().manual_seed(0)
    ds = _make(synthetic, augment=True, generator=gen)
    for _ in range(4):  # several draws -> both flipped and unflipped items
        image, mask = ds[0]
        bright = image.mean(dim=0) > 0.5
        damage = mask == 1
        # boundary column can differ by one pixel after resampling
        agreement = (bright == damage).float().mean().item()
        assert agreement > 0.95


# --- real data ---

realdata = pytest.mark.skipif(
    not (PROCESSED / "cardd.json").is_file() or not REAL_ROOT.is_dir(),
    reason=f"processed corpus or CarDD images missing ({PROCESSED}, {REAL_ROOT})",
)


@realdata
@pytest.mark.realdata
def test_real_cardd_items(tmp_path):
    data = json.loads((PROCESSED / "cardd.json").read_text(encoding="utf-8"))
    annotated_ids = {ann["image_id"] for ann in data["annotations"]}

    selected = []
    for img in data["images"]:
        if img["id"] not in annotated_ids:
            continue
        path = None
        extra = img.get("memoire_file_path")
        if extra and Path(extra).is_file():
            path = Path(extra)
        else:
            for sub in ("train", "val", "test"):
                candidate = REAL_ROOT / sub / img["file_name"]
                if candidate.is_file():
                    path = candidate
                    break
        if path is None:
            continue
        img = dict(img)
        img["split"] = "train"
        img["memoire_file_path"] = str(path)
        selected.append(img)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        pytest.skip("could not resolve 3 annotated CarDD image files on disk")

    ids = {img["id"] for img in selected}
    subset = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "images": selected,
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in ids],
        "categories": data["categories"],
    }
    coco_json = tmp_path / "cardd_subset.json"
    coco_json.write_text(json.dumps(subset), encoding="utf-8")

    ds = DamageSegDataset(coco_json, images_root=None, split="train",
                          mode="binary", input_size=256)
    assert len(ds) == 3
    non_empty = 0
    for i in range(3):
        image, mask = ds[i]
        assert image.shape == (3, 256, 256) and image.dtype == torch.float32
        assert mask.shape == (256, 256) and mask.dtype == torch.int64
        assert set(mask.unique().tolist()) <= {0, 1}
        non_empty += int(bool((mask == 1).any()))
    assert non_empty >= 1
