"""Tests du loader CarDD : synthétiques + chiffres de contrôle sur données réelles."""

import json
import os
from pathlib import Path

import pytest

from memoire.data import cardd

REAL_ROOT = Path(os.environ.get("MEMOIRE_DATASETS", "/home/hdcc5629/memoire-datasets")) / "cardd"

CATEGORIES = [
    {"id": 1, "name": "dent", "supercategory": "car damages"},
    {"id": 2, "name": "scratch", "supercategory": "car damages"},
    {"id": 3, "name": "crack", "supercategory": "car damages"},
    {"id": 4, "name": "glass shatter", "supercategory": "car damages"},
    {"id": 5, "name": "lamp broken", "supercategory": "car damages"},
    {"id": 6, "name": "tire flat", "supercategory": "car damages"},
]

SQUARE = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
COLLINEAR = [0.0, 0.0, 5.0, 5.0, 10.0, 10.0]


def _write_split(root: Path, split: str, images: list[dict], annotations: list[dict]) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    for image in images:
        (root / split / image["file_name"]).write_bytes(b"\xff\xd8fake")
    payload = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }
    (root / f"{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _ann(ann_id: int, image_id: int, category_id: int, segmentation: list) -> dict:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": 1.0,
        "bbox": [0, 0, 1, 1],
        "iscrowd": 0,
        "attributes": {"occluded": False},
    }


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "cardd"
    _write_split(
        root,
        "train",
        images=[
            {"id": 1, "width": 100, "height": 80, "file_name": "000001.jpg", "license": 0},
            {"id": 2, "width": 60, "height": 60, "file_name": "000002.jpg", "license": 0},
        ],
        annotations=[
            _ann(1, 1, 2, [SQUARE]),
            _ann(2, 1, 1, [COLLINEAR]),  # instance entierement degeneree -> ignoree
            _ann(3, 2, 3, [SQUARE, COLLINEAR]),  # 1 anneau valide + 1 degenere
        ],
    )
    _write_split(
        root,
        "val",
        images=[{"id": 3, "width": 40, "height": 40, "file_name": "000003.jpg", "license": 0}],
        annotations=[_ann(1, 3, 6, [SQUARE])],
    )
    _write_split(
        root,
        "test",
        images=[{"id": 4, "width": 40, "height": 40, "file_name": "000004.jpg", "license": 0}],
        annotations=[_ann(1, 4, 4, [SQUARE])],
    )
    return root


def test_synthetic_record_schema(synthetic_root: Path):
    records = cardd.load_records(synthetic_root)
    assert len(records) == 4
    by_id = {r["image_id"]: r for r in records}
    rec = by_id["cardd/000001"]
    assert rec["source"] == "cardd"
    assert rec["split_hint"] == "train"
    assert rec["group_id"] == rec["image_id"]
    assert rec["width"] == 100 and rec["height"] == 80
    assert Path(rec["file_path"]) == synthetic_root / "train" / "000001.jpg"
    assert Path(rec["file_path"]).is_file()


def test_synthetic_degenerate_polygons_dropped(synthetic_root: Path):
    records = cardd.load_records(synthetic_root)
    by_id = {r["image_id"]: r for r in records}
    # Annotation 2 (3 points colineaires) ignoree entierement.
    assert len(by_id["cardd/000001"]["instances"]) == 1
    # Annotation multi-anneaux : seul l'anneau valide est conserve.
    inst = by_id["cardd/000002"]["instances"]
    assert len(inst) == 1
    assert inst[0]["polygon"] == [SQUARE]


def test_synthetic_bbox_and_area_from_polygon(synthetic_root: Path):
    records = cardd.load_records(synthetic_root)
    by_id = {r["image_id"]: r for r in records}
    inst = by_id["cardd/000001"]["instances"][0]
    assert inst["source_class"] == "scratch"
    assert inst["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert inst["area"] == pytest.approx(400.0)


def test_synthetic_split_hints(synthetic_root: Path):
    records = cardd.load_records(synthetic_root)
    hints = {r["image_id"]: r["split_hint"] for r in records}
    assert hints["cardd/000003"] == "val"
    assert hints["cardd/000004"] == "test"


def test_synthetic_rle_raises(synthetic_root: Path):
    payload = json.loads((synthetic_root / "val.json").read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = {"counts": [1, 2, 3], "size": [40, 40]}
    (synthetic_root / "val.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="RLE"):
        cardd.load_records(synthetic_root)


def test_synthetic_iscrowd_raises(synthetic_root: Path):
    payload = json.loads((synthetic_root / "test.json").read_text(encoding="utf-8"))
    payload["annotations"][0]["iscrowd"] = 1
    (synthetic_root / "test.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="iscrowd"):
        cardd.load_records(synthetic_root)


def test_synthetic_missing_image_raises(synthetic_root: Path):
    (synthetic_root / "train" / "000002.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        cardd.load_records(synthetic_root)


# --- Données réelles (chiffres de contrôle CONVENTIONS.md) ---

realdata = pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason=f"corpus CarDD absent : {REAL_ROOT}"
)


@pytest.fixture(scope="module")
def real_records() -> list[dict]:
    return cardd.load_records(REAL_ROOT)


@realdata
def test_real_image_counts(real_records):
    assert len(real_records) == 4000
    per_split = {}
    for rec in real_records:
        per_split[rec["split_hint"]] = per_split.get(rec["split_hint"], 0) + 1
    assert per_split == {"train": 2816, "val": 810, "test": 374}


@realdata
def test_real_instance_counts(real_records):
    per_split = {}
    for rec in real_records:
        per_split[rec["split_hint"]] = per_split.get(rec["split_hint"], 0) + len(rec["instances"])
    assert per_split == {"train": 6211, "val": 1744, "test": 785}
    assert sum(per_split.values()) == 8740


@realdata
def test_real_class_counts(real_records):
    counts = {}
    for rec in real_records:
        for inst in rec["instances"]:
            counts[inst["source_class"]] = counts.get(inst["source_class"], 0) + 1
    assert counts == {
        "dent": 2543,
        "scratch": 3595,
        "crack": 898,
        "glass shatter": 681,
        "lamp broken": 704,
        "tire flat": 319,
    }


@realdata
def test_real_ids_unique_and_grouping(real_records):
    ids = [r["image_id"] for r in real_records]
    assert len(set(ids)) == len(ids)
    assert all(r["group_id"] == r["image_id"] for r in real_records)
    assert all(r["source"] == "cardd" for r in real_records)


@realdata
def test_real_instances_well_formed(real_records):
    for rec in real_records:
        assert rec["instances"], f"image sans instance inattendue : {rec['image_id']}"
        for inst in rec["instances"]:
            assert inst["area"] > 0.0
            x, y, w, h = inst["bbox"]
            assert w > 0 and h > 0
            assert 0 <= x and 0 <= y
            assert x + w <= rec["width"] + 1 and y + h <= rec["height"] + 1
