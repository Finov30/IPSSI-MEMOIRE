"""Tests du loader HitL : synthétiques (dossiers inversés) + chiffres de contrôle réels."""

import json
import os
from pathlib import Path

import pytest

from memoire.data import hitl

REAL_ROOT = (
    Path(os.environ.get("MEMOIRE_DATASETS", "/home/hdcc5629/memoire-datasets"))
    / "humans-in-the-loop"
)

SQUARE = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
COLLINEAR = [[0.0, 0.0], [5.0, 5.0], [10.0, 10.0]]
HOLE = [[15.0, 15.0], [20.0, 15.0], [20.0, 20.0], [15.0, 20.0]]


def _meta(titles: list[str]) -> dict:
    return {
        "classes": [
            {"title": t, "shape": "polygon", "color": "#000000", "geometry_config": {}, "id": i}
            for i, t in enumerate(titles, start=1)
        ]
    }


def _obj(obj_id: int, class_title: str, exterior: list, interior: list | None = None) -> dict:
    return {
        "id": obj_id,
        "classId": 1,
        "description": "",
        "geometryType": "polygon",
        "labelerLogin": "GhazalehHITL",
        "createdAt": "2023-01-18T10:43:53.315Z",
        "updatedAt": "2023-01-18T10:43:53.315Z",
        "tags": [],
        "classTitle": class_title,
        "points": {"exterior": exterior, "interior": interior or []},
    }


def _write_image(root: Path, name: str, size: dict, objects: list[dict]) -> None:
    dataset = root / hitl.DAMAGE_DIR_NAME / "File1"
    (dataset / "img").mkdir(parents=True, exist_ok=True)
    (dataset / "ann").mkdir(parents=True, exist_ok=True)
    (dataset / "img" / name).write_bytes(b"\x89PNGfake")
    payload = {"tags": [], "description": "", "objects": objects, "size": size}
    (dataset / "ann" / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def synthetic_root(tmp_path: Path) -> Path:
    root = tmp_path / "hitl"
    (root / hitl.DAMAGE_DIR_NAME).mkdir(parents=True)
    (root / hitl.DAMAGE_DIR_NAME / "meta.json").write_text(
        json.dumps(_meta(sorted(hitl.DAMAGE_CLASSES))), encoding="utf-8"
    )
    _write_image(
        root,
        "Car damages 1.png",
        {"height": 80, "width": 100},
        [
            _obj(1, "Scratch", SQUARE),
            _obj(2, "Scratch", COLLINEAR),  # degenere -> ignore
            _obj(3, "Dent", SQUARE, interior=[HOLE]),  # trou perdu, instance gardee
        ],
    )
    _write_image(
        root,
        "Car damages 2.jpg",
        {"height": 40, "width": 60},
        [_obj(4, "Paint chip", SQUARE)],
    )
    return root


def test_synthetic_record_schema(synthetic_root: Path):
    records = hitl.load_records(synthetic_root)
    assert len(records) == 2
    by_id = {r["image_id"]: r for r in records}
    rec = by_id["hitl/Car damages 1"]
    assert rec["source"] == "hitl"
    assert rec["split_hint"] is None
    assert rec["group_id"] == rec["image_id"]
    assert rec["width"] == 100 and rec["height"] == 80
    assert Path(rec["file_path"]).name == "Car damages 1.png"
    assert Path(rec["file_path"]).is_file()
    assert by_id["hitl/Car damages 2"]["width"] == 60


def test_synthetic_degenerate_dropped_and_holes_lost(synthetic_root: Path):
    records = hitl.load_records(synthetic_root)
    by_id = {r["image_id"]: r for r in records}
    inst = by_id["hitl/Car damages 1"]["instances"]
    # Le polygone colineaire est ignore ; l'instance a trou est conservee
    # avec son seul contour exterieur.
    assert [i["source_class"] for i in inst] == ["Scratch", "Dent"]
    flat_square = [v for pt in SQUARE for v in pt]
    assert inst[1]["polygon"] == [flat_square]


def test_synthetic_bbox_and_area(synthetic_root: Path):
    records = hitl.load_records(synthetic_root)
    by_id = {r["image_id"]: r for r in records}
    inst = by_id["hitl/Car damages 2"]["instances"][0]
    assert inst["source_class"] == "Paint chip"
    assert inst["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert inst["area"] == pytest.approx(400.0)


def test_synthetic_inverted_folder_check(synthetic_root: Path):
    # Si meta.json declare des classes de pieces, le loader doit refuser :
    # c'est la protection contre l'inversion des noms de dossiers.
    (synthetic_root / hitl.DAMAGE_DIR_NAME / "meta.json").write_text(
        json.dumps(_meta(["Hood", "Fender", "Headlight"])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dommages"):
        hitl.load_records(synthetic_root)


def test_synthetic_non_polygon_geometry_raises(synthetic_root: Path):
    ann_path = (
        synthetic_root / hitl.DAMAGE_DIR_NAME / "File1" / "ann" / "Car damages 2.jpg.json"
    )
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    payload["objects"][0]["geometryType"] = "bitmap"
    ann_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="geometrie"):
        hitl.load_records(synthetic_root)


def test_synthetic_unknown_class_raises(synthetic_root: Path):
    ann_path = (
        synthetic_root / hitl.DAMAGE_DIR_NAME / "File1" / "ann" / "Car damages 2.jpg.json"
    )
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    payload["objects"][0]["classTitle"] = "Hood"
    ann_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="classe inattendue"):
        hitl.load_records(synthetic_root)


def test_synthetic_missing_folder_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        hitl.load_records(tmp_path)


# --- Données réelles (chiffres de contrôle CONVENTIONS.md) ---

realdata = pytest.mark.skipif(
    not REAL_ROOT.is_dir(), reason=f"corpus HitL absent : {REAL_ROOT}"
)


@pytest.fixture(scope="module")
def real_records() -> list[dict]:
    return hitl.load_records(REAL_ROOT)


@realdata
def test_real_image_count(real_records):
    assert len(real_records) == 814


@realdata
def test_real_instance_count(real_records):
    # 9084 instances brutes - 4 polygones degeneres (3 points colineaires,
    # aire nulle) ignores par le loader = 9080.
    assert sum(len(r["instances"]) for r in real_records) == 9080


@realdata
def test_real_class_counts(real_records):
    counts = {}
    for rec in real_records:
        for inst in rec["instances"]:
            counts[inst["source_class"]] = counts.get(inst["source_class"], 0) + 1
    # Comptages bruts moins les 4 degeneres : 3 Scratch et 1 Paint chip.
    assert counts == {
        "Scratch": 3239,
        "Dent": 1664,
        "Broken part": 1500,
        "Paint chip": 1355,
        "Missing part": 632,
        "Flaking": 337,
        "Corrosion": 277,
        "Cracked": 76,
    }


@realdata
def test_real_ids_unique_and_grouping(real_records):
    ids = [r["image_id"] for r in real_records]
    assert len(set(ids)) == len(ids)
    assert all(r["group_id"] == r["image_id"] for r in real_records)
    assert all(r["split_hint"] is None for r in real_records)
    assert all(r["source"] == "hitl" for r in real_records)


@realdata
def test_real_instances_well_formed(real_records):
    for rec in real_records:
        assert rec["instances"], f"image sans instance inattendue : {rec['image_id']}"
        for inst in rec["instances"]:
            assert inst["area"] > 0.0
            _x, _y, w, h = inst["bbox"]
            assert w > 0 and h > 0
