"""Tests du chargeur VehiDE (échantillon synthétique + contrôle sur données réelles)."""

import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from memoire.data.vehide import (
    extract_group_id,
    flatten_polygon,
    link_thumbnail_groups,
    load_records,
    merge_session_groups,
    parse_regions,
    parse_timestamp,
    polygon_area,
    polygon_bbox,
)

REAL_ROOT = Path("/home/hdcc5629/memoire-datasets/vehide")


# ---------------------------------------------------------------------------
# Fonctions pures
# ---------------------------------------------------------------------------


def test_flatten_polygon_interleaves_coordinates():
    assert flatten_polygon([1, 2, 3], [4, 5, 6]) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_flatten_polygon_rejects_length_mismatch():
    with pytest.raises(ValueError):
        flatten_polygon([1, 2], [3])


def test_polygon_area_square():
    # Carré 10x10 : aire 100 quelle que soit l'orientation du contour.
    square = [0, 0, 10, 0, 10, 10, 0, 10]
    assert polygon_area(square) == 100.0
    assert polygon_area(list(reversed(square))) == 100.0


def test_polygon_area_degenerate():
    assert polygon_area([493, 458, 493, 457, 493, 457]) == 0.0  # points colinéaires
    assert polygon_area([1, 1, 2, 2]) == 0.0  # < 3 points


def test_polygon_bbox():
    assert polygon_bbox([2, 3, 10, 5, 6, 20]) == [2, 3, 8, 17]


def test_extract_group_id():
    assert extract_group_id("01012020_172204image853193.jpg") == "01012020_172204"
    assert extract_group_id("16012020_104036image125354.jpg") == "16012020_104036"
    # Noms hors motif horodaté : singleton = nom de fichier entier.
    assert extract_group_id("z1692240140997_0a1b2c.jpg") == "z1692240140997_0a1b2c.jpg"
    assert extract_group_id("IMG_0042.jpg") == "IMG_0042.jpg"


def test_parse_regions_drops_degenerate_and_counts():
    regions = [
        {"all_x": [0, 10, 10, 0], "all_y": [0, 0, 10, 10], "class": "tray_son"},
        {"all_x": [493, 493, 493], "all_y": [458, 457, 457], "class": "tray_son"},
    ]
    instances, dropped = parse_regions(regions)
    assert dropped == 1
    assert len(instances) == 1
    inst = instances[0]
    assert inst["source_class"] == "tray_son"
    assert inst["polygon"] == [[0.0, 0.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0]]
    assert inst["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert inst["area"] == 100.0


def test_parse_regions_drops_polygon_fully_outside_image():
    # Reproduit l'annotation fantôme réelle : x dans [2142, 2860] pour w=1632
    # (masque rasterisé vide malgré une aire de lacet non nulle).
    regions = [
        {"all_x": [2142, 2860, 2860, 2142], "all_y": [284, 284, 1077, 1077], "class": "rach"},
        {"all_x": [0, 10, 10, 0], "all_y": [0, 0, 10, 10], "class": "rach"},
    ]
    instances, dropped = parse_regions(regions, width=1632, height=1224)
    assert dropped == 1
    assert len(instances) == 1
    # Sans dimensions fournies, pas de contrôle de cadre (comportement historique).
    instances, dropped = parse_regions(regions)
    assert dropped == 0
    assert len(instances) == 2


def test_parse_regions_keeps_polygon_partially_outside_image():
    regions = [{"all_x": [-5, 10, 10, -5], "all_y": [0, 0, 10, 10], "class": "mop_lom"}]
    instances, dropped = parse_regions(regions, width=20, height=20)
    assert dropped == 0
    assert len(instances) == 1


def test_parse_timestamp_both_formats():
    assert parse_timestamp("06012020_095245") is not None  # DDMMYYYY
    assert parse_timestamp("20200130_190111") is not None  # YYYYMMDD
    assert parse_timestamp("z1692240140997_0a1b2c.jpg") is None


def test_merge_session_groups_clusters_nearby_timestamps():
    # Cas réel : 08:13:42 / 08:13:43 / 08:13:48, même véhicule (plaque 077742).
    records = [
        {"group_id": "02012020_081342", "file_path": "a.jpg"},
        {"group_id": "02012020_081343", "file_path": "b.jpg"},
        {"group_id": "02012020_081348", "file_path": "c.jpg"},
        {"group_id": "02012020_100000", "file_path": "d.jpg"},  # > 60 s : autre session
        {"group_id": "IMG_0042.jpg", "file_path": "e.jpg"},  # non horodaté : inchangé
    ]
    merge_session_groups(records)
    assert records[0]["group_id"] == records[1]["group_id"] == records[2]["group_id"]
    assert records[3]["group_id"] != records[0]["group_id"]
    assert records[4]["group_id"] == "IMG_0042.jpg"


def test_link_thumbnail_groups_reuses_twin_group():
    records = [
        {"group_id": "18.jpg", "file_path": "/data/image/image/18.jpg"},
        {"group_id": "Thumbnail18.jpg", "file_path": "/data/validation/validation/Thumbnail18.jpg"},
        {"group_id": "ThumbnailX.jpg", "file_path": "/data/image/image/ThumbnailX.jpg"},  # sans jumeau
    ]
    link_thumbnail_groups(records)
    assert records[1]["group_id"] == "18.jpg"
    assert records[2]["group_id"] == "ThumbnailX.jpg"


# ---------------------------------------------------------------------------
# Mini-dataset synthétique (aucune dépendance aux données réelles)
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_root(tmp_path):
    """Reproduit la mise en page VehiDE : JSON à la racine, images sous des dossiers doublés."""
    train_dir = tmp_path / "image" / "image"
    val_dir = tmp_path / "validation" / "validation"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    train = {
        "01012020_172204image853193.jpg": {
            "name": "01012020_172204image853193.jpg",
            "regions": [
                {"all_x": [0, 20, 20, 0], "all_y": [0, 0, 10, 10], "class": "mop_lom"},
                {"all_x": [5, 5, 5], "all_y": [1, 2, 2], "class": "tray_son"},  # dégénéré
            ],
        },
        "01012020_172204image999999.jpg": {  # même session -> même group_id
            "name": "01012020_172204image999999.jpg",
            "regions": [
                {"all_x": [1, 4, 1], "all_y": [1, 1, 5], "class": "rach"},
            ],
        },
        "z1692240140997_deadbeef.jpg": {  # hors motif -> group_id = nom de fichier
            "name": "z1692240140997_deadbeef.jpg",
            "regions": [
                {"all_x": [2, 8, 8, 2], "all_y": [2, 2, 6, 6], "class": "vo_kinh"},
            ],
        },
    }
    val = {
        "02022020_080000image1.jpg": {
            "name": "02022020_080000image1.jpg",
            "regions": [
                {"all_x": [0, 6, 3], "all_y": [0, 0, 9], "class": "thung"},
            ],
        },
    }

    (tmp_path / "0Train_via_annos.json").write_text(json.dumps(train), encoding="utf-8")
    (tmp_path / "0Val_via_annos.json").write_text(json.dumps(val), encoding="utf-8")

    sizes = {
        train_dir / "01012020_172204image853193.jpg": (32, 24),
        train_dir / "01012020_172204image999999.jpg": (16, 16),
        train_dir / "z1692240140997_deadbeef.jpg": (20, 10),
        val_dir / "02022020_080000image1.jpg": (12, 18),
    }
    for path, size in sizes.items():
        Image.new("RGB", size, color=(80, 80, 80)).save(path)
    return tmp_path


def test_load_records_synthetic(synthetic_root, caplog):
    with caplog.at_level(logging.WARNING, logger="memoire.data.vehide"):
        records = load_records(synthetic_root)

    assert len(records) == 4
    by_id = {r["image_id"]: r for r in records}
    assert set(by_id) == {
        "vehide/01012020_172204image853193",
        "vehide/01012020_172204image999999",
        "vehide/z1692240140997_deadbeef",
        "vehide/02022020_080000image1",
    }

    first = by_id["vehide/01012020_172204image853193"]
    assert first["source"] == "vehide"
    assert first["split_hint"] == "train"
    assert (first["width"], first["height"]) == (32, 24)
    assert first["group_id"] == "01012020_172204"
    assert Path(first["file_path"]).is_absolute()
    assert Path(first["file_path"]).exists()
    # Le polygone dégénéré est ignoré : une seule instance survit.
    assert len(first["instances"]) == 1
    assert first["instances"][0]["source_class"] == "mop_lom"
    assert first["instances"][0]["area"] == 200.0
    assert first["instances"][0]["bbox"] == [0.0, 0.0, 20.0, 10.0]

    # Deux photos de la même session partagent le group_id.
    assert by_id["vehide/01012020_172204image999999"]["group_id"] == "01012020_172204"
    # Nom hors motif : group_id singleton = nom de fichier.
    assert by_id["vehide/z1692240140997_deadbeef"]["group_id"] == "z1692240140997_deadbeef.jpg"

    val_rec = by_id["vehide/02022020_080000image1"]
    assert val_rec["split_hint"] == "val"
    assert (val_rec["width"], val_rec["height"]) == (12, 18)

    # Le compteur de polygones dégénérés est loggé, jamais silencieux.
    assert any("degenerate" in message for message in caplog.messages)
    assert any("1" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# Chiffres de contrôle sur les données réelles (optionnel)
# ---------------------------------------------------------------------------


@pytest.mark.realdata
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real VehiDE dataset not available")
def test_control_figures_real_dataset():
    records = load_records(REAL_ROOT)

    # Images : 13 945 = 11 621 train + 2 324 val.
    assert len(records) == 13_945
    splits = [r["split_hint"] for r in records]
    assert splits.count("train") == 11_621
    assert splits.count("val") == 2_324

    # Instances : 36 081 brutes - 1 polygone dégénéré (aire nulle)
    # - 1 polygone entièrement hors cadre (masque vide) = 36 079.
    total_instances = sum(len(r["instances"]) for r in records)
    assert total_instances == 36_079

    # Distribution de classes attendue (écartés : 1 'tray_son' dégénéré,
    # 1 'rach' hors cadre).
    counts: dict[str, int] = {}
    for record in records:
        for instance in record["instances"]:
            counts[instance["source_class"]] = counts.get(instance["source_class"], 0) + 1
    assert counts == {
        "tray_son": 14_646,  # 14 647 - 1 dégénéré
        "mop_lom": 5_681,
        "rach": 5_508,  # 5 509 - 1 hors cadre
        "mat_bo_phan": 2_818,
        "be_den": 2_782,
        "thung": 2_423,
        "vo_kinh": 2_221,
    }

    # Sanité du schéma sur l'ensemble.
    for record in records:
        assert record["width"] > 0 and record["height"] > 0
        assert record["group_id"]
        for instance in record["instances"]:
            assert instance["area"] > 0.0
            assert len(instance["polygon"][0]) >= 6
