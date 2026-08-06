"""Tests des statistiques par corpus et de l'export COCO."""

import numpy as np
import pytest
from pycocotools.coco import COCO

from memoire.data.coco_export import export_coco, records_to_coco
from memoire.data.stats import aggregate_by_source, compute_stats

SQUARE = [[10.0, 10.0, 60.0, 10.0, 60.0, 60.0, 10.0, 60.0]]


def make_record(image_id, source, n_instances, width=1000, height=1000, classes=("scratch",)):
    return {
        "image_id": image_id,
        "file_path": f"/data/{image_id}.jpg",
        "width": width,
        "height": height,
        "source": source,
        "split_hint": None,
        "group_id": image_id,
        "instances": [
            {
                "source_class": classes[i % len(classes)],
                "polygon": SQUARE,
                "bbox": [10.0, 10.0, 50.0, 50.0],
                "area": 2500.0,
            }
            for i in range(n_instances)
        ],
    }


def make_corpus(source, n_images, n_instances_total, **kwargs):
    """Corpus synthétique avec un nombre total d'instances exact."""
    base, extra = divmod(n_instances_total, n_images)
    return [
        make_record(f"{source}/img-{i:05d}", source, base + (1 if i < extra else 0), **kwargs)
        for i in range(n_images)
    ]


def test_compute_stats_per_image_columns():
    records = [
        make_record("vehide/img-a", "vehide", 3, width=2000, height=1500,
                    classes=("scratch", "dent")),
        make_record("vehide/img-b", "vehide", 0),
    ]
    df = compute_stats(records)
    assert list(df["image_id"]) == ["vehide/img-a", "vehide/img-b"]
    row = df.iloc[0]
    assert row["n_instances"] == 3
    assert row["megapixels"] == pytest.approx(3.0)
    assert row["instances_per_megapixel"] == pytest.approx(1.0)
    assert row["classes"] == "dent|scratch"
    assert df.iloc[1]["n_instances"] == 0
    assert df.iloc[1]["classes"] == ""


def test_aggregate_densities_match_expected_corpus_values():
    records = (
        make_corpus("vehide", 100, 259)
        + make_corpus("cardd", 100, 219)
        + make_corpus("hitl", 50, 558)
    )
    agg = aggregate_by_source(compute_stats(records))
    assert agg.loc["vehide", "density"] == pytest.approx(2.59, abs=0.005)
    assert agg.loc["cardd", "density"] == pytest.approx(2.19, abs=0.005)
    assert agg.loc["hitl", "density"] == pytest.approx(11.16, abs=0.005)
    assert agg.loc["vehide", "n_images"] == 100
    assert agg.loc["vehide", "n_instances"] == 259


def test_aggregate_counts_distinct_classes():
    records = [
        make_record("hitl/img-a", "hitl", 4, classes=("Scratch", "Dent")),
        make_record("hitl/img-b", "hitl", 2, classes=("Corrosion",)),
    ]
    agg = aggregate_by_source(compute_stats(records))
    assert agg.loc["hitl", "n_classes"] == 3


def test_coco_export_is_reloadable_by_pycocotools(tmp_path):
    records = [
        make_record("cardd/img-a", "cardd", 2, classes=("dent", "scratch")),
        make_record("cardd/img-b", "cardd", 1, classes=("scratch",)),
        make_record("vehide/img-c", "vehide", 3, classes=("crack",)),
    ]
    path = tmp_path / "annotations.json"
    export_coco(records, path)

    coco = COCO(str(path))
    assert len(coco.imgs) == 3
    assert len(coco.anns) == 6
    assert sorted(c["name"] for c in coco.cats.values()) == ["crack", "dent", "scratch"]

    ann_ids = coco.getAnnIds(imgIds=[1])
    anns = coco.loadAnns(ann_ids)
    assert anns and all(a["iscrowd"] == 0 for a in anns)
    mask = coco.annToMask(anns[0])
    assert mask.shape == (1000, 1000)
    assert int(np.sum(mask)) > 0


def test_coco_ids_are_stable_integers():
    records = [
        make_record("vehide/img-b", "vehide", 1),
        make_record("vehide/img-a", "vehide", 1),
    ]
    first = records_to_coco(records)
    second = records_to_coco(list(reversed(records)))
    assert first == second
    assert [img["id"] for img in first["images"]] == [1, 2]
    assert first["images"][0]["memoire_image_id"] == "vehide/img-a"
    assert all(isinstance(ann["id"], int) for ann in first["annotations"])


def test_coco_export_applies_taxonomy_mapping():
    records = [
        make_record("hitl/img-a", "hitl", 2, classes=("Scratch", "Paint chip")),
        make_record("hitl/img-b", "hitl", 1, classes=("Missing part",)),
    ]

    def mapping_fn(source_class):
        mapping = {"Scratch": "scratch", "Paint chip": "scratch", "Missing part": None}
        return mapping[source_class]

    coco = records_to_coco(records, mapping_fn=mapping_fn)
    assert [c["name"] for c in coco["categories"]] == ["scratch"]
    assert len(coco["annotations"]) == 2
    assert len(coco["images"]) == 2


def test_coco_export_rejects_duplicate_image_ids():
    records = [
        make_record("vehide/img-a", "vehide", 1),
        make_record("vehide/img-a", "vehide", 2),
    ]
    with pytest.raises(ValueError, match="duplicate image_id"):
        records_to_coco(records)
