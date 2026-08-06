"""Tests du split anti-fuite par group_id."""

import pytest

from memoire.data.splits import check_no_leak, make_group_split

TRIANGLE = [[0.0, 0.0, 20.0, 0.0, 20.0, 20.0]]


def make_record(image_id, group_id, n_instances=1, source="vehide", split_hint=None):
    return {
        "image_id": image_id,
        "file_path": f"/data/{image_id}.jpg",
        "width": 800,
        "height": 600,
        "source": source,
        "split_hint": split_hint,
        "group_id": group_id,
        "instances": [
            {
                "source_class": "scratch",
                "polygon": TRIANGLE,
                "bbox": [0.0, 0.0, 20.0, 20.0],
                "area": 200.0,
            }
            for _ in range(n_instances)
        ],
    }


def make_corpus(n_groups=200, images_per_group=2):
    records = []
    for g in range(n_groups):
        gid = f"vehide/session-{g:04d}"
        for i in range(images_per_group):
            records.append(
                make_record(f"vehide/img-{g:04d}-{i}", gid, n_instances=1 + (g % 5))
            )
    return records


def test_check_no_leak_raises_on_shared_group():
    leaked = make_record("vehide/img-a", "vehide/veh-1")
    splits = {
        "train": [leaked, make_record("vehide/img-b", "vehide/veh-2")],
        "val": [make_record("vehide/img-c", "vehide/veh-1")],
    }
    with pytest.raises(AssertionError, match="veh-1"):
        check_no_leak(splits)


def test_check_no_leak_passes_on_disjoint_groups():
    splits = {
        "train": [make_record("vehide/img-a", "vehide/veh-1")],
        "val": [make_record("vehide/img-b", "vehide/veh-2")],
    }
    check_no_leak(splits)


def test_make_group_split_has_no_leak_and_keeps_all_records():
    records = make_corpus()
    splits = make_group_split(records, seed=42)
    check_no_leak(splits)
    all_ids = sorted(r["image_id"] for recs in splits.values() for r in recs)
    assert all_ids == sorted(r["image_id"] for r in records)


def test_groups_never_straddle_two_splits():
    records = make_corpus(n_groups=100, images_per_group=3)
    splits = make_group_split(records, seed=7)
    for gid in {r["group_id"] for r in records}:
        homes = [name for name, recs in splits.items() if any(r["group_id"] == gid for r in recs)]
        assert len(homes) == 1


def test_fractions_are_approximately_respected():
    records = make_corpus(n_groups=300)
    splits = make_group_split(
        records, fractions={"train": 0.8, "val": 0.1, "test": 0.1}, seed=0
    )
    total = len(records)
    assert len(splits["train"]) / total == pytest.approx(0.8, abs=0.05)
    assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.05)
    assert len(splits["test"]) / total == pytest.approx(0.1, abs=0.05)


def test_split_is_deterministic_at_fixed_seed():
    records = make_corpus()
    first = make_group_split(records, seed=123)
    second = make_group_split(records, seed=123)
    for name in first:
        assert [r["image_id"] for r in first[name]] == [r["image_id"] for r in second[name]]


def test_different_seeds_give_different_assignments():
    records = make_corpus(n_groups=300)
    a = make_group_split(records, seed=1)
    b = make_group_split(records, seed=2)
    assert any(
        {r["image_id"] for r in a[name]} != {r["image_id"] for r in b[name]} for name in a
    )


def test_keep_official_preserves_cardd_hints():
    records = []
    for i in range(60):
        hint = "train" if i < 40 else ("val" if i < 50 else "test")
        image_id = f"cardd/img-{i:04d}"
        records.append(
            make_record(image_id, group_id=image_id, source="cardd", split_hint=hint)
        )
    records += make_corpus(n_groups=40, images_per_group=1)

    splits = make_group_split(records, seed=0, keep_official=True)
    check_no_leak(splits)
    for name, recs in splits.items():
        for rec in recs:
            if rec["split_hint"] is not None:
                assert name == rec["split_hint"]
    assert sum(len(recs) for recs in splits.values()) == len(records)


def test_invalid_fractions_rejected():
    records = make_corpus(n_groups=5)
    with pytest.raises(ValueError):
        make_group_split(records, fractions={"train": 0.7, "val": 0.7})


def test_stratification_balances_density_across_splits():
    records = []
    for g in range(300):
        gid = f"vehide/session-{g:04d}"
        n_instances = 1 if g < 150 else 12
        records.append(make_record(f"vehide/img-{g:04d}", gid, n_instances=n_instances))
    splits = make_group_split(records, seed=3)
    densities = {
        name: sum(len(r["instances"]) for r in recs) / len(recs)
        for name, recs in splits.items()
    }
    overall = sum(len(r["instances"]) for r in records) / len(records)
    for value in densities.values():
        assert value == pytest.approx(overall, rel=0.35)
