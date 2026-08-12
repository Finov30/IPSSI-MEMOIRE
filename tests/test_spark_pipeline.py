"""Tests for the Spark harmonisation stage (memoire.data.spark_pipeline).

Needs a JVM. Skipped automatically when pyspark or Java is unavailable — run
for real via ``docker compose run --rm spark python -m pytest tests/test_spark_pipeline.py``.
"""

from __future__ import annotations

import pytest

pyspark = pytest.importorskip("pyspark")

from memoire.data import stats as pandas_stats
from memoire.data.splits import check_no_leak

TRIANGLE = [[0.0, 0.0, 20.0, 0.0, 20.0, 20.0]]


def make_record(image_id, group_id, n_instances=1, source="vehide", split_hint=None,
                 width=800, height=600):
    return {
        "image_id": image_id,
        "file_path": f"/data/{image_id}.jpg",
        "width": width,
        "height": height,
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


def make_corpus(n_groups=200, images_per_group=2, source="vehide", split_hint=None):
    records = []
    for g in range(n_groups):
        gid = f"{source}/session-{g:04d}"
        for i in range(images_per_group):
            records.append(
                make_record(
                    f"{source}/img-{g:04d}-{i}",
                    gid,
                    n_instances=1 + (g % 5),
                    source=source,
                    split_hint=split_hint,
                )
            )
    return records


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.appName("memoire-tests")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001 - any JVM/py4j failure means "skip", not "fail"
        pytest.skip(f"no working JVM for pyspark: {exc}")
    yield session
    session.stop()


def test_build_image_df_computes_density(spark):
    from memoire.data.spark_pipeline import build_image_df

    records = [
        make_record("a", "g1", n_instances=2, width=1000, height=1000),
        make_record("b", "g1", n_instances=0, width=500, height=500),
    ]
    rows = {r["image_id"]: r for r in build_image_df(spark, records).collect()}
    assert rows["a"]["megapixels"] == pytest.approx(1.0)
    assert rows["a"]["instances_per_megapixel"] == pytest.approx(2.0)
    assert rows["b"]["n_instances"] == 0
    assert rows["b"]["instances_per_megapixel"] == pytest.approx(0.0)


def test_aggregate_by_source_matches_pandas(spark):
    from memoire.data.spark_pipeline import aggregate_by_source, build_image_df

    records = make_corpus(n_groups=30, images_per_group=3, source="cardd")
    spark_agg = aggregate_by_source(build_image_df(spark, records))
    pandas_agg = pandas_stats.aggregate_by_source(pandas_stats.compute_stats(records))

    assert spark_agg.loc["cardd", "n_images"] == pandas_agg.loc["cardd", "n_images"]
    assert spark_agg.loc["cardd", "n_instances"] == pandas_agg.loc["cardd", "n_instances"]
    assert spark_agg.loc["cardd", "density"] == pytest.approx(pandas_agg.loc["cardd", "density"])
    assert spark_agg.loc["cardd", "mean_megapixels"] == pytest.approx(
        pandas_agg.loc["cardd", "mean_megapixels"]
    )


def test_make_group_split_spark_no_leak_and_reasonable_proportions(spark):
    from memoire.data.spark_pipeline import make_group_split_spark

    records = make_corpus(n_groups=300, images_per_group=2, source="vehide")
    splits = make_group_split_spark(spark, records, seed=42, keep_official=False, n_strata=3)

    total = sum(len(v) for v in splits.values())
    assert total == len(records)
    assert splits["train"]
    assert splits["val"]
    assert splits["test"]
    # Roughly matches DEFAULT_FRACTIONS (0.8/0.1/0.1); loose bounds, this is a
    # random stratified assignment, not an exact quota.
    assert 0.7 < len(splits["train"]) / total < 0.9

    # check_no_leak already ran inside make_group_split_spark (would have
    # raised); re-assert explicitly here as the test's own contract.
    check_no_leak(splits)
    group_of = {}
    for name, recs in splits.items():
        for rec in recs:
            assert group_of.setdefault(rec["group_id"], name) == name


def test_make_group_split_spark_keep_official(spark):
    from memoire.data.spark_pipeline import make_group_split_spark

    official = make_corpus(n_groups=20, images_per_group=2, source="cardd", split_hint="test")
    unofficial = make_corpus(n_groups=100, images_per_group=2, source="cardd", split_hint=None)
    # Distinguish group ids between the two batches to avoid accidental overlap.
    for rec in unofficial:
        rec["group_id"] = rec["group_id"].replace("session", "resplit")
        rec["image_id"] = rec["image_id"].replace("img", "resplit-img")

    splits = make_group_split_spark(
        spark, official + unofficial, seed=7, keep_official=True, n_strata=3
    )
    official_ids = {r["group_id"] for r in official}
    test_group_ids = {r["group_id"] for r in splits["test"]}
    assert official_ids <= test_group_ids


def test_make_group_split_spark_rejects_bad_fractions(spark):
    from memoire.data.spark_pipeline import make_group_split_spark

    records = make_corpus(n_groups=5, images_per_group=2)
    with pytest.raises(ValueError):
        make_group_split_spark(spark, records, fractions={"train": 0.5, "val": 0.6})
