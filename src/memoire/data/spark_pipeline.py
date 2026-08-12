"""PySpark harmonisation stage (mémoire, chap. 5.2) : calcul de densité
d'instances par image et échantillonnage stratifié reproductible.

Justification (sujet v2, §4bis) : le volume ne justifie pas Spark (~17 000
images tiennent en pandas). Ce qui le justifie est le régime de production visé
et la nécessité de rendre la mesure — densité d'instances par image, la
variable centrale de l'étude — reproductible à l'échelle. Ce module est
volontairement la seule partie du pipeline qui touche à Spark : le parsing par
format (VIA/COCO/Supervisely, propre à chaque source) et l'export COCO restent
dans leurs modules Python existants (``memoire.data.{vehide,cardd,hitl}`` et
``memoire.data.coco_export``) — ce n'est pas un problème de calcul distribué.

Miroir de contrat de ``memoire.data.splits.make_group_split`` : même signature
logique, même garantie anti-fuite (revalidée par l'implémentation Python déjà
testée avant de retourner), calcul de la stratification par densité effectué
via des transformations Spark (agrégations, fenêtres) plutôt que pandas.

Nécessite un JVM (Java) — à exécuter dans le service Docker ``spark``
(``docker compose run --rm spark ...``), pas forcément sur l'hôte.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from memoire.data.splits import DEFAULT_FRACTIONS, check_no_leak

# Explicit schema, not inferred: a corpus with no official splits (e.g. HitL)
# has split_hint=None on every row, and Spark cannot infer a type from an
# all-null sample.
IMAGE_ROW_SCHEMA = T.StructType(
    [
        T.StructField("image_id", T.StringType(), False),
        T.StructField("source", T.StringType(), False),
        T.StructField("group_id", T.StringType(), False),
        T.StructField("split_hint", T.StringType(), True),
        T.StructField("width", T.IntegerType(), False),
        T.StructField("height", T.IntegerType(), False),
        T.StructField("n_instances", T.IntegerType(), False),
    ]
)


def records_to_rows(records: Sequence[dict]) -> list[dict]:
    """One row per image: identity, dimensions, instance count (no PII, no geometry)."""
    return [
        {
            "image_id": rec["image_id"],
            "source": rec["source"],
            "group_id": rec["group_id"],
            "split_hint": rec.get("split_hint"),
            "width": int(rec["width"]),
            "height": int(rec["height"]),
            "n_instances": len(rec["instances"]),
        }
        for rec in records
    ]


def build_image_df(spark: SparkSession, records: Sequence[dict]) -> DataFrame:
    """One row per image, with megapixels and density (chap. 4.6 control metric)."""
    df = spark.createDataFrame(records_to_rows(records), schema=IMAGE_ROW_SCHEMA)
    return df.withColumn(
        "megapixels", F.col("width") * F.col("height") / F.lit(1_000_000.0)
    ).withColumn(
        "instances_per_megapixel",
        F.when(F.col("megapixels") > 0, F.col("n_instances") / F.col("megapixels")).otherwise(
            F.lit(0.0)
        ),
    )


def aggregate_by_source(image_df: DataFrame):
    """Per-corpus aggregates (pandas DataFrame, same shape as memoire.data.stats)."""
    return (
        image_df.groupBy("source")
        .agg(
            F.count("*").alias("n_images"),
            F.sum("n_instances").alias("n_instances"),
            F.avg("n_instances").alias("density"),
            F.avg("megapixels").alias("mean_megapixels"),
        )
        .orderBy("source")
        .toPandas()
        .set_index("source")
    )


def _stable_hash_col(seed: int):
    """Deterministic per-group hash: independent of Spark's execution/partitioning order."""
    return F.sha2(F.concat(F.lit(f"{seed}:"), F.col("group_id")), 256)


def make_group_split_spark(
    spark: SparkSession,
    records: Sequence[dict],
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
    keep_official: bool = False,
    n_strata: int = 3,
) -> dict[str, list[dict]]:
    """Spark-computed density-stratified group split; same anti-leak contract
    as ``memoire.data.splits.make_group_split``.

    - Never splits by image, only by ``group_id`` (vehicle/incident).
    - ``keep_official=True``: groups whose images carry a single official
      ``split_hint`` (CarDD) stay there; the rest are stratified by density.
    - Stratification: groups bucketed into ``n_strata`` quantile buckets of
      mean instance density, then assigned train/val/test within each bucket
      by cumulative image-count fraction, ordered by a seeded stable hash
      (reproducible, independent of Spark partition ordering).
    - ``check_no_leak`` (the already-tested pure-Python guard) re-validates
      the result before returning — Spark computes the split, it does not
      police the anti-leak guarantee itself.
    """
    fractions = dict(fractions) if fractions is not None else dict(DEFAULT_FRACTIONS)
    if not fractions or any(f < 0 for f in fractions.values()):
        raise ValueError(f"fractions must be non-negative and non-empty, got {fractions}")
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions.values())}")
    split_names = list(fractions)

    image_df = spark.createDataFrame(records_to_rows(records), schema=IMAGE_ROW_SCHEMA)
    group_df = image_df.groupBy("group_id").agg(
        F.avg("n_instances").alias("group_density"),
        F.first("split_hint", ignorenulls=True).alias("split_hint"),
        F.sum("n_instances").alias("group_n_instances"),
        F.count("*").alias("group_n_images"),
    )

    if keep_official:
        official_df = group_df.filter(F.col("split_hint").isin(split_names)).select(
            "group_id", F.col("split_hint").alias("split")
        )
        remaining_df = group_df.filter(~F.col("split_hint").isin(split_names) | F.col("split_hint").isNull())
    else:
        official_df = spark.createDataFrame([], schema="group_id string, split string")
        remaining_df = group_df

    if remaining_df.head(1):
        strata_window = Window.orderBy("group_density", "group_id")
        stratified = remaining_df.withColumn(
            "stratum", F.ntile(max(1, min(n_strata, remaining_df.count()))).over(strata_window)
        ).withColumn("hash_key", _stable_hash_col(seed))

        order_window = (
            Window.partitionBy("stratum")
            .orderBy("hash_key", "group_id")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
        total_window = Window.partitionBy("stratum")
        stratified = stratified.withColumn(
            "cum_images", F.sum("group_n_images").over(order_window)
        ).withColumn("stratum_total_images", F.sum("group_n_images").over(total_window))
        stratified = stratified.withColumn(
            "placed_before", F.col("cum_images") - F.col("group_n_images")
        ).withColumn(
            "cum_fraction",
            F.col("placed_before") / F.col("stratum_total_images"),
        )

        cumulative = 0.0
        boundaries: list[tuple[str, float]] = []
        for name in split_names:
            cumulative += fractions[name]
            boundaries.append((name, cumulative))
        boundaries[-1] = (boundaries[-1][0], 1.0 + 1e-9)

        split_expr = None
        for name, upper in reversed(boundaries):
            branch = F.when(F.col("cum_fraction") < F.lit(upper), F.lit(name))
            split_expr = branch if split_expr is None else branch.otherwise(split_expr)
        stratified_assignment = stratified.select("group_id", split_expr.alias("split"))
    else:
        stratified_assignment = spark.createDataFrame([], schema="group_id string, split string")

    assignment_df = official_df.unionByName(stratified_assignment)
    assignment = {row["group_id"]: row["split"] for row in assignment_df.collect()}

    splits: dict[str, list[dict]] = {name: [] for name in split_names}
    for rec in records:
        splits[assignment[rec["group_id"]]].append(rec)

    check_no_leak(splits)
    return splits
