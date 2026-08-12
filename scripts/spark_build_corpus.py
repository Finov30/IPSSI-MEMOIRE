#!/usr/bin/env python
"""Construit le corpus unifié — variante Spark (mémoire, chap. 5.2).

Reprend ``scripts/build_corpus.py`` à l'identique pour tout ce qui n'est pas
le calcul de densité et le split stratifié (chargement par corpus, taxonomie,
export COCO, rapport) : seules ces deux étapes tournent sur Spark
(``memoire.data.spark_pipeline``) au lieu de pandas.

Nécessite un JVM — à exécuter dans le service Docker ``spark`` :
    docker compose run --rm spark python scripts/spark_build_corpus.py \
        --datasets-root /data/memoire-datasets --out data/processed

Usage (identique à build_corpus.py) :
    uv run python scripts/spark_build_corpus.py [--datasets-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import time
from pathlib import Path

from pyspark.sql import SparkSession

from memoire.data.coco_export import export_coco
from memoire.data.spark_pipeline import aggregate_by_source, build_image_df, make_group_split_spark
from memoire.data.splits import check_no_leak

logger = logging.getLogger("spark_build_corpus")

# scripts/ has no __init__.py (thin-CLI convention): load build_corpus.py as a
# module to reuse its loading/taxonomy/report helpers instead of duplicating
# them — only the density/split computation differs between the two scripts.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("build_corpus", _SCRIPTS_DIR / "build_corpus.py")
build_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_corpus)


def build_splits_spark(
    spark: SparkSession, corpora: dict[str, list[dict]]
) -> dict[str, dict[str, list[dict]]]:
    """Spark equivalent of build_corpus.build_splits."""
    splits_by_source: dict[str, dict[str, list[dict]]] = {}
    for source, records in corpora.items():
        splits_by_source[source] = make_group_split_spark(
            spark,
            records,
            seed=build_corpus.SPLIT_SEED,
            keep_official=(source == "cardd"),
        )
    merged: dict[str, list[dict]] = {}
    for per_source in splits_by_source.values():
        for name, recs in per_source.items():
            merged.setdefault(name, []).extend(recs)
    check_no_leak(merged)
    return splits_by_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=build_corpus.DEFAULT_DATASETS_ROOT)
    parser.add_argument("--out", type=Path, default=build_corpus.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = SparkSession.builder.appName("memoire-build-corpus").master("local[*]").getOrCreate()
    try:
        taxonomy = build_corpus.load_taxonomy(build_corpus.TAXONOMY_PATH)
        corpora, metrics = build_corpus.load_all(args.datasets_root)
        build_corpus.check_taxonomy_coverage(taxonomy, corpora)

        splits_by_source = build_splits_spark(spark, corpora)
        split_of_image = {
            rec["image_id"]: name
            for per_source in splits_by_source.values()
            for name, recs in per_source.items()
            for rec in recs
        }

        all_records = [rec for records in corpora.values() for rec in records]
        for rec in all_records:
            rec["split"] = split_of_image[rec["image_id"]]

        image_df = build_image_df(spark, all_records)
        per_source_agg = aggregate_by_source(image_df)
        per_image_pd = image_df.toPandas()
        per_image_pd["split"] = per_image_pd["image_id"].map(split_of_image)
        per_image_pd.to_parquet(out_dir / "stats.parquet", index=False)

        export_seconds: dict[str, float] = {}
        for source, records in corpora.items():
            started = time.perf_counter()
            export_coco(
                records, out_dir / f"{source}.json", build_corpus.make_mapping_fn(taxonomy, source)
            )
            export_seconds[source] = time.perf_counter() - started
            logger.info("%s.json exporté en %.1fs", source, export_seconds[source])

        canonical_counts = build_corpus.canonical_instance_counts(taxonomy, corpora)
        discrepancies = build_corpus.compare_to_controls(corpora, metrics)
        report_path = build_corpus.write_report(
            out_dir,
            corpora,
            metrics,
            per_source_agg,
            canonical_counts,
            splits_by_source,
            discrepancies,
            export_seconds,
        )
        logger.info("Rapport écrit (Spark) : %s", report_path)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
