#!/usr/bin/env python
"""Point de croisement Python pur / Spark sur le split stratifié (chap. 5.2/8.6).

Usage (nécessite une JVM -> via le service docker, comme spark_build_corpus) :
    docker compose run --rm spark python scripts/bench_split_scaling.py \
        [--scales 20000,50000,100000,250000,500000,1000000] [--repeats 3] \
        [--out figures/bench_split.csv]

Le sujet (01-Sujet-memoire.md, §4bis) assume que Spark est sur-dimensionné pour
~19 000 images et propose en « piste bonus » de mesurer le point de croisement
pandas/Spark sur ce traitement précis. Ce script mesure exactement cela : le
temps du split stratifié par densité anti-fuite (le traitement central confié à
Spark, chap. 5.2) sur des corpus synthétiques de taille croissante, via les deux
implémentations au contrat identique — `memoire.data.splits.make_group_split`
(référence Python pur) et `memoire.data.spark_pipeline.make_group_split_spark`.

Les enregistrements synthétiques reproduisent la structure réelle (group_id
multi-images ~2 images/groupe, densités hétérogènes, 10 % de split_hint
officiels) ; le démarrage de la session Spark est mesuré à part (coût fixe,
payé une seule fois par exécution, jamais par corpus).
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from memoire.data.splits import make_group_split


def synthetic_records(n_images: int, seed: int = 0) -> list[dict]:
    """Corpus synthétique de ``n_images`` reproduisant la structure réelle."""
    rng = random.Random(seed)
    records = []
    i = 0
    while i < n_images:
        group_size = rng.choice([1, 1, 2, 2, 3])  # ~2 images/groupe en moyenne
        group_id = f"synthetic/{i:08d}"
        official = rng.random() < 0.10
        hint = rng.choice(["train", "val", "test"]) if official else None
        for _ in range(min(group_size, n_images - i)):
            records.append(
                {
                    "image_id": f"synthetic/img{i:08d}",
                    "source": "synthetic",
                    "group_id": group_id,
                    "split_hint": hint,
                    "width": rng.choice([640, 1000, 1920, 3024]),
                    "height": rng.choice([480, 750, 1080, 4032]),
                    # les deux implémentations comptent len(rec["instances"])
                    "instances": [{}] * rng.choice([0, 1, 1, 2, 2, 3, 4, 6, 11]),
                }
            )
            i += 1
    return records


def time_python(records: list[dict], repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        make_group_split(records, seed=42, keep_official=True)
        best = min(best, time.perf_counter() - start)
    return best


def time_spark(spark, records: list[dict], repeats: int) -> float:
    from memoire.data.spark_pipeline import make_group_split_spark

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        make_group_split_spark(spark, records, seed=42, keep_official=True)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scales",
        default="20000,50000,100000,250000,500000,1000000",
        help="tailles de corpus (images), séparées par des virgules",
    )
    parser.add_argument("--repeats", type=int, default=3, help="mesures par point (min retenu)")
    parser.add_argument("--out", type=Path, default=Path("figures/bench_split.csv"))
    args = parser.parse_args()
    scales = [int(s) for s in args.scales.split(",") if s.strip()]

    from pyspark.sql import SparkSession

    start = time.perf_counter()
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("memoire-bench-split")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    startup_s = time.perf_counter() - start
    print(f"Démarrage session Spark (coût fixe, une fois par exécution) : {startup_s:.1f} s")

    rows = []
    for n in scales:
        records = synthetic_records(n)
        t_py = time_python(records, args.repeats)
        t_spark = time_spark(spark, records, args.repeats)
        rows.append(
            {
                "n_images": n,
                "python_s": round(t_py, 3),
                "spark_s": round(t_spark, 3),
                "spark_startup_s": round(startup_s, 1),
            }
        )
        print(f"n={n:>9,} : python {t_py:8.3f} s | spark {t_spark:8.3f} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK: {args.out}")
    spark.stop()


if __name__ == "__main__":
    main()
