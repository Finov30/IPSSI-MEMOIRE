"""Mémoire — le DAG comme protocole expérimental (sujet v2, chap. 5.3).

Trois étapes, dépendances explicites : préparation du corpus (harmonisation +
densité + split stratifié, sur Spark — chap. 5.2) -> une tâche d'entraînement
par point de volume x seed (mappage dynamique Airflow, une tâche par run de la
grille) -> agrégation en une courbe volume/performance.

Reprise après échec : chaque tâche d'entraînement vérifie l'existence de son
``summary.json`` (memoire.training.volume_curve.build_run_config) avant de
relancer un entraînement déjà terminé — un DAG relancé après un échec partiel
ne refait pas le travail déjà fait.

Déclenchement réel (webserver/scheduler) ou test direct :
    docker compose run --rm airflow-scheduler \
        airflow dags test memoire_pipeline 2026-01-01 \
        -c '{"overrides": {"iterations": 10, "batch_size": 2, "input_size": 64,
                            "base_channels": 8, "depth": 2, "num_workers": 0}}'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pendulum
import yaml
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

REPO_ROOT = Path("/app")
DATASETS_ROOT = Path("/data/memoire-datasets")
OUT_DIR = REPO_ROOT / "data" / "processed"
CAMPAIGN_ROOT = REPO_ROOT / "runs" / "volume-curve"


def _apply_override(config: dict, dotted_key: str, value: Any) -> None:
    """Set ``value`` at ``dotted_key`` in ``config``, creating nested dicts.

    Mirrors scripts/train.py's apply_override (kept local to avoid importing
    a script-shaped module into the Airflow image).
    """
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


@dag(
    dag_id="memoire_pipeline",
    description="Préparation du corpus (Spark) + campagne courbe de volume (grille d'expériences)",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    params={
        "points": "50,100,250,500,1000,full",
        "seeds": "42,43,44",
        "corpus": "cardd",
        # Overrides appliqués à configs/train.yaml pour chaque run (dotted keys) ;
        # laisser vide pour une vraie campagne, réduire pour une vérification rapide sur CPU.
        "overrides": {},
    },
    tags=["memoire"],
)
def memoire_pipeline():
    @task
    def prepare_corpus() -> str:
        """Harmonisation + densité + split stratifié (Spark), chap. 5.2."""
        from pyspark.sql import SparkSession

        from memoire.data import cardd, hitl, vehide
        from memoire.data.coco_export import export_coco
        from memoire.data.spark_pipeline import make_group_split_spark
        from memoire.data.splits import check_no_leak
        from memoire.data.taxonomy import ExcludedClassError, load_taxonomy

        taxonomy = load_taxonomy(REPO_ROOT / "configs" / "taxonomy.yaml")
        loaders = {
            "vehide": (vehide.load_records, "vehide"),
            "cardd": (cardd.load_records, "cardd"),
            "hitl": (hitl.load_records, "humans-in-the-loop"),
        }
        corpora = {
            source: loader(DATASETS_ROOT / subdir) for source, (loader, subdir) in loaders.items()
        }

        spark = (
            SparkSession.builder.appName("memoire-airflow-prepare").master("local[*]").getOrCreate()
        )
        try:
            splits_by_source = {}
            for source, records in corpora.items():
                splits_by_source[source] = make_group_split_spark(
                    spark, records, seed=42, keep_official=(source == "cardd")
                )
            merged: dict[str, list[dict]] = {}
            for per_source in splits_by_source.values():
                for name, recs in per_source.items():
                    merged.setdefault(name, []).extend(recs)
            check_no_leak(merged)

            split_of_image = {
                rec["image_id"]: name
                for per_source in splits_by_source.values()
                for name, recs in per_source.items()
                for rec in recs
            }
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            for source, records in corpora.items():
                for rec in records:
                    rec["split"] = split_of_image[rec["image_id"]]

                def mapping_fn(source_class: str, _source: str = source) -> str | None:
                    try:
                        return taxonomy.canonical(_source, source_class)
                    except ExcludedClassError:
                        return None

                export_coco(records, OUT_DIR / f"{source}.json", mapping_fn)

            n_images = sum(len(r) for r in corpora.values())
            return f"{n_images} images harmonisées -> {OUT_DIR}"
        finally:
            spark.stop()

    @task
    def build_grid() -> list[dict]:
        from memoire.training.volume_curve import parse_points, parse_seeds

        context = get_current_context()
        params = context["params"]
        points = parse_points(params["points"])
        seeds = parse_seeds(params["seeds"])
        return [{"point": p, "seed": s} for p in points for s in seeds]

    @task
    def train_run(point: int | None, seed: int, corpus_ready: str) -> dict:
        from memoire.training.volume_curve import build_run_config

        context = get_current_context()
        params = context["params"]
        corpus_json = OUT_DIR / f"{params['corpus']}.json"

        base_config = yaml.safe_load((REPO_ROOT / "configs" / "train.yaml").read_text())
        base_config["corpus"] = [str(corpus_json)]
        config, run_dir = build_run_config(base_config, point, seed, CAMPAIGN_ROOT, iterations=None)
        for key, value in (params.get("overrides") or {}).items():
            _apply_override(config, key, value)

        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:
            from memoire.training.train import train

            summary = train(config)

        return {
            "point": point,
            "seed": seed,
            "num_train_images": summary["num_train_images"],
            "best_val_iou": summary["best_val_iou"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "output_dir": summary["output_dir"],
        }

    @task
    def aggregate(results: list[dict]) -> str:
        from memoire.training.volume_curve import write_csv

        csv_path = CAMPAIGN_ROOT / "volume_curve.csv"
        write_csv(results, csv_path)
        return str(csv_path)

    corpus_ready = prepare_corpus()
    grid = build_grid()
    results = train_run.partial(corpus_ready=corpus_ready).expand_kwargs(grid)
    aggregate(results)


memoire_pipeline()
