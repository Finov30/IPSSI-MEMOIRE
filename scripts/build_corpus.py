#!/usr/bin/env python
"""Construit le corpus unifié : chargement, taxonomie, stats, splits, exports.

Sorties (sous ``data/processed/``, dossier gitignoré) :
- ``{vehide,cardd,hitl}.json`` : exports COCO harmonisés (classes canoniques) ;
- ``stats.parquet``            : stats par image (+ split attribué) ;
- ``REPORT.md``                : comptages, densités, dégénérés, écarts vs
  chiffres de contrôle de docs/CONVENTIONS.md.

Usage :
    uv run python scripts/build_corpus.py [--datasets-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd

from memoire.data import cardd, hitl, vehide
from memoire.data.coco_export import export_coco
from memoire.data.image_check import filter_unreadable
from memoire.data.splits import check_no_leak, make_group_split
from memoire.data.stats import aggregate_by_source, compute_stats
from memoire.data.taxonomy import ExcludedClassError, Taxonomy, load_taxonomy

logger = logging.getLogger("build_corpus")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_ROOT = Path("/home/hdcc5629/memoire-datasets")
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"

SPLIT_SEED = 42

LOADERS = {
    "vehide": (vehide.load_records, "vehide"),
    "cardd": (cardd.load_records, "cardd"),
    "hitl": (hitl.load_records, "humans-in-the-loop"),
}

# Chiffres de contrôle (docs/CONVENTIONS.md, comptage direct juillet 2026).
CONTROL_FIGURES = {
    "vehide": {"images": 13_945, "instances": 36_081},
    "cardd": {"images": 4_000, "instances": 8_740},
    "hitl": {"images": 814, "instances": 9_084},
}

_DEGENERATE_PATTERNS = {
    "vehide": re.compile(r"dropped (\d+) degenerate"),
    "cardd": re.compile(r"(\d+) anneaux degeneres ignores, (\d+) instances entierement"),
    "hitl": re.compile(r"polygone degenere ignore"),
}
_HITL_HOLES_RE = re.compile(r"(\d+) trou\(x\) interieur\(s\)")


class _CaptureHandler(logging.Handler):
    """Capture les messages des loaders pour compter les dégénérés loggés."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def count_degenerates(messages: list[str], source: str) -> dict[str, int]:
    """Extrait des logs les compteurs de dégénérés d'un corpus."""
    out = {"degenerate_polygons": 0, "dropped_instances": 0, "interior_holes_dropped": 0}
    pattern = _DEGENERATE_PATTERNS[source]
    for msg in messages:
        if source == "vehide":
            m = pattern.search(msg)
            if m:
                out["degenerate_polygons"] += int(m.group(1))
                out["dropped_instances"] += int(m.group(1))
        elif source == "cardd":
            m = pattern.search(msg)
            if m:
                out["degenerate_polygons"] += int(m.group(1))
                out["dropped_instances"] += int(m.group(2))
        else:  # hitl
            if pattern.search(msg):
                out["degenerate_polygons"] += 1
                out["dropped_instances"] += 1
            m = _HITL_HOLES_RE.search(msg)
            if m:
                out["interior_holes_dropped"] += int(m.group(1))
    return out


def load_all(datasets_root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Charge les 3 corpus ; retourne (records par source, métriques de chargement)."""
    corpora: dict[str, list[dict]] = {}
    metrics: dict[str, dict] = {}
    for source, (loader, subdir) in LOADERS.items():
        handler = _CaptureHandler()
        module_logger = logging.getLogger(f"memoire.data.{source}")
        previous_level = module_logger.level
        module_logger.addHandler(handler)
        module_logger.setLevel(logging.DEBUG)
        started = time.perf_counter()
        try:
            records = loader(datasets_root / subdir)
            # A file existing with valid metadata doesn't mean it decodes —
            # some real JPEGs are truncated (chap. 4.4). Never train on one
            # silently crashing a run hours in; drop and count it here.
            records, n_corrupted, n_corrupted_instances = filter_unreadable(records, source)
        finally:
            module_logger.removeHandler(handler)
            module_logger.setLevel(previous_level)
        elapsed = time.perf_counter() - started
        corpora[source] = records
        metrics[source] = {
            "load_seconds": elapsed,
            "corrupted_images_dropped": n_corrupted,
            "corrupted_instances_dropped": n_corrupted_instances,
            **count_degenerates(handler.messages, source),
        }
        logger.info(
            "%s: %d images, %d instances chargées en %.1fs",
            source,
            len(records),
            sum(len(r["instances"]) for r in records),
            elapsed,
        )
    return corpora, metrics


def check_taxonomy_coverage(taxonomy: Taxonomy, corpora: dict[str, list[dict]]) -> None:
    """Vérifie que toute classe observée est arbitrée dans la taxonomie."""
    for source, records in corpora.items():
        observed = {inst["source_class"] for rec in records for inst in rec["instances"]}
        taxonomy.validate_coverage(source, observed)


def make_mapping_fn(taxonomy: Taxonomy, source: str):
    """mapping_fn (classe source -> canonique, None si exclue) pour l'export COCO."""

    def mapping_fn(source_class: str) -> str | None:
        try:
            return taxonomy.canonical(source, source_class)
        except ExcludedClassError:
            return None

    return mapping_fn


def build_splits(corpora: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """Split anti-fuite par corpus (par group_id, jamais par image).

    CarDD garde ses splits officiels (keep_official=True). VehiDE est re-splitté
    par group_id : certains groupes chevauchent train/val officiels, l'anti-fuite
    prime (cf. CONVENTIONS.md). HitL n'a pas de split officiel.
    """
    splits_by_source: dict[str, dict[str, list[dict]]] = {}
    for source, records in corpora.items():
        splits_by_source[source] = make_group_split(
            records,
            seed=SPLIT_SEED,
            keep_official=(source == "cardd"),
        )
    # Contrôle global : aucun group_id dans deux splits, toutes sources confondues.
    merged: dict[str, list[dict]] = {}
    for per_source in splits_by_source.values():
        for name, recs in per_source.items():
            merged.setdefault(name, []).extend(recs)
    check_no_leak(merged)
    return splits_by_source


def canonical_instance_counts(
    taxonomy: Taxonomy, corpora: dict[str, list[dict]]
) -> pd.DataFrame:
    """Effectifs d'instances par classe canonique et par source."""
    rows: dict[str, dict[str, int]] = {}
    for source, records in corpora.items():
        fn = make_mapping_fn(taxonomy, source)
        for rec in records:
            for inst in rec["instances"]:
                canonical = fn(inst["source_class"])
                if canonical is None:
                    continue
                rows.setdefault(canonical, {}).setdefault(source, 0)
                rows[canonical][source] += 1
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0).astype(int)
    df = df.reindex(sorted(df.index))
    df["total"] = df.sum(axis=1)
    df.index.name = "canonical_class"
    return df


def compare_to_controls(
    corpora: dict[str, list[dict]], metrics: dict[str, dict]
) -> list[str]:
    """Compare les comptages réels aux chiffres de contrôle ; liste les écarts."""
    discrepancies: list[str] = []
    for source, control in CONTROL_FIGURES.items():
        records = corpora[source]
        n_images = len(records)
        n_instances = sum(len(r["instances"]) for r in records)
        dropped = metrics[source]["dropped_instances"]
        n_corrupted = metrics[source]["corrupted_images_dropped"]
        n_corrupted_instances = metrics[source]["corrupted_instances_dropped"]
        explained_instance_drop = dropped + n_corrupted_instances
        if n_images != control["images"]:
            image_gap = control["images"] - n_images
            if image_gap == n_corrupted:
                discrepancies.append(
                    f"{source}: {n_images} images vs {control['images']} attendues "
                    f"— écart de {image_gap} entièrement expliqué par {n_corrupted} "
                    f"fichier(s) image corrompu(s)/illisible(s) ignoré(s) par le loader"
                )
            else:
                discrepancies.append(
                    f"{source}: {n_images} images vs {control['images']} attendues "
                    f"(écart {image_gap:+d}, dont {n_corrupted} corrompu(s) — reste "
                    f"{image_gap - n_corrupted} non expliqué)"
                )
        if n_instances != control["instances"]:
            gap = control["instances"] - n_instances
            if gap == explained_instance_drop:
                extra = (
                    f" et {n_corrupted_instances} sur des images corrompues ignorées"
                    if n_corrupted_instances
                    else ""
                )
                discrepancies.append(
                    f"{source}: {n_instances} instances vs {control['instances']} attendues "
                    f"— écart de {gap} entièrement expliqué par {dropped} polygone(s) "
                    f"dégénéré(s) (aire nulle ou < 3 points) ignoré(s) par le loader{extra}"
                )
            else:
                discrepancies.append(
                    f"{source}: {n_instances} instances vs {control['instances']} attendues "
                    f"(écart {gap}, dont {dropped} dégénéré(s) et {n_corrupted_instances} sur "
                    f"images corrompues — reste {gap - explained_instance_drop} non expliqué)"
                )
    return discrepancies


def df_to_markdown(df: pd.DataFrame) -> str:
    """Table markdown d'un DataFrame indexé (sans dépendance à tabulate)."""
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(idx)] + [str(v) for v in row] for idx, row in zip(df.index, df.values)]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_report(
    out_dir: Path,
    corpora: dict[str, list[dict]],
    metrics: dict[str, dict],
    per_source_agg: pd.DataFrame,
    canonical_counts: pd.DataFrame,
    splits_by_source: dict[str, dict[str, list[dict]]],
    discrepancies: list[str],
    export_seconds: dict[str, float],
) -> Path:
    """Écrit data/processed/REPORT.md et retourne son chemin."""
    lines: list[str] = []
    lines.append("# Rapport de construction du corpus unifié")
    lines.append("")
    lines.append(f"Généré par `scripts/build_corpus.py` (seed split = {SPLIT_SEED}).")
    lines.append("")

    lines.append("## Comptages vs chiffres de contrôle (docs/CONVENTIONS.md)")
    lines.append("")
    lines.append(
        "| Corpus | Images | Images attendues | Images corrompues ignorées "
        "| Instances | Instances attendues | Polygones dégénérés ignorés |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for source, control in CONTROL_FIGURES.items():
        records = corpora[source]
        lines.append(
            f"| {source} | {len(records)} | {control['images']} "
            f"| {metrics[source]['corrupted_images_dropped']} "
            f"| {sum(len(r['instances']) for r in records)} | {control['instances']} "
            f"| {metrics[source]['dropped_instances']} |"
        )
    lines.append("")

    lines.append("### Écarts et explications")
    lines.append("")
    if discrepancies:
        for item in discrepancies:
            lines.append(f"- {item}")
    else:
        lines.append("- Aucun écart.")
    if metrics["hitl"]["interior_holes_dropped"]:
        lines.append(
            f"- hitl : {metrics['hitl']['interior_holes_dropped']} trou(x) intérieur(s) "
            "de polygones ignoré(s) (le format segmentation COCO liste-de-listes "
            "plates ne représente pas les trous) — n'affecte pas le compte d'instances."
        )
    lines.append("")

    lines.append("## Densités et agrégats par corpus")
    lines.append("")
    lines.append(df_to_markdown(per_source_agg.round(3)))
    lines.append("")
    lines.append(
        "Densités de contrôle attendues (instances/image) : "
        "VehiDE ~2.59, CarDD ~2.19, HitL ~11.16."
    )
    lines.append("")

    lines.append("## Instances par classe canonique (après taxonomie)")
    lines.append("")
    lines.append(df_to_markdown(canonical_counts))
    lines.append("")

    lines.append("## Splits anti-fuite (par group_id, jamais par image)")
    lines.append("")
    lines.append("| Corpus | train | val | test | Politique |")
    lines.append("|---|---|---|---|---|")
    for source, splits in splits_by_source.items():
        policy = (
            "splits officiels conservés (keep_official)"
            if source == "cardd"
            else f"re-split par group_id, seed {SPLIT_SEED}, stratifié par densité"
        )
        lines.append(
            f"| {source} | {len(splits.get('train', []))} | {len(splits.get('val', []))} "
            f"| {len(splits.get('test', []))} | {policy} |"
        )
    lines.append("")
    lines.append("`check_no_leak` validé par corpus et sur l'union des trois corpus.")
    lines.append("")

    lines.append("## Temps d'exécution")
    lines.append("")
    lines.append("| Corpus | Chargement (s) | Export COCO (s) |")
    lines.append("|---|---|---|")
    for source in CONTROL_FIGURES:
        lines.append(
            f"| {source} | {metrics[source]['load_seconds']:.1f} "
            f"| {export_seconds[source]:.1f} |"
        )
    lines.append("")

    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(TAXONOMY_PATH)
    corpora, metrics = load_all(args.datasets_root)
    check_taxonomy_coverage(taxonomy, corpora)

    splits_by_source = build_splits(corpora)
    split_of_image = {
        rec["image_id"]: name
        for per_source in splits_by_source.values()
        for name, recs in per_source.items()
        for rec in recs
    }

    all_records = [rec for records in corpora.values() for rec in records]
    for rec in all_records:
        rec["split"] = split_of_image[rec["image_id"]]
    per_image = compute_stats(all_records)
    per_image["split"] = per_image["image_id"].map(split_of_image)
    per_source_agg = aggregate_by_source(per_image)
    per_image.to_parquet(out_dir / "stats.parquet", index=False)

    export_seconds: dict[str, float] = {}
    for source, records in corpora.items():
        started = time.perf_counter()
        export_coco(records, out_dir / f"{source}.json", make_mapping_fn(taxonomy, source))
        export_seconds[source] = time.perf_counter() - started
        logger.info("%s.json exporté en %.1fs", source, export_seconds[source])

    canonical_counts = canonical_instance_counts(taxonomy, corpora)
    discrepancies = compare_to_controls(corpora, metrics)
    report_path = write_report(
        out_dir,
        corpora,
        metrics,
        per_source_agg,
        canonical_counts,
        splits_by_source,
        discrepancies,
        export_seconds,
    )
    logger.info("Rapport écrit : %s", report_path)


if __name__ == "__main__":
    main()
