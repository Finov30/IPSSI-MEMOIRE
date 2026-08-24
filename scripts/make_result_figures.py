#!/usr/bin/env python
"""Figures et tableaux de résultats du mémoire (chap. 8) depuis les runs réelles.

Usage:
    uv run python scripts/make_result_figures.py [--runs-root runs] [--out figures]

Trois figures, toutes calculées sur les runs effectivement terminées — jamais sur
des chiffres recopiés :

- fig-8-1-courbe-volume.png   : IoU dommage en fonction du nombre d'images
                                d'entraînement (H1, axe volume, chap. 7.1)
- fig-8-2-courbe-densite.png  : IoU dommage par strate d'instances/image, à
                                volume constant (H2, chap. 7.1)
- fig-8-3-ablations.png       : substituts endogènes au pré-entraînement
                                (H3, chap. 7.3)

Plus les tableaux agrégés correspondants en CSV, prêts à intégrer au mémoire.

La source de vérité est le couple ``config.json`` + ``summary.json`` écrit dans
chaque répertoire de run, jamais le ``volume_curve.csv`` agrégé : celui-ci n'est
écrit qu'une fois la campagne entière terminée (``write_csv`` est appelé après la
boucle de ``run_campaign``), alors que ces figures doivent pouvoir être tracées
pendant la campagne, sur les runs déjà disponibles. Les points manquants sont
signalés plutôt que silencieusement omis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mêmes conventions typographiques et chromatiques que scripts/make_figures.py.
TEXT = "#111111"
MUTED = "#52514e"
GRID = "#d9d8d4"
PRIMARY = "#2a78d6"
ACCENT = "#eb6834"
THIRD = "#1baf7a"

# Ordre catégoriel fixe des strates de densité (chap. 7.1), jamais alphabétique.
DENSITY_ORDER = ["1", "2", "3", "4+"]

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def collect_runs(root: Path) -> pd.DataFrame:
    """Agrège les runs terminées sous ``root`` (récursif, un niveau de campagne).

    Une run n'est retenue que si elle a un ``summary.json`` : c'est le seul
    marqueur de fin réussie du code d'entraînement, et c'est aussi celui sur
    lequel se fonde la reprise de campagne.
    """
    rows: list[dict] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        bucket = config.get("density_bucket")
        rows.append(
            {
                "campagne": run_dir.parent.name,
                "run": run_dir.name,
                "seed": config["seed"],
                "model": config["model"],
                "copy_paste": bool(config["copy_paste"]),
                "n_corpus": len(config["corpus"]),
                "density_bucket": _bucket_label(bucket),
                "num_train_images": summary["num_train_images"],
                "best_val_iou": summary["best_val_iou"],
                "best_iteration": summary["best_iteration"],
                "final_train_loss": summary["final_train_loss"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "iterations": summary["iterations"],
            }
        )
    return pd.DataFrame(rows)


def _bucket_label(bucket: list | None) -> str | None:
    """Étiquette lisible d'un ``density_bucket`` ``[min, max|null]``."""
    if not bucket:
        return None
    lo, hi = bucket
    if hi is None:
        return f"{lo}+"
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def aggregate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Moyenne, écart-type et étendue de l'IoU dommage sur les graines."""
    agg = (
        df.groupby(by)
        .agg(
            n_seeds=("seed", "nunique"),
            iou_moyenne=("best_val_iou", "mean"),
            iou_ecart_type=("best_val_iou", "std"),
            iou_min=("best_val_iou", "min"),
            iou_max=("best_val_iou", "max"),
            heures_gpu=("elapsed_seconds", lambda s: s.sum() / 3600.0),
        )
        .reset_index()
    )
    # Une seule graine : l'écart-type n'est pas défini, 0 serait un mensonge.
    return agg


def fig_volume_curve(df: pd.DataFrame, out: Path) -> Path | None:
    """H1 : IoU dommage en fonction du volume d'entraînement, 3 graines."""
    volume = df[(df["density_bucket"].isna()) & (df["model"] == "unet") & (~df["copy_paste"])]
    if volume.empty:
        return None
    agg = aggregate(volume, "num_train_images").sort_values("num_train_images")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(
        agg["num_train_images"], agg["iou_min"], agg["iou_max"],
        color=PRIMARY, alpha=0.15, linewidth=0,
    )
    ax.plot(
        agg["num_train_images"], agg["iou_moyenne"],
        color=PRIMARY, marker="o", markersize=4.5, linewidth=1.6, zorder=3,
    )
    # Graines individuelles : la dispersion réelle, pas seulement son résumé.
    ax.scatter(
        volume["num_train_images"], volume["best_val_iou"],
        color=PRIMARY, s=9, alpha=0.45, zorder=2,
    )
    ax.set_xscale("log")
    ax.set_xticks(agg["num_train_images"])
    # Rotation : en échelle log, les points hauts (10 000 / 14 621) et voisins
    # (2 000 / 2 800) se chevauchent à l'horizontale.
    ax.set_xticklabels(
        [f"{int(n):,}".replace(",", " ") for n in agg["num_train_images"]],
        rotation=30, ha="right", fontsize=8.5,
    )
    ax.minorticks_off()
    ax.set_xlabel("Images d'entraînement (échelle logarithmique)")
    ax.set_ylabel("IoU dommage (validation)")
    ax.set_title(
        "H1 — Performance en fonction du volume, à budget de calcul constant",
        fontsize=11, pad=10, loc="left",
    )
    n_seeds = int(agg["n_seeds"].min())
    iterations = int(volume["iterations"].mode().iloc[0])
    ax.text(
        0.99, 0.03,
        f"{n_seeds} graine(s) par point · {iterations:,} itérations · bande = étendue min-max".replace(",", " "),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED,
    )
    _despine(ax)
    path = out / "fig-8-1-courbe-volume.png"
    fig.savefig(path)
    plt.close(fig)
    agg.to_csv(out / "tableau-8-1-courbe-volume.csv", index=False)
    return path


def fig_density_curve(df: pd.DataFrame, out: Path) -> Path | None:
    """H2 : IoU dommage par strate de densité, à volume constant."""
    density = df[df["density_bucket"].notna()]
    if density.empty:
        return None
    agg = aggregate(density, "density_bucket")
    agg["_ordre"] = agg["density_bucket"].map({b: i for i, b in enumerate(DENSITY_ORDER)})
    agg = agg.sort_values("_ordre").drop(columns="_ordre")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = range(len(agg))
    ax.bar(x, agg["iou_moyenne"], color=THIRD, width=0.55, zorder=2)
    ax.errorbar(
        x, agg["iou_moyenne"],
        yerr=[agg["iou_moyenne"] - agg["iou_min"], agg["iou_max"] - agg["iou_moyenne"]],
        fmt="none", ecolor=MUTED, elinewidth=1.0, capsize=4, zorder=3,
    )
    for xi, (_, row) in zip(x, agg.iterrows()):
        ax.text(xi, row["iou_moyenne"], f"{row['iou_moyenne']:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["density_bucket"])
    ax.set_xlabel("Instances par image (strate)")
    ax.set_ylabel("IoU dommage (validation)")
    volume = int(density["num_train_images"].mode().iloc[0])
    ax.set_title(
        f"H2 — Performance en fonction de la densité, à volume constant ({volume} images)",
        fontsize=11, pad=10, loc="left",
    )
    n_seeds = int(agg["n_seeds"].min())
    ax.text(
        0.99, 0.03, f"{n_seeds} graine(s) par strate · barres = étendue min-max",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED,
    )
    _despine(ax)
    path = out / "fig-8-2-courbe-densite.png"
    fig.savefig(path)
    plt.close(fig)
    agg.to_csv(out / "tableau-8-2-courbe-densite.csv", index=False)
    return path


def fig_ablations(df: pd.DataFrame, out: Path) -> Path | None:
    """H3 : hiérarchie des substituts endogènes, au point d'ancrage commun."""
    ablations = df[df["density_bucket"].isna()].copy()
    ablations["condition"] = ablations.apply(
        lambda r: "U-Net sans skips" if r["model"] == "baseline"
        else ("Copy-Paste" if r["copy_paste"] else "Référence (U-Net)"),
        axis=1,
    )
    variants = ablations[ablations["condition"] != "Référence (U-Net)"]
    if variants.empty:
        return None

    # La référence n'est comparable qu'au même volume que les ablations (chap. 7.6).
    anchors = sorted(variants["num_train_images"].unique())
    subset = ablations[ablations["num_train_images"].isin(anchors)]
    agg = aggregate(subset, "condition")
    order = ["Référence (U-Net)", "U-Net sans skips", "Copy-Paste"]
    agg["_ordre"] = agg["condition"].map({c: i for i, c in enumerate(order)})
    agg = agg.dropna(subset=["_ordre"]).sort_values("_ordre").drop(columns="_ordre")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = [MUTED, ACCENT, THIRD][: len(agg)]
    x = range(len(agg))
    ax.bar(x, agg["iou_moyenne"], color=colors, width=0.55, zorder=2)
    ax.errorbar(
        x, agg["iou_moyenne"],
        yerr=[agg["iou_moyenne"] - agg["iou_min"], agg["iou_max"] - agg["iou_moyenne"]],
        fmt="none", ecolor=TEXT, elinewidth=1.0, capsize=4, zorder=3,
    )
    for xi, (_, row) in zip(x, agg.iterrows()):
        ax.text(xi, row["iou_moyenne"], f"{row['iou_moyenne']:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg["condition"])
    ax.set_ylabel("IoU dommage (validation)")
    anchor_label = " et ".join(f"{int(a)}" for a in anchors)
    ax.set_title(
        f"H3 — Substituts endogènes au pré-entraînement ({anchor_label} images)",
        fontsize=11, pad=10, loc="left",
    )
    _despine(ax)
    path = out / "fig-8-3-ablations.png"
    fig.savefig(path)
    plt.close(fig)
    agg.to_csv(out / "tableau-8-3-ablations.csv", index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = collect_runs(args.runs_root)
    if df.empty:
        raise SystemExit(
            f"aucune run terminée sous {args.runs_root} (summary.json absent partout) — "
            "la campagne n'a pas encore produit de résultat"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    total_hours = df["elapsed_seconds"].sum() / 3600.0
    print(f"{len(df)} run(s) terminée(s), {total_hours:.1f} h GPU cumulées")
    for campagne, group in df.groupby("campagne"):
        print(f"  - {campagne}: {len(group)} run(s), {group['seed'].nunique()} graine(s)")

    for label, produced in (
        ("H1 volume", fig_volume_curve(df, args.out)),
        ("H2 densité", fig_density_curve(df, args.out)),
        ("H3 ablations", fig_ablations(df, args.out)),
    ):
        print(f"{label:14s} : {produced if produced else 'aucune run disponible, figure ignorée'}")

    df.sort_values(["campagne", "num_train_images", "seed"]).to_csv(
        args.out / "runs-detail.csv", index=False
    )
    print(f"détail par run : {args.out / 'runs-detail.csv'}")


if __name__ == "__main__":
    main()
