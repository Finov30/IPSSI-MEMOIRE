#!/usr/bin/env python
"""Figures du mémoire (chap. 4/7.1) depuis les données réelles de data/processed/.

Usage:
    uv run python scripts/make_figures.py [--out figures]

Trois figures, toutes calculées sur le corpus effectivement chargé (stats.parquet
et exports COCO produits par scripts/build_corpus.py) — jamais sur des chiffres
recopiés :

- fig-4-1-densite.png    : distribution de la densité d'instances par corpus
                           (la variable centrale de H2, chap. 7.1)
- fig-4-2-resolution.png : distribution des résolutions par corpus (la variable
                           confondante du chap. 4.6, dispersion interne VehiDE)
- fig-4-3-classes.png    : instances par classe canonique, empilées par source
                           (l'arbitrage taxonomique du chap. 4.5 et son déséquilibre)

Ordre catégoriel fixe (jamais recyclé) : VehiDE, CarDD, Humans in the Loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROCESSED = Path("data/processed")

# Palette catégorielle validée (validate_palette.js, all-pairs, mode light) ;
# l'aqua est sous 3:1 sur fond clair -> étiquettes directes systématiques.
SOURCES = ["vehide", "cardd", "hitl"]
LABELS = {"vehide": "VehiDE", "cardd": "CarDD", "hitl": "Humans in the Loop"}
COLORS = {"vehide": "#2a78d6", "cardd": "#eb6834", "hitl": "#1baf7a"}

TEXT = "#111111"
MUTED = "#52514e"
GRID = "#d9d8d4"

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


def fig_density(stats: pd.DataFrame, out: Path) -> None:
    """Petits multiples : part des images par nombre d'instances, un panneau par corpus."""
    clip = 12  # les queues (>12 instances) sont regroupées dans un dernier bac
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.4), sharex=True)
    for ax, source in zip(axes, SOURCES):
        sub = stats.loc[stats["source"] == source, "n_instances"].clip(upper=clip)
        counts = sub.value_counts(normalize=True).sort_index()
        xs = counts.index.to_numpy()
        ax.bar(
            xs,
            counts.to_numpy(),
            width=0.82,
            color=COLORS[source],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        mean = stats.loc[stats["source"] == source, "n_instances"].mean()
        ax.axvline(mean, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        # HitL a sa masse à droite (bac 12+) : étiquette à gauche pour ce panneau.
        right_side = source != "hitl"
        mean_txt = f"{mean:.2f}".replace(".", ",")
        ax.annotate(
            f"{LABELS[source]} — moyenne {mean_txt} inst./image",
            xy=(0.99 if right_side else 0.01, 0.86),
            xycoords="axes fraction",
            ha="right" if right_side else "left",
            fontsize=9.5,
            color=TEXT,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="x", visible=False)
        _despine(ax)
    axes[-1].set_xticks(range(clip + 1))
    axes[-1].set_xticklabels([str(x) for x in range(clip)] + [f"{clip}+"])
    axes[-1].set_xlabel("Instances annotées par image")
    axes[1].set_ylabel("Part des images du corpus")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out / "fig-4-1-densite.png")
    plt.close(fig)


def fig_resolution(stats: pd.DataFrame, out: Path) -> None:
    """Petits multiples : distribution des résolutions (échelle log), un panneau par corpus."""
    import numpy as np

    lo = max(stats["megapixels"].min(), 0.01)
    hi = stats["megapixels"].max()
    bins = np.geomspace(lo * 0.95, hi * 1.05, 40)
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.4), sharex=True)
    for ax, source in zip(axes, SOURCES):
        sub = stats.loc[stats["source"] == source, "megapixels"]
        weights = pd.Series(1.0 / len(sub), index=sub.index)
        ax.hist(
            sub,
            bins=bins,
            weights=weights,
            color=COLORS[source],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        ax.set_xscale("log")
        mean = sub.mean()
        ax.axvline(mean, color=TEXT, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4)
        def fr(x: float) -> str:
            return f"{x:.2f}".replace(".", ",")
        span = f"{fr(sub.min())} à {fr(sub.max())} Mpx"
        ax.annotate(
            f"{LABELS[source]} — moyenne {fr(mean)} Mpx ({span})",
            xy=(0.01, 0.86),
            xycoords="axes fraction",
            ha="left",
            fontsize=9.5,
            color=TEXT,
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="x", visible=False)
        _despine(ax)
    axes[-1].set_xlabel("Résolution de l'image (mégapixels, échelle logarithmique)")
    axes[1].set_ylabel("Part des images du corpus")
    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(out / "fig-4-2-resolution.png")
    plt.close(fig)


def class_counts_by_source() -> pd.DataFrame:
    """Instances par (classe canonique, source), comptées dans les exports COCO."""
    rows = []
    for source in SOURCES:
        coco = json.loads((PROCESSED / f"{source}.json").read_text(encoding="utf-8"))
        names = {cat["id"]: cat["name"] for cat in coco["categories"]}
        for ann in coco["annotations"]:
            rows.append({"source": source, "classe": names[ann["category_id"]]})
    df = pd.DataFrame(rows)
    return df.groupby(["classe", "source"]).size().unstack(fill_value=0)


def fig_classes(out: Path) -> None:
    """Barres horizontales empilées par source, classes triées par effectif total."""
    counts = class_counts_by_source()
    counts = counts.reindex(columns=SOURCES, fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values().index]  # plus rare en bas -> total en haut

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    left = pd.Series(0, index=counts.index, dtype=float)
    for source in SOURCES:
        values = counts[source]
        ax.barh(
            counts.index,
            values,
            left=left,
            height=0.72,
            color=COLORS[source],
            edgecolor="white",
            linewidth=1.2,
            label=LABELS[source],
            zorder=3,
        )
        left = left + values
    for i, (classe, total) in enumerate(counts.sum(axis=1).items()):
        ax.annotate(
            f"{int(total):,}".replace(",", " "),
            xy=(total, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8.5,
            color=MUTED,
        )
    ax.set_xlim(0, counts.sum(axis=1).max() * 1.12)
    ax.set_xlabel("Instances annotées (corpus harmonisé)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(out / "fig-4-3-classes.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    stats = pd.read_parquet(PROCESSED / "stats.parquet")
    fig_density(stats, args.out)
    fig_resolution(stats, args.out)
    fig_classes(args.out)
    for name in ("fig-4-1-densite", "fig-4-2-resolution", "fig-4-3-classes"):
        print(f"OK: {args.out / name}.png")


if __name__ == "__main__":
    main()
