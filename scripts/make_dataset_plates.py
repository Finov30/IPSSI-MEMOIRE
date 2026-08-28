#!/usr/bin/env python
"""Planches d'images annotées, une par corpus (annexe G du mémoire).

Usage:
    uv run python scripts/make_dataset_plates.py [--out figures] [--seed 42] [--n 6]

Le mémoire décrit longuement ses trois corpus — densités, résolutions, conventions
d'annotation — sans jamais montrer à quoi ressemble une image annotée. Ces planches
comblent ce manque : pour chaque source, ``--n`` images tirées à graine fixée, avec
leurs masques de vérité terrain superposés et coloriés par classe canonique.

Le tirage est déterministe (``random.Random(seed)``) : deux exécutions produisent
les mêmes images, ce qui rend la planche citable dans un mémoire. Les images sont
lues depuis ``data-raw/`` et les annotations depuis les exports COCO harmonisés de
``data/processed/``, si bien que ce qui est montré est exactement ce que le modèle
reçoit après harmonisation de taxonomie — et non l'annotation d'origine.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Racines réelles des fichiers image : les exports COCO ne portent qu'un nom de
# fichier, la localisation dépendant de l'arborescence propre à chaque corpus.
RACINES = {
    "vehide": ["data-raw/vehide/image/image"],
    "cardd": ["data-raw/cardd/train", "data-raw/cardd/val", "data-raw/cardd/test"],
    "hitl": ["data-raw/humans-in-the-loop/Car parts dataset/File1/img"],
}
LIBELLES = {"vehide": "VehiDE", "cardd": "CarDD", "hitl": "Humans in the Loop"}

# Même palette que scripts/make_figures.py : lisible en niveaux de gris à l'impression.
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#393b79", "#637939", "#8c6d31",
]


def _trouver(nom: str, racines: list[str]) -> Path | None:
    for r in racines:
        p = Path(r) / nom
        if p.exists():
            return p
    return None


def _polygones(segmentation) -> list[np.ndarray]:
    """Un segment COCO est une liste de polygones aplatis [x1,y1,x2,y2,…]."""
    out = []
    if isinstance(segmentation, list):
        for poly in segmentation:
            if isinstance(poly, list) and len(poly) >= 6:
                out.append(np.asarray(poly, dtype=float).reshape(-1, 2))
    return out


def planche(source: str, out: Path, n: int, seed: int) -> Path | None:
    coco = json.loads(Path(f"data/processed/{source}.json").read_text(encoding="utf-8"))
    classes = {c["id"]: c["name"] for c in coco["categories"]}
    couleur = {cid: PALETTE[i % len(PALETTE)] for i, cid in enumerate(sorted(classes))}

    par_image = defaultdict(list)
    for a in coco["annotations"]:
        par_image[a["image_id"]].append(a)
    infos = {im["id"]: im for im in coco["images"]}

    # On ne tire que parmi les images annotées et effectivement présentes sur disque.
    candidats = [i for i in par_image if i in infos]
    random.Random(seed).shuffle(candidats)
    retenus = []
    for iid in candidats:
        chemin = _trouver(infos[iid]["file_name"], RACINES[source])
        if chemin is not None:
            retenus.append((iid, chemin))
        if len(retenus) == n:
            break
    if not retenus:
        print(f"  {source} : aucune image trouvée sur disque, planche ignorée")
        return None

    cols = 3
    rows = (len(retenus) + cols - 1) // cols
    # 3,6 de hauteur par rangée : les titres portent la résolution et le nombre
    # d'instances, il leur faut leur place sous peine de chevaucher la rangée
    # précédente — un tirage aléatoire mêle des ratios très différents.
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    vues: set[int] = set()

    for ax, (iid, chemin) in zip(axes, retenus):
        with Image.open(chemin) as im:
            ax.imshow(im.convert("RGB"))
        for a in par_image[iid]:
            cid = a["category_id"]
            vues.add(cid)
            for pts in _polygones(a.get("segmentation")):
                ax.add_patch(mpatches.Polygon(
                    pts, closed=True, facecolor=couleur[cid], edgecolor=couleur[cid],
                    alpha=0.38, linewidth=1.4,
                ))
        inf = infos[iid]
        ax.set_title(f"{inf['width']}×{inf['height']} px · {len(par_image[iid])} instance(s)",
                     fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(retenus):]:
        ax.axis("off")

    fig.legend(
        handles=[mpatches.Patch(color=couleur[c], label=classes[c]) for c in sorted(vues)],
        loc="lower center", ncol=min(6, max(1, len(vues))), frameon=False, fontsize=9,
    )
    fig.suptitle(f"{LIBELLES[source]} — {len(retenus)} images annotées, tirage à graine {seed}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.subplots_adjust(hspace=0.28)
    out.mkdir(parents=True, exist_ok=True)
    chemin_out = out / f"planche-corpus-{source}.png"
    fig.savefig(chemin_out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {LIBELLES[source]:<20} {len(retenus)} images -> {chemin_out}")
    return chemin_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()
    for source in ("vehide", "cardd", "hitl"):
        planche(source, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
