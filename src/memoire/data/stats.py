"""Statistiques descriptives des records unifiés (chap. 4.6).

``compute_stats`` produit un DataFrame par image ; ``aggregate_by_source``
produit les agrégats par corpus (densité moyenne attendue : VehiDE ~2.59,
CarDD ~2.19, HitL ~11.16). Ces stats alimentent l'échantillonnage stratifié.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

PER_IMAGE_COLUMNS = [
    "image_id",
    "source",
    "width",
    "height",
    "megapixels",
    "n_instances",
    "instances_per_megapixel",
    "classes",
]

CLASS_SEPARATOR = "|"


def compute_stats(records: Sequence[dict]) -> pd.DataFrame:
    """DataFrame par image : source, taille, densité et classes présentes."""
    rows = []
    for rec in records:
        n_instances = len(rec["instances"])
        megapixels = rec["width"] * rec["height"] / 1e6
        classes = sorted({inst["source_class"] for inst in rec["instances"]})
        rows.append(
            {
                "image_id": rec["image_id"],
                "source": rec["source"],
                "width": rec["width"],
                "height": rec["height"],
                "megapixels": megapixels,
                "n_instances": n_instances,
                "instances_per_megapixel": n_instances / megapixels if megapixels else 0.0,
                "classes": CLASS_SEPARATOR.join(classes),
            }
        )
    return pd.DataFrame(rows, columns=PER_IMAGE_COLUMNS)


def aggregate_by_source(per_image: pd.DataFrame) -> pd.DataFrame:
    """Agrégats par corpus à partir du DataFrame de ``compute_stats``.

    ``density`` = instances par image (moyenne), la métrique de contrôle
    du sujet (VehiDE ~2.59, CarDD ~2.19, HitL ~11.16).
    """
    if per_image.empty:
        return pd.DataFrame(
            columns=["n_images", "n_instances", "density", "mean_megapixels", "n_classes"]
        )

    grouped = per_image.groupby("source", sort=True)
    out = pd.DataFrame(
        {
            "n_images": grouped.size(),
            "n_instances": grouped["n_instances"].sum(),
            "density": grouped["n_instances"].mean(),
            "mean_megapixels": grouped["megapixels"].mean(),
        }
    )

    n_classes = {}
    for source, sub in per_image.groupby("source", sort=True):
        classes: set[str] = set()
        for value in sub["classes"]:
            if value:
                classes.update(value.split(CLASS_SEPARATOR))
        n_classes[source] = len(classes)
    out["n_classes"] = pd.Series(n_classes)
    out.index.name = "source"
    return out
