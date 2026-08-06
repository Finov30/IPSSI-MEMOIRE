"""Export des records unifiés vers un JSON COCO valide (pycocotools).

L'harmonisation de taxonomie reste hors de ce module : elle est injectée via
``mapping_fn`` (classe source -> classe unifiée, ``None`` pour écarter
l'instance), typiquement fournie par ``taxonomy.py``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

MappingFn = Callable[[str], "str | None"]


def records_to_coco(
    records: Sequence[dict],
    mapping_fn: MappingFn | None = None,
) -> dict:
    """Convertit les records unifiés en dictionnaire COCO.

    Ids entiers stables : images triées par ``image_id`` puis numérotées à
    partir de 1 ; catégories triées par nom, numérotées à partir de 1.
    Toutes les annotations sont exportées avec ``iscrowd=0``.
    """
    ordered = sorted(records, key=lambda rec: rec["image_id"])
    seen: set[str] = set()
    for rec in ordered:
        if rec["image_id"] in seen:
            raise ValueError(f"duplicate image_id: {rec['image_id']!r}")
        seen.add(rec["image_id"])

    def resolve(source_class: str) -> str | None:
        return mapping_fn(source_class) if mapping_fn is not None else source_class

    category_names = sorted(
        {
            name
            for rec in ordered
            for inst in rec["instances"]
            if (name := resolve(inst["source_class"])) is not None
        }
    )
    category_ids = {name: i for i, name in enumerate(category_names, start=1)}

    images = []
    annotations = []
    ann_id = 1
    for img_id, rec in enumerate(ordered, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(rec["file_path"]),
                "width": int(rec["width"]),
                "height": int(rec["height"]),
                "source": rec["source"],
                "memoire_image_id": rec["image_id"],
                "group_id": rec["group_id"],
            }
        )
        for inst in rec["instances"]:
            name = resolve(inst["source_class"])
            if name is None:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_ids[name],
                    "segmentation": [[float(v) for v in poly] for poly in inst["polygon"]],
                    "bbox": [float(v) for v in inst["bbox"]],
                    "area": float(inst["area"]),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    return {
        "info": {"description": "memoire M2 - car damage segmentation (unified records)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": "damage"}
            for name in category_names
        ],
    }


def export_coco(
    records: Sequence[dict],
    out_path: str | Path,
    mapping_fn: MappingFn | None = None,
) -> dict:
    """Écrit le JSON COCO sur disque et retourne le dictionnaire exporté."""
    coco = records_to_coco(records, mapping_fn=mapping_fn)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(coco, fh, ensure_ascii=False)
    return coco
