"""Loader CarDD (COCO officiel) vers les records unifiés de CONVENTIONS.md."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE = "cardd"
SPLITS = ("train", "val", "test")


def shoelace_area(flat_polygon: list[float]) -> float:
    """Aire d'un anneau polygonal [x1, y1, x2, y2, ...] (formule du lacet)."""
    n = len(flat_polygon) // 2
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = flat_polygon[2 * i], flat_polygon[2 * i + 1]
        x2, y2 = flat_polygon[2 * ((i + 1) % n)], flat_polygon[2 * ((i + 1) % n) + 1]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def bbox_from_rings(rings: list[list[float]]) -> list[float]:
    """Bbox COCO [x, y, w, h] englobant tous les anneaux d'un polygone."""
    xs = [x for ring in rings for x in ring[0::2]]
    ys = [y for ring in rings for y in ring[1::2]]
    x_min, y_min = min(xs), min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def _clean_rings(segmentation: list[list[float]]) -> tuple[list[list[float]], int]:
    """Filtre les anneaux dégénérés (< 3 points ou aire de lacet nulle)."""
    kept: list[list[float]] = []
    dropped = 0
    for ring in segmentation:
        if len(ring) < 6 or shoelace_area(ring) == 0.0:
            dropped += 1
        else:
            kept.append([float(v) for v in ring])
    return kept, dropped


def _load_split(root: Path, split: str) -> list[dict]:
    ann_path = root / f"{split}.json"
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    records: list[dict] = []
    n_degenerate = 0
    n_dropped_instances = 0

    for image in sorted(coco["images"], key=lambda im: im["id"]):
        file_path = root / split / image["file_name"]
        if not file_path.is_file():
            raise FileNotFoundError(f"CarDD: image manquante sur disque : {file_path}")

        stem = Path(image["file_name"]).stem
        image_id = f"{SOURCE}/{stem}"
        instances: list[dict] = []

        for ann in sorted(anns_by_image.get(image["id"], []), key=lambda a: a["id"]):
            segmentation = ann["segmentation"]
            if not isinstance(segmentation, list) or ann.get("iscrowd", 0) != 0:
                # L'inspection du corpus n'a trouvé aucun RLE (100 % polygones,
                # iscrowd=0) : toute occurrence signale un corpus inattendu.
                raise ValueError(
                    f"CarDD: segmentation RLE/iscrowd inattendue "
                    f"(split={split}, annotation id={ann['id']})"
                )
            rings, dropped = _clean_rings(segmentation)
            n_degenerate += dropped
            if not rings:
                n_dropped_instances += 1
                logger.warning(
                    "CarDD: instance ignoree (polygone entierement degenere), "
                    "split=%s annotation id=%s image=%s",
                    split,
                    ann["id"],
                    image["file_name"],
                )
                continue
            instances.append(
                {
                    "source_class": categories[ann["category_id"]],
                    "polygon": rings,
                    "bbox": bbox_from_rings(rings),
                    "area": sum(shoelace_area(ring) for ring in rings),
                }
            )

        records.append(
            {
                "image_id": image_id,
                "file_path": str(file_path),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "source": SOURCE,
                "split_hint": split,
                # Pas d'identifiant vehicule dans les JSON ni image_info.xlsx :
                # group_id = image_id (cf. CONVENTIONS.md), splits officiels
                # conserves via split_hint.
                "group_id": image_id,
                "instances": instances,
            }
        )

    logger.info(
        "CarDD[%s]: %d images, %d instances, %d anneaux degeneres ignores, "
        "%d instances entierement degenerees ignorees",
        split,
        len(records),
        sum(len(r["instances"]) for r in records),
        n_degenerate,
        n_dropped_instances,
    )
    return records


def load_records(root: Path) -> list[dict]:
    """Charge les 3 splits officiels CarDD (train/val/test) en records unifies.

    ``root`` est le dossier CarDD contenant train.json/val.json/test.json et
    les dossiers d'images train/, val/, test/.
    """
    root = Path(root)
    records: list[dict] = []
    for split in SPLITS:
        records.extend(_load_split(root, split))
    logger.info(
        "CarDD: total %d images, %d instances",
        len(records),
        sum(len(r["instances"]) for r in records),
    )
    return records
