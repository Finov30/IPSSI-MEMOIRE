"""Loader Humans in the Loop (export Supervisely) vers les records unifiés.

Piège connu (cf. CONVENTIONS.md) : les dossiers du corpus sont inversés —
``Car parts dataset/`` contient les DOMMAGES, ``Car damages dataset/`` les
pièces. Le loader lit ``Car parts dataset/`` et vérifie à l'exécution, via les
noms de classes de meta.json, qu'il lit bien des dommages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SOURCE = "hitl"

# Nom (trompeur) du dossier qui contient reellement les dommages.
DAMAGE_DIR_NAME = "Car parts dataset"

# Les 8 classes de dommages attendues dans meta.json du dossier lu.
DAMAGE_CLASSES = frozenset(
    {
        "Scratch",
        "Dent",
        "Broken part",
        "Paint chip",
        "Missing part",
        "Flaking",
        "Corrosion",
        "Cracked",
    }
)


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


def bbox_from_ring(ring: list[float]) -> list[float]:
    """Bbox COCO [x, y, w, h] d'un anneau plat."""
    xs, ys = ring[0::2], ring[1::2]
    x_min, y_min = min(xs), min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def _check_damage_classes(dataset_dir: Path) -> None:
    """Vérifie via meta.json qu'on lit bien le dossier des dommages."""
    meta_path = dataset_dir / "meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    titles = {cls["title"] for cls in meta["classes"]}
    if titles != DAMAGE_CLASSES:
        raise ValueError(
            f"HitL: le dossier {dataset_dir.name!r} ne declare pas les 8 classes "
            f"de dommages attendues (dossiers inverses dans ce corpus : verifier "
            f"meta.json, pas le nom du dossier). Classes trouvees : {sorted(titles)}, "
            f"attendues : {sorted(DAMAGE_CLASSES)}"
        )


def load_records(root: Path) -> list[dict]:
    """Charge le corpus de dommages HitL en records unifies.

    ``root`` est le dossier racine de l'export contenant 'Car parts dataset/'
    (dommages, malgre son nom) et 'Car damages dataset/' (pieces, ignore).
    """
    root = Path(root)
    dataset_dir = root / DAMAGE_DIR_NAME
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"HitL: dossier introuvable : {dataset_dir}")
    _check_damage_classes(dataset_dir)

    ann_dir = dataset_dir / "File1" / "ann"
    img_dir = dataset_dir / "File1" / "img"

    records: list[dict] = []
    n_degenerate = 0
    n_holes_dropped = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        image_name = ann_path.name[: -len(".json")]  # 'Car damages 101.png'
        image_path = img_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"HitL: image manquante sur disque : {image_path}")

        with open(ann_path, encoding="utf-8") as f:
            ann = json.load(f)

        instances: list[dict] = []
        for obj in ann["objects"]:
            if obj["geometryType"] != "polygon":
                raise ValueError(
                    f"HitL: geometrie inattendue {obj['geometryType']!r} "
                    f"(objet id={obj['id']}, {ann_path.name})"
                )
            class_title = obj["classTitle"]
            if class_title not in DAMAGE_CLASSES:
                raise ValueError(
                    f"HitL: classe inattendue {class_title!r} "
                    f"(objet id={obj['id']}, {ann_path.name})"
                )

            ring = [float(v) for pt in obj["points"]["exterior"] for v in pt]
            if len(ring) < 6 or shoelace_area(ring) == 0.0:
                n_degenerate += 1
                logger.warning(
                    "HitL: polygone degenere ignore (classe=%s, objet id=%s, %s)",
                    class_title,
                    obj["id"],
                    ann_path.name,
                )
                continue

            interior = obj["points"].get("interior") or []
            if interior:
                # Le format segmentation COCO liste-de-listes plates ne
                # represente pas les trous : on ne garde que le contour
                # exterieur, avec compteur logge.
                n_holes_dropped += len(interior)
                logger.warning(
                    "HitL: %d trou(x) interieur(s) ignore(s) (objet id=%s, %s)",
                    len(interior),
                    obj["id"],
                    ann_path.name,
                )

            instances.append(
                {
                    "source_class": class_title,
                    "polygon": [ring],
                    "bbox": bbox_from_ring(ring),
                    "area": shoelace_area(ring),
                }
            )

        stem = Path(image_name).stem  # 'Car damages 101'
        image_id = f"{SOURCE}/{stem}"
        records.append(
            {
                "image_id": image_id,
                "file_path": str(image_path),
                "width": int(ann["size"]["width"]),
                "height": int(ann["size"]["height"]),
                "source": SOURCE,
                "split_hint": None,  # pas de split officiel dans cet export
                # Aucune info vehicule/sinistre exploitable : group_id = image_id
                # (cf. CONVENTIONS.md).
                "group_id": image_id,
                "instances": instances,
            }
        )

    logger.info(
        "HitL: %d images, %d instances, %d polygones degeneres ignores, "
        "%d trous interieurs ignores",
        len(records),
        sum(len(r["instances"]) for r in records),
        n_degenerate,
        n_holes_dropped,
    )
    return records
