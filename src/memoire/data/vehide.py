"""Chargeur VehiDE : VIA simplifié -> records unifiés (docs/CONVENTIONS.md).

Format source : deux fichiers JSON (`0Train_via_annos.json`, `0Val_via_annos.json`)
mappant ``nom_fichier.jpg -> {"name": ..., "regions": [{"all_x", "all_y", "class"}]}``.
Les dimensions d'image ne sont pas dans le JSON et sont lues depuis les fichiers (PIL).

Note anti-fuite : le ``group_id`` part du préfixe horodaté (DDMMYYYY_HHMMSS,
parfois YYYYMMDD_HHMMSS ; sinon nom de fichier), puis deux consolidations :
les horodatages voisins (écart <= SESSION_GAP_SECONDS) sont fusionnés en une
même session véhicule, et les copies ``Thumbnail<nom>.jpg`` héritent du
``group_id`` de leur jumeau plein format ``<nom>.jpg``. Certains groupes
chevauchent les splits officiels train/val ; l'arbitrage entre ``split_hint``
et le split par ``group_id`` est fait dans ``splits.py``, pas ici.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

SOURCE = "vehide"

ANNOTATION_FILES = {"train": "0Train_via_annos.json", "val": "0Val_via_annos.json"}
IMAGE_DIRS = {"train": ("image", "image"), "val": ("validation", "validation")}

# Motif dominant des noms de fichiers : DDMMYYYY_HHMMSSimage<digits>.jpg
_TIMESTAMP_PREFIX_RE = re.compile(r"^(\d{8}_\d{6})")
_TIMESTAMP_FORMATS = ("%d%m%Y_%H%M%S", "%Y%m%d_%H%M%S")

# Deux horodatages séparés d'au plus ce délai appartiennent à la même session
# (même véhicule photographié sur plusieurs secondes/minutes consécutives).
SESSION_GAP_SECONDS = 60

# Copies redimensionnées présentes dans le corpus sous "Thumbnail<nom>.jpg".
THUMBNAIL_PREFIX = "Thumbnail"


def flatten_polygon(all_x: Sequence[float], all_y: Sequence[float]) -> list[float]:
    """Zippe all_x/all_y en liste plate COCO [x1, y1, x2, y2, ...]."""
    if len(all_x) != len(all_y):
        raise ValueError(f"all_x and all_y lengths differ: {len(all_x)} != {len(all_y)}")
    flat: list[float] = []
    for x, y in zip(all_x, all_y):
        flat.append(float(x))
        flat.append(float(y))
    return flat


def polygon_area(flat: Sequence[float]) -> float:
    """Aire du polygone par la formule du lacet (shoelace)."""
    n = len(flat) // 2
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        j = (i + 1) % n
        acc += flat[2 * i] * flat[2 * j + 1] - flat[2 * j] * flat[2 * i + 1]
    return abs(acc) / 2.0


def polygon_bbox(flat: Sequence[float]) -> list[float]:
    """Bbox [x, y, w, h] englobant le polygone plat."""
    xs = flat[0::2]
    ys = flat[1::2]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def extract_group_id(filename: str) -> str:
    """Préfixe horodaté 'DDMMYYYY_HHMMSS' si présent, sinon nom de fichier entier."""
    match = _TIMESTAMP_PREFIX_RE.match(filename)
    return match.group(1) if match else filename


def parse_timestamp(group_id: str) -> datetime | None:
    """Interprète un group_id horodaté (DDMMYYYY_HHMMSS, sinon YYYYMMDD_HHMMSS)."""
    for fmt in _TIMESTAMP_FORMATS:
        try:
            # Heure locale de l'appareil, comparée uniquement à elle-même :
            # datetime naïf voulu.
            return datetime.strptime(group_id, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    return None


def merge_session_groups(records: Sequence[dict], gap_seconds: int = SESSION_GAP_SECONDS) -> None:
    """Fusionne en une session les group_id horodatés séparés d'au plus gap_seconds.

    Le préfixe à la seconde près est trop fin : un même véhicule est photographié
    sur plusieurs secondes consécutives et se retrouverait éclaté entre splits.
    Modifie les records en place (group_id = premier horodatage de la session).
    """
    stamped: dict[str, datetime] = {}
    for rec in records:
        gid = rec["group_id"]
        if gid not in stamped:
            ts = parse_timestamp(gid)
            if ts is not None:
                stamped[gid] = ts
    mapping: dict[str, str] = {}
    session_gid: str | None = None
    previous: datetime | None = None
    for gid, ts in sorted(stamped.items(), key=lambda item: (item[1], item[0])):
        if previous is None or (ts - previous).total_seconds() > gap_seconds:
            session_gid = gid
        mapping[gid] = session_gid  # type: ignore[assignment]
        previous = ts
    for rec in records:
        rec["group_id"] = mapping.get(rec["group_id"], rec["group_id"])


def link_thumbnail_groups(records: Sequence[dict]) -> None:
    """Aligne le group_id des 'Thumbnail<nom>.jpg' sur celui du jumeau '<nom>.jpg'.

    Le corpus contient des copies redimensionnées préfixées 'Thumbnail' ; sans
    cette liaison la même photo peut tomber dans deux splits (fuite image).
    Modifie les records en place.
    """
    by_name = {Path(rec["file_path"]).name: rec for rec in records}
    linked = 0
    for rec in records:
        name = Path(rec["file_path"]).name
        if not name.startswith(THUMBNAIL_PREFIX):
            continue
        twin = by_name.get(name[len(THUMBNAIL_PREFIX):])
        if twin is not None and twin is not rec:
            rec["group_id"] = twin["group_id"]
            linked += 1
    if linked:
        logger.info(
            "%s: %d thumbnail duplicate(s) linked to their full-size twin's group",
            SOURCE,
            linked,
        )


def parse_regions(
    regions: Sequence[dict], width: int | None = None, height: int | None = None
) -> tuple[list[dict], int]:
    """Convertit les régions VIA en instances ; retourne (instances, nb_écartés).

    Écarte les polygones dégénérés (< 3 points, aire nulle) et, si width/height
    sont fournis, ceux entièrement hors du cadre image (masque rasterisé vide).
    """
    instances: list[dict] = []
    dropped = 0
    for region in regions:
        flat = flatten_polygon(region["all_x"], region["all_y"])
        area = polygon_area(flat)
        if len(flat) < 6 or area <= 0.0:
            dropped += 1
            continue
        if width is not None and height is not None:
            xs = flat[0::2]
            ys = flat[1::2]
            if min(xs) >= width or min(ys) >= height or max(xs) <= 0.0 or max(ys) <= 0.0:
                dropped += 1
                continue
        instances.append(
            {
                "source_class": region["class"],
                "polygon": [flat],
                "bbox": polygon_bbox(flat),
                "area": area,
            }
        )
    return instances, dropped


def read_image_size(path: Path) -> tuple[int, int]:
    """Lit (width, height) depuis l'en-tête du fichier, sans décoder les pixels."""
    with Image.open(path) as image:
        return image.size


def load_split(root: Path, split: str) -> tuple[list[dict], int]:
    """Charge un split officiel ; retourne (records, nb_polygones_dégénérés)."""
    annotation_path = root / ANNOTATION_FILES[split]
    image_dir = root.joinpath(*IMAGE_DIRS[split])
    with open(annotation_path, encoding="utf-8") as handle:
        annotations = json.load(handle)

    records: list[dict] = []
    dropped_total = 0
    for filename, entry in annotations.items():
        image_path = image_dir / filename
        width, height = read_image_size(image_path)
        instances, dropped = parse_regions(entry["regions"], width, height)
        if dropped:
            logger.debug(
                "%s: %d degenerate or out-of-frame polygon(s) dropped in %s",
                SOURCE,
                dropped,
                filename,
            )
        dropped_total += dropped
        records.append(
            {
                "image_id": f"{SOURCE}/{Path(filename).stem}",
                "file_path": str(image_path.resolve()),
                "width": width,
                "height": height,
                "source": SOURCE,
                "split_hint": split,
                "group_id": extract_group_id(filename),
                "instances": instances,
            }
        )
    return records, dropped_total


def load_records(root: Path) -> list[dict]:
    """Charge VehiDE complet (train + val) en records unifiés.

    Les polygones dégénérés (< 3 points, aire nulle) ou entièrement hors du
    cadre image sont ignorés et comptés dans un warning — jamais silencieusement
    (cf. chiffres de contrôle). Les ``group_id`` sont ensuite consolidés :
    sessions horodatées voisines fusionnées, thumbnails liés à leur jumeau.
    """
    root = Path(root)
    records: list[dict] = []
    dropped_total = 0
    for split in ("train", "val"):
        split_records, dropped = load_split(root, split)
        records.extend(split_records)
        dropped_total += dropped
    if dropped_total:
        logger.warning(
            "%s: dropped %d degenerate or out-of-frame polygon(s); instance total "
            "will differ from the raw control figure by that amount",
            SOURCE,
            dropped_total,
        )
    merge_session_groups(records)
    link_thumbnail_groups(records)
    # Namespaced last, after the timestamp parsing/merging above (which needs
    # the bare "DDMMYYYY_HHMMSS" form) — cardd.py/hitl.py already prefix their
    # group_id with their source name; without it here, the cross-corpus
    # anti-leak check's collision-freedom depended on VehiDE's raw timestamp
    # format never colliding with "cardd/..."/"hitl/..." strings (true today,
    # not an enforced invariant).
    for rec in records:
        rec["group_id"] = f"{SOURCE}/{rec['group_id']}"
    return records
