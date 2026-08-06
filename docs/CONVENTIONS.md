# Conventions du pipeline de données

## Schéma de record unifié (interne, avant export COCO)

Chaque module de dataset (`src/memoire/data/{vehide,cardd,hitl}.py`) expose :

```python
def load_records(root: Path) -> list[dict]
```

où chaque record vaut :

```python
{
    "image_id": str,        # identifiant unique stable : "<source>/<nom_fichier_sans_ext>"
    "file_path": str,       # chemin absolu de l'image
    "width": int, "height": int,
    "source": str,          # "vehide" | "cardd" | "hitl"
    "split_hint": str | None,  # split officiel s'il existe ("train"/"val"/"test"), sinon None
    "group_id": str,        # identifiant véhicule/sinistre pour le split anti-fuite
    "instances": [
        {
            "source_class": str,       # classe d'origine, non traduite
            "polygon": [[x1, y1, x2, y2, ...]],  # format segmentation COCO (liste de listes plates)
            "bbox": [x, y, w, h],      # dérivée du polygone
            "area": float,             # aire du polygone (shoelace)
        }
    ],
}
```

Règles :
- Coordonnées en pixels, origine en haut à gauche, jamais normalisées.
- Polygones dégénérés (< 3 points, aire nulle) : ignorés avec compteur loggé, jamais silencieusement.
- Aucune classe n'est traduite/fusionnée dans les loaders — l'harmonisation est le rôle exclusif
  de `taxonomy.py` (config `configs/taxonomy.yaml`), pour que l'arbitrage reste documentable (chap. 4.5).
- `group_id` : CarDD n'a pas d'identifiant véhicule → `group_id = image_id` (mais splits officiels
  conservés via `split_hint`). VehiDE : préfixe horodaté du nom de fichier (photos d'une même session
  = même véhicule). HitL : `image_id`.

## Chiffres de contrôle (comptage direct, juillet 2026 — source : sujet v2)

| Corpus | Images | Instances |
|---|---|---|
| VehiDE | 13 945 (11 621 train / 2 324 val) | 36 081 |
| CarDD | 4 000 (2 816 / 810 / 374) | 8 740 |
| HitL (dommages) | 814 | 9 084 |

Tout écart entre nos loaders et ces chiffres doit être expliqué (polygones dégénérés comptés, etc.),
jamais ignoré.

## Piège HitL

Les dossiers sont inversés : `Car damages dataset/` contient les **pièces**, `Car parts dataset/`
contient les **dommages** (Scratch, Dent, Broken part, Paint chip, Missing part, Flaking, Corrosion,
Cracked). Le loader lit `Car parts dataset/` et le vérifie par les noms de classes, pas par le nom du dossier.

## Splits

- Jamais de split par image : split par `group_id` (véhicule/sinistre).
- `splits.py` fournit `check_no_leak(splits) -> None` qui lève si un `group_id` apparaît dans
  deux splits — ce contrôle sera branché en CI (bloquant).
- Seeds fixées et tracées ; échantillonnage stratifié par densité d'instances.
