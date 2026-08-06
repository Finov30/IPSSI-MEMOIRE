# Entraînement sur Google Colab

Cellules prêtes à coller, dans l'ordre. Le code ne suppose rien de la machine :
tout passe par `configs/train.yaml` et les `--override` du CLI. Prérequis :
un runtime GPU (T4/L4 — menu *Exécution → Modifier le type d'exécution*) et un
compte Kaggle avec un jeton API (`kaggle.json`, généré depuis
*Kaggle → Settings → API → Create New Token*).

## 1. Vérifier le GPU

```python
!nvidia-smi
```

## 2. Cloner le repo et installer le paquet

Torch GPU est préinstallé sur Colab : on installe seulement le paquet et ses
dépendances données. MLflow est optionnel (fallback JSONL sinon).

```python
!git clone https://github.com/<VOTRE_COMPTE>/IPSSI-MEMOIRE.git
%cd /content/IPSSI-MEMOIRE
!pip install -q -e .
# Optionnel, pour le tracking MLflow :
# !pip install -q mlflow
```

## 3. Identifiants Kaggle

Téléverser votre `kaggle.json` quand la cellule le demande.

```python
from google.colab import files

files.upload()  # sélectionner kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

## 4. Télécharger les trois corpus

Même arborescence que `data-raw/` en local : `vehide/`, `humans-in-the-loop/`,
`cardd/` sous une racine unique (~9 Go au total, compter quelques minutes).

```python
import os

ROOT = "/content/datasets"
os.makedirs(ROOT, exist_ok=True)

def fetch(ref, zip_name, dest):
    assert os.system(f"kaggle datasets download -d {ref} -p {ROOT} --force") == 0, ref
    assert os.system(f"unzip -q -o {ROOT}/{zip_name}.zip -d {ROOT}/{dest}") == 0, ref
    os.remove(f"{ROOT}/{zip_name}.zip")

fetch(
    "hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection",
    "vehide-dataset-automatic-vehicle-damage-detection",
    "vehide",
)
fetch("humansintheloop/car-parts-and-car-damages", "car-parts-and-car-damages",
      "humans-in-the-loop")
fetch("issamjebnouni/cardd", "cardd", "cardd")

!du -sh /content/datasets/*/
```

## 5. Construire le corpus unifié (exports COCO + splits anti-fuite)

```python
%cd /content/IPSSI-MEMOIRE
!python scripts/build_corpus.py --datasets-root /content/datasets --out data/processed
!ls -lh data/processed/
```

## 6. Lancer l'entraînement

La config par défaut est un run de développement (200 itérations). Sur GPU, on
allonge le schedule et on ajuste la sortie ; tout est surchargeable sans
toucher au YAML. `subset_n_images` est le levier des points de la courbe de
volume (sous-échantillonnage déterministe du train, seedé).

```python
!python scripts/train.py --config configs/train.yaml \
  --override device=cuda \
  --override "corpus=[data/processed/cardd.json]" \
  --override iterations=20000 \
  --override warmup_iterations=500 \
  --override val_every=500 \
  --override batch_size=8 \
  --override num_workers=2 \
  --override output_dir=runs/cardd-binary-full
# Point de courbe de volume : ajouter p.ex. --override subset_n_images=500
```

Suivi en cours de run : `runs/<nom>/history.jsonl` (une ligne JSON par
itération d'entraînement et par évaluation).

```python
!tail -n 3 runs/cardd-binary-full/history.jsonl
```

## 7. Rapatrier checkpoints et historique vers Drive

Les checkpoints (`last.pt`, `best.pt`), `history.jsonl`, `config.json` et
`summary.json` sont dans `output_dir`. À copier vers Drive avant la fin de la
session (les VM Colab sont éphémères) — relancer cette cellule périodiquement
pendant les longs runs.

```python
from google.colab import drive

drive.mount("/content/drive")
!mkdir -p "/content/drive/MyDrive/memoire-runs"
!cp -r runs/cardd-binary-full "/content/drive/MyDrive/memoire-runs/"
!ls -lh "/content/drive/MyDrive/memoire-runs/cardd-binary-full"
```

## Notes

- **Aucun poids pré-entraîné, aucune donnée externe** : la contrainte du
  mémoire s'applique aussi sur Colab — seuls les trois corpus Kaggle listés
  ci-dessus entrent dans le pipeline.
- Les valeurs surchargées sont typées automatiquement (`int`, `float`,
  booléens/listes YAML) ; les clés pointées atteignent les sections imbriquées,
  p.ex. `--override mlflow.tracking_uri=file:./mlruns`.
- Smoke-run préalable (CONVENTIONS-TRAINING.md) : avant un long run, vérifier
  sur un sous-ensemble (`--override subset_n_images=50 --override
  iterations=200`) que la loss décroît et que l'IoU dommage en val dépasse une
  prédiction vide.
