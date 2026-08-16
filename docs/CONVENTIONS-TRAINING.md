# Conventions — modèle et entraînement (Phase 2)

Complète `docs/CONVENTIONS.md` (Phase 1 — données). Contrainte absolue du mémoire :
**aucun poids pré-entraîné, aucune donnée externe**. Initialisation aléatoire uniquement.

## Décisions imposées par le sujet (non négociables)

- **U-Net 2D from scratch** (Ronneberger 2015), 16–32 canaux de base configurables.
- **GroupNorm partout, jamais BatchNorm** (batch cible 4–8 sur 1 GPU ; He et al. : la BN
  s'effondre à ce régime). Un test vérifie qu'aucun `nn.BatchNorm*` n'existe dans le modèle.
- **Initialisation gaussienne He** (√(2/N), `kaiming_normal_`) sur les convolutions.
- **Loss Dice + Cross-Entropy** (somme pondérée, poids configurables).
- **Segmentation binaire d'abord** (dommage vs fond), mode multiclasse (3 classes —
  fond/larges/fines, chap. 6.4 ; jamais les 13 classes canoniques de
  `configs/taxonomy.yaml`, intraitables sous la contrainte from-scratch stricte)
  prévu par la même API. Le regroupement large/fine par classe canonique est
  documenté et arbitré dans `configs/taxonomy.yaml` (champ `size`), jamais
  dans le code.
- **Schedule long** : le nombre d'itérations est un paramètre de config, jamais codé en dur.
- **Résolution homogène** : redimensionnement à `input_size` configurable (défaut 512×512,
  letterbox avec padding pour préserver le ratio) — neutralisation de la variable confondante
  résolution (chap. 4.6). Le padding du letterbox est étiqueté fond et **entre dans la loss**
  (CE et Dice) : ~25-33 % de pixels trivialement apprenables par image CarDD — assumé et
  documenté ; l'IoU dommage rapportée n'en est pas affectée (le padding n'y contribue qu'en
  cas de faux positif).

## Interfaces

### Dataset (`src/memoire/training/dataset.py`)

```python
class DamageSegDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json: Path, images_root: Path | None, split: str,
                 mode: str = "binary",       # "binary" | "multiclass"
                 input_size: int = 512,
                 augment: bool = False,      # flips horizontaux + copy-paste (ci-dessous)
                 taxonomy: Taxonomy | None = None,  # requis si mode="multiclass"
                 copy_paste: bool = False,   # Ghiasi et al. 2021 (chap. 7.3), ablation H3
                 copy_paste_prob: float = 0.5)
    def __getitem__(self, i) -> tuple[Tensor, Tensor]
        # image : float32 C×H×W dans [0,1] ; masque : int64 H×W
        # binaire : {0 fond, 1 dommage} ; multiclasse : {0 fond, 1 large, 2 fine}
```

Le regroupement multiclasse est global (via `Taxonomy.size()`, sur le nom canonique
de la catégorie), jamais dérivé de la liste `categories` locale à un fichier COCO —
sinon combiner plusieurs corpus désaligne silencieusement les valeurs de masque
d'un fichier à l'autre (bug réel trouvé et corrigé : voir l'historique git).

**Copy-paste** (`copy_paste=True`, train split uniquement, désactivé par défaut) : avec
probabilité `copy_paste_prob`, colle sur l'image cible (dans l'espace déjà letterboxé) une
instance d'une autre image du même split, redimensionnée aléatoirement (jitter d'échelle
0.5×-1.5×, Ghiasi et al. 2021). Substitut endogène au corpus cible — aucune donnée externe —
pour l'ablation H3 (chap. 7.3). Utilise `self.generator`, donc bénéficie du même correctif de
décorrélation par worker que le flip.

- Source : les JSON COCO de `data/processed/` (champ `split` présent dans chaque image,
  champ `memoire_image_id` = identité complète, `file_name` = basename).
- Rasterisation des masques via `pycocotools` (annToMask), union des instances.
- Le chemin réel de l'image vient de l'extra `memoire_file_path` si présent, sinon
  `images_root / file_name`.

### Modèle (`src/memoire/model/unet.py`)

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=32,
                 depth=4, gn_groups=8)
    # forward : B×C×H×W -> B×num_classes×H×W (logits, même H×W)
```

### Losses / métriques (`src/memoire/training/losses.py`, `metrics.py`)

```python
class DiceCELoss(nn.Module):  # ce_weight, dice_weight ; logits B×K×H×W, target B×H×W
def confusion_update(...)     # accumulation streaming
def iou_per_class(...) -> dict[int, float] ; dice_per_class(...) -> dict[int, float]
```

### Boucle (`src/memoire/training/train.py` + `configs/train.yaml`)

- Config YAML unique : seed, corpus (liste de JSON processed), mode, input_size, base_channels,
  batch_size, lr, iterations, val_every, device, subset_n_images (pour les points de la courbe
  de volume), output_dir.
- Seeds fixées partout (torch, numpy, random, DataLoader worker_init + generator).
- MLflow **optionnel** : si `mlflow` importable et `tracking_uri` configuré, log params +
  métriques + artefacts ; sinon fallback CSV/JSON dans `output_dir`. Jamais bloquant.
- Checkpoint (state_dict + config + itération) : `last.pt` et `best.pt` (meilleure IoU val).
- CLI : `uv run python scripts/train.py --config configs/train.yaml [--override key=val]`.

## CI (GitHub Actions, `.github/workflows/ci.yml`)

- Jobs : ruff + pytest (Python 3.10, `uv sync`). Les tests `realdata` sont automatiquement
  skippés en CI (datasets absents) — les tests synthétiques du garde-fou anti-fuite
  (`check_no_leak`) sont, eux, bloquants.
- Torch CPU en CI : installer via l'index `https://download.pytorch.org/whl/cpu`.

## Chemin Colab

`docs/COLAB.md` : cellules prêtes à coller — clone du repo, téléchargement Kaggle des corpus,
`build_corpus.py`, `train.py` sur GPU T4/L4. Le code ne doit rien supposer de la machine :
tout passe par la config.

## Contrôles du smoke-run (préalable à tout run Colab)

Sur un sous-ensemble CarDD (~50 images train / 20 val, binaire, 512², CPU) :
la loss d'entraînement doit décroître nettement et l'IoU dommage en val doit dépasser
celle d'une prédiction aléatoire/vide. C'est un test de plomberie, pas de performance.
