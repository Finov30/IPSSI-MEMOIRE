# Environnement d'exécution de la campagne (annexe E)

Relevé factuel de la machine sur laquelle la campagne du chapitre 7 a été
exécutée, et des mesures qui ont fixé ses paramètres. Tout ce qui figure ici a
été mesuré sur cette machine, jamais estimé ni recopié d'une documentation.

## Matériel

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop, **6 143 MiB** de VRAM, SM 8.6 (GA106) |
| Limite de puissance | 65 W appliquée (défaut constructeur 80 W, maximum 95 W) |
| CPU | 12 threads logiques |
| RAM système | 7,6 Gio (hôte Windows ~16 Go, WSL2 en prend la moitié par défaut) |
| Disque | 897 Go libres |
| OS | WSL2, Linux 6.6.87.2-microsoft-standard-WSL2 |

## Logiciel

| | |
|---|---|
| Python | 3.10.19 (imposé par `.python-version`) |
| PyTorch | **2.6.0+cu124** |
| Driver CUDA | 12.7 (driver Windows 566.14) |

**Point de reproductibilité important.** L'index PyPI par défaut installe un
build **cu130**, incompatible avec un driver CUDA 12.7 : `torch.cuda.is_available()`
renvoie `False` avec l'avertissement « the NVIDIA driver on your system is too
old ». La campagne a donc été exécutée sur un build cu124, installé
explicitement :

```
uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.4,<2.7'
```

## Corpus effectivement chargé

Reconstruit localement par `scripts/build_corpus.py` (les données brutes ne sont
pas versionnées, et aucun remote DVC n'est déclaré). Les compteurs obtenus sont
conformes aux chiffres de contrôle de `docs/CONVENTIONS.md` :

| Source | Images | Instances | Écart aux chiffres de contrôle |
|---|---|---|---|
| VehiDE | 13 929 | 36 047 | 16 JPEG corrompus et 2 polygones dégénérés écartés |
| CarDD | 4 000 | 8 740 | exact |
| HitL | 814 | 9 080 | 4 polygones dégénérés écartés |

Splits par `group_id` (jamais par image), seed 42 :
**train 14 621 · val 2 283 · test 1 839**.

Effectifs des strates de densité du train, qui conditionnent la faisabilité de
l'axe H2 (`density_indices` échoue si une strate contient moins d'images que le
volume demandé) :

| Strate (instances/image) | VehiDE seul | Corpus complet |
|---|---|---|
| 1 | 4 819 | 6 104 |
| 2 | 2 102 | 2 802 |
| 3 | 1 457 | 1 870 |
| 4+ | 2 775 | 3 845 |

Toutes les strates dépassent largement le volume constant de 500 images retenu
pour H2.

## Mesures de dimensionnement

U-Net base 32, profondeur 4, entrée 512×512, mode binaire, fp32.

| Régime | s/itération | Pic VRAM |
|---|---|---|
| batch 2, à froid | 0,214 | 1,74 Go |
| batch 4, à froid | 0,477 | 3,39 Go |
| **batch 4, régime soutenu** | **0,92 – 1,01** | 3,39 Go |
| batch 4, AMP fp16 soutenu | 0,75 | 3,27 Go |
| batch 8 (fp32 comme fp16) | **OOM** | — |

En run réel, dataloader compris, le pic mesuré est de **4 007 MiB sur 6 143**.
Une validation complète sur les 2 283 images de validation coûte **≈ 150 s**, et
ce coût est indépendant du point de volume — la validation n'est jamais
sous-échantillonnée.

### Bridage du GPU

Le débit soutenu est environ deux fois plus faible que le débit à froid. Le
relevé des « clocks event reasons » sous charge montre deux limiteurs qui
s'enchaînent :

```
84 °C, 397 MHz, 63,8 W  → SW Power Cap        (plafond de 65 W atteint)
88 °C, 690 MHz, 56,9 W  → SW Thermal Slowdown (la puissance retombe sous 65 W)
```

Le seuil matériel de ralentissement thermique (102 °C) n'est jamais approché :
le second limiteur est un seuil logiciel du vBIOS, autour de 88 °C. Le profil
d'alimentation Windows « performances élevées » a été essayé et **dégrade** le
débit (1,02 s/it contre 0,92), parce qu'il empêche le GPU de redescendre en
repos entre les phases ; le profil « utilisation normale » a été conservé.

Conséquence à retenir pour toute reproduction : sur ce matériel, le facteur
limitant n'est ni la VRAM ni la taille du modèle (8,64 M paramètres, 33 Mio),
mais l'enveloppe thermique et électrique du châssis portable.

## Paramètres figés de la campagne

Fixés avant le premier run et identiques pour les 39 runs, conformément à la
règle de comparaison du chapitre 7.6 — seules varient `subset_n_images`,
`density_bucket`, `model` et `copy_paste`.

| Paramètre | Valeur | Raison |
|---|---|---|
| `batch_size` | 4 | pic 3,39 Go ; batch 8 est OOM même en fp16 |
| `num_workers` | 2 | 7,6 Gio de RAM ; 12 workers exposent à l'OOM système |
| `iterations` | 30 000 | schedule long exigé par l'entraînement de zéro |
| `warmup_iterations` | 1 000 | |
| `val_every` | 2 000 | 15 validations par run, soit ≈ 0,6 h |
| `input_size` | 512 | valeur du chapitre 6.2, non dégradée |
| `lr` / `weight_decay` | 3e-4 / 1e-2 | |
| précision | fp32 | l'AMP n'apporte que 18 % en régime soutenu et romprait la reproductibilité bit-à-bit |

**Le schedule est fixé en itérations, non en epochs.** À batch 4, 30 000
itérations représentent 240 epochs au point 500 mais 8,2 epochs au point 14 621.
Les points hauts sont donc structurellement moins entraînés que les points bas :
la courbe H1 se lit à **budget de calcul constant**, et non à convergence
appariée. Une courbe à epochs appariées demanderait plus de 1 300 h sur ce
matériel.

## Coût observé

≈ 8,4 h d'entraînement et ≈ 0,6 h de validation par run, soit **≈ 9 h par run**
et **≈ 190 h pour les 21 runs de l'axe volume**.

## Limites de reprise

`summary.json` n'est écrit qu'à la fin d'un run réussi, et la boucle
d'entraînement ne relit jamais `last.pt` : une interruption à l'itération 29 000
perd le run entier. La reprise fonctionne **entre** les runs (relancer la même
commande saute tout run déjà terminé), jamais **à l'intérieur** d'un run. Sur
une campagne de plusieurs jours, la mise en veille du système est donc le
principal risque de perte.
