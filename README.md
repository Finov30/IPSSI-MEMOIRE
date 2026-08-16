# Apprendre sans mémoire

Chaîne de traitement pour la segmentation de dommages automobiles par un réseau entraîné de zéro.

Mémoire de fin d'études — Master 2 Big Data & Intelligence Artificielle, campus de Nice.

**Problématique :** combien de données annotées faut-il pour qu'un modèle entraîné de zéro
détecte de façon fiable les dommages d'un véhicule de location ?

## Structure

```
src/memoire/data/    Loaders des corpus, harmonisation, splits, export COCO
configs/             Taxonomie unifiée (arbitrage documenté, chap. 4.5)
scripts/             Points d'entrée (construction du corpus, etc.)
tests/               Tests pytest (synthétiques + realdata)
docs/                Conventions du pipeline
data-raw -> ~/memoire-datasets   (symlink, hors git)
data/processed/      Sorties de conversion (hors git, versionnées par DVC — data/processed.dvc)
```

## Corpus

| Corpus | Images | Instances | Format source |
|---|---|---|---|
| VehiDE | 13 945 | 36 081 | Polygones VIA (non standard), 7 classes |
| CarDD | 4 000 (2 816 / 810 / 374) | 8 740 | COCO, 6 classes |
| Humans in the Loop | 814 | 9 084 | Polygones, 8 classes (⚠️ dossiers sources inversés) |

Les chiffres proviennent d'un comptage direct des fichiers d'annotations ; tout écart mesuré
par le pipeline est documenté dans `data/processed/REPORT.md`.

## Mise en route

```bash
uv venv .venv && uv pip install -e '.[dev,train]'
uv run pytest
uv run python scripts/build_corpus.py
```

`train` (torch, mlflow) est nécessaire dès `uv run pytest` — plusieurs tests
(modèle, boucle d'entraînement, courbe de volume) l'importent. `spark` (pyspark,
nécessite un JVM) n'est utile que pour `scripts/spark_build_corpus.py` et
`docker compose run spark ...` ; voir `docker-compose.yml` et `airflow/`.

## Versionnement des données (DVC, chap. 5.6)

`data/processed/` (exports COCO harmonisés, `stats.parquet`, `REPORT.md`) n'est jamais dans git —
seul `data/processed.dvc` (hash + taille) l'est, ce qui rattache chaque commit à une version
précise et vérifiable des données sans committer de binaires.

```bash
uv pip install -e '.[data]'
dvc remote add --local -d local <chemin-de-stockage>   # une fois par machine, jamais dans git
uv run python scripts/build_corpus.py                  # régénère data/processed/
uv run dvc add data/processed && uv run dvc push        # nouvelle version + upload
uv run dvc pull                                         # récupère la version référencée par le commit courant
```

## Garde-fous

- **Split par véhicule/session, jamais par image** — `check_no_leak` lève si un groupe
  apparaît dans deux splits ; bloquant en CI (`tests/test_splits.py`) et exécuté en conditions
  réelles à chaque construction du corpus.
- **Taxonomie explicite** — aucune fusion de classes implicite : chaque décision
  (mapper / conserver / exclure) est dans `configs/taxonomy.yaml` avec sa justification.
- **Comptages de contrôle** — les loaders sont testés contre les volumes vérifiés à la main.
