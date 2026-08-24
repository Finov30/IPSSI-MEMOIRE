# Apprendre sans mémoire

Chaîne de traitement pour la segmentation de dommages automobiles par un réseau entraîné de zéro.

Mémoire de fin d'études — Master 2 Big Data & Intelligence Artificielle, campus de Nice.

**Problématique :** combien de données annotées faut-il pour qu'un modèle entraîné de zéro
détecte de façon fiable les dommages d'un véhicule de location ?

## Structure

```
src/memoire/data/    Loaders des corpus, harmonisation, splits, export COCO
src/memoire/serving/ Chemin d'inférence en flux (Kafka -> masque, chap. 5.4)
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

## Inférence en flux (Kafka, chap. 5.4)

Un service consomme `inspection.photos.v1`, segmente chaque photo avec un checkpoint entraîné
et publie `inspection.masks.v1` ; les messages définitivement invalides partent en
`inspection.photos.dlq.v1` (rejouables tels quels après correction). Les photos transitent
**par référence** (claim-check : URI + sha256), jamais en octets bruts : `max.message.bytes`
reste à 1 Mio, ce qui rend tout écart au principe immédiatement visible. La clé de partition
est `inspection_id` — ordre garanti par état des lieux, sans partition chaude par agence.

```bash
uv pip install -e '.[serve]'                    # kafka-python : pur Python, extra optionnel
docker compose --profile serving up -d          # Kafka (KRaft, sans ZooKeeper) + topics + service
uv run python scripts/serve_inference.py --config configs/streaming.yaml \
    --checkpoint runs/<run>/best.pt --bootstrap-servers localhost:29092
uv run python scripts/publish_photos.py --images <dossier> --inspection-id NCE01/AB-123-CD/...
```

Le profil `serving` est **explicite** : ni `docker compose up`, ni les services airflow, ni
`docker compose run --rm spark ...` ne démarrent Kafka. Le chemin d'entraînement reste
inchangé, et `tests/test_serving_isolation.py` le vérifie mécaniquement (aucun module de
`memoire/{data,model,training}` n'importe `kafka` ni `memoire.serving`). Toute la logique
(décodage, DLQ, commit d'offsets, arrêt propre) est testée **sans broker**, avec des faux en
mémoire ; l'architecture du modèle est relue du checkpoint, jamais d'un YAML.

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
