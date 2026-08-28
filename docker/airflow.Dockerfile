# Airflow image for the memoire pipeline DAG (mémoire, chap. 5.3 — "le DAG
# comme protocole expérimental"). Extends the official image with the memoire
# package itself (CPU torch: the train_run tasks call
# memoire.training.train.train() in-process — real GPU runs are launched
# separately; this proves the DAG's orchestration logic end to end on CPU).
# No JVM: the corpus-prep task runs the pandas path, whose determinism comes
# from a stable hash of the group ids and not from the engine executing it.
FROM apache/airflow:2.11.0-python3.10

USER airflow
COPY --chown=airflow:root pyproject.toml /app/pyproject.toml
COPY --chown=airflow:root src /app/src
WORKDIR /app
RUN pip install --no-cache-dir -e ".[train]" --extra-index-url https://download.pytorch.org/whl/cpu
