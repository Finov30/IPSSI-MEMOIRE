# Airflow image for the memoire pipeline DAG (mémoire, chap. 5.3 — "le DAG
# comme protocole expérimental"). Extends the official image with Java (the
# corpus-prep task calls the Spark harmonisation stage directly) and the
# memoire package itself (CPU torch: the train_run tasks call
# memoire.training.train.train() in-process — real GPU runs are launched
# separately, e.g. on Colab; this proves the DAG's orchestration logic end to
# end on CPU).
FROM apache/airflow:2.11.0-python3.10

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --chown=airflow:root pyproject.toml /app/pyproject.toml
COPY --chown=airflow:root src /app/src
WORKDIR /app
RUN pip install --no-cache-dir -e ".[spark,train]" --extra-index-url https://download.pytorch.org/whl/cpu
