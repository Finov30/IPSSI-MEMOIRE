# Spark stage of the data pipeline (mémoire, chap. 5.2 — harmonisation, calcul
# de densité, échantillonnage stratifié). Runs the JVM Spark needs, so this
# image is the reproducible way to run scripts/spark_build_corpus.py, on any
# host regardless of whether Java is installed there.
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[spark,dev]"

CMD ["python", "scripts/spark_build_corpus.py", "--help"]
