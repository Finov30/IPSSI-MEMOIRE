# Service d'inférence en flux (chap. 5.4) : consommateur Kafka + modèle entraîné.
# Torch CPU explicitement : le GPU de la machine d'entraînement n'est pas
# partagé avec le service, et une roue CUDA pèserait ~2,5 Gio pour rien.
FROM python:3.10-slim

WORKDIR /app
# Seuls les métadonnées et les sources sont copiées pour l'installation : le
# reste du dépôt est monté en lecture seule à l'exécution (docker-compose.yml).
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir -e ".[serve]" \
      --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python", "scripts/serve_inference.py", "--help"]
