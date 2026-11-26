# Lightweight standalone MLflow Tracking & Registry Server Dockerfile.
# Uses persistent volume storage at /mlflow (no training datasets or credentials in image layers).

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /mlflow

RUN pip install --no-cache-dir "mlflow>=2.22.0,<4.0.0" && \
    addgroup --system mlflowgroup && \
    adduser --system --ingroup mlflowgroup mlflowuser && \
    mkdir -p /mlflow/artifacts && \
    chown -R mlflowuser:mlflowgroup /mlflow

USER mlflowuser

EXPOSE 5000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=3)"]

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
