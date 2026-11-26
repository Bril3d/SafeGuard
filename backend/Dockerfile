# Multi-stage runtime Dockerfile for SafeGuard FastAPI service.
# Strictly excludes training dependencies (PyTorch, Ultralytics, DVC, MLflow), credentials, and datasets.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

WORKDIR /app

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    mkdir -p /app/models/production /app/logs && \
    chown -R appuser:appgroup /app

COPY --from=builder /opt/venv /opt/venv
COPY configs/class_mapping.yaml ./configs/class_mapping.yaml
COPY backend/__init__.py ./backend/__init__.py
COPY backend/app ./backend/app
COPY src/__init__.py ./src/__init__.py
COPY src/compliance ./src/compliance
COPY src/tracking ./src/tracking

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
