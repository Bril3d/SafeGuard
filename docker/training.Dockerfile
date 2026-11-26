# Offline ML Training, Data Harmonization, DVC & Prefect Orchestration Dockerfile.
# Strictly separated from backend/Dockerfile (no training packages ship into online inference runtime).

FROM python:3.12-slim AS training

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-dev.txt ./requirements-dev.txt
COPY backend/requirements.txt ./backend/requirements.txt

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" torch torchvision && \
    pip install --no-cache-dir -r requirements-dev.txt && \
    pip install --no-cache-dir "dvc>=3.50.0,<4.0.0" "ultralytics>=8.3.0,<9.0.0"

COPY configs ./configs
COPY src ./src
COPY dvc.yaml params.yaml ./

CMD ["python", "-m", "src.orchestration.prefect_flows", "--mode", "dry-run"]
