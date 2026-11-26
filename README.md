# SafeGuard — Real-Time Construction PPE Detection and Monitoring

Academic Deep Learning & End-to-End MLOps Project (Axis 3: Real-Time Object Detection).

SafeGuard is a local decision-support prototype detecting workers and visible Personal Protective Equipment (`helmet`, `vest`, `gloves`) as well as explicit `no_helmet` cues in construction-site images and live webcam streams. It combines a DVC + Prefect + MLflow experimentation & Continuous Training lifecycle with an isolated ONNX Runtime FastAPI backend, anonymous ByteTrack compliance engine, Prometheus/Grafana/Evidently observability, and React + TypeScript operations console.

> **Decision-Support Boundary**: SafeGuard surfaces potential `no_helmet` events requiring human verification. Missing positive PPE (`helmet` or `vest` undetected) never triggers an automatic violation alert, drift metrics are strictly statistical/proxy indicators without ground-truth labels, and no facial recognition, biometric templates, or persistent worker profiles are stored.

---

## Architecture Overview

```text
  Browser / Webcam ──HTTP/JSON──▶ Frontend (Nginx :3000 / Vite :5173)
                                      │ /api/* reverse proxy
                                      ▼
                                 Backend API (FastAPI + ONNX Runtime :8000) ──▶ /metrics ──▶ Prometheus (:9090) ──▶ Grafana (:3001)
                                      ├── /predict, /predict/visualize, /predict/frame
                                      ├── /predict/compliance (ByteTrack + Rules + Alerts)
                                      └── /compliance/alerts, /compliance/analytics
  Prefect UI (:4200) ◀── src/orchestration/prefect_flows.py ──▶ DVC + MLflow (:5000) ◀── Evidently Proxy Drift (src/monitoring/drift.py)
```

Key architectural invariants:
- **Runtime Isolation**: `backend/Dockerfile` installs only [backend/requirements.txt](backend/requirements.txt) (ONNX Runtime, FastAPI, OpenCV headless, `prometheus-client`) and excludes PyTorch, Ultralytics, DVC, MLflow, Prefect, Evidently, datasets, and `.env` secrets via [.dockerignore](.dockerignore).
- **Separate Dockerfiles**:
  - [frontend/Dockerfile](frontend/Dockerfile) — Multi-stage Node 22 build (`npm ci && npm run build`) served via `nginx:1.27-alpine`.
  - [backend/Dockerfile](backend/Dockerfile) — Multi-stage Python 3.12 non-root runtime loading the exported ONNX artifact once during FastAPI lifespan startup.
  - [mlflow/Dockerfile](mlflow/Dockerfile) — Standalone non-root MLflow server backed by persistent volume `mlflow_data:/mlflow`.
  - [docker/training.Dockerfile](docker/training.Dockerfile) — Isolated containerized training, DVC, MLflow, Evidently, and Prefect execution environment (`docker compose --profile training`).

---

## Prerequisites

For **Docker Compose deployment** (recommended):
- Docker Engine / Docker Desktop with Docker Compose v2 (`docker compose version`)
- `6 GB+` available RAM (CPU inference; GPU is not required)
- Free local ports `3000` (Frontend), `8000` (FastAPI), `5000` (MLflow), `4200` (Prefect), `9090` (Prometheus), and `3001` (Grafana)

For **local bare-metal development / training**:
- Python `3.12`
- Node.js `22+` and `npm`
- Git & DVC (only required for offline dataset/model pipelines)

---

## Model Bootstrap Process (Required for Live Inference)

Large binaries (`.onnx`, `.pt`, raw/processed datasets) are excluded from Git and Docker image layers. Before running inference in Docker or locally:

1. Ensure the verified production ONNX artifact and sidecar metadata exist in `models/production/`:
   - `models/production/safeguard_yolo11s.onnx`
   - `models/production/safeguard_yolo11s.metadata.json`
2. **Option A — Re-export from the validated training checkpoint / DVC stage**:
   ```powershell
   .\.venv-train\Scripts\python.exe -m src.models.export_onnx `
     --output models/production/safeguard_yolo11s.onnx `
     --verification-output reports/benchmark/verification_report.json
   ```
   Or rerun DVC / Prefect:
   ```powershell
   dvc repro export
   # Or run orchestrated Prefect CT flow (dry-run or --execute):
   python -m src.orchestration.prefect_flows --mode ct --force-retrain
   ```
3. **Option B — Unloaded Cold-Start Behavior**:
   If starting from a fresh checkout before exporting the ONNX model, the backend still starts cleanly: `GET /health` returns HTTP `200` with `"model_loaded": false`, while `/predict*` returns HTTP `503` (`model_unavailable`) until `safeguard_yolo11s.onnx` is placed in `models/production/`.

---

## Local Deployment with Docker Compose

1. *(Optional)* Copy the environment template if overriding ports, thresholds, or credentials (never commit `.env`):
   ```powershell
   Copy-Item .env.example .env
   ```

2. Launch all stack services (`frontend`, `backend`, `mlflow`, `prefect`, `prometheus`, `grafana`) on bridge network `safeguard_net`:
   ```powershell
   docker compose up --build
   ```

3. Verify service UIs & endpoints once health checks pass:
   - **Frontend Operations Console**: `http://localhost:3000`
   - **Backend Health / Swagger / Metrics**: `http://localhost:8000/health`, `/docs`, `/metrics`
   - **MLflow Tracking & Registry UI**: `http://localhost:5000`
   - **Prefect Orchestration UI**: `http://localhost:4200`
   - **Prometheus Targets & Alerts**: `http://localhost:9090`
   - **Grafana Monitoring Dashboard**: `http://localhost:3001`

4. *(Optional)* Execute containerized training/orchestration dry-run via the isolated `training` profile:
   ```powershell
   docker compose --profile training run --rm training
   ```

5. Stop services cleanly while preserving persistent volumes (`mlflow_data`, `prefect_data`, `prometheus_data`, `grafana_data`, `monitoring_logs`):
   ```powershell
   docker compose down
   ```

---

## Bare-Metal Local Development, Drift Checks & Verification

### Environment Setup (CPU Lint, Backend & Test Suite)
```powershell
python -m venv .venv-backend
.\.venv-backend\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Dev Server
```powershell
cd frontend
npm ci
npm run dev
```

### Orchestration, Drift Analysis & Quality Gates (CPU-Compatible)
```powershell
# Python formatting, linting, and full pytest suite (45 unit + API tests)
black --check .
flake8
pytest

# Proxy statistical drift audit (Evidently AI) & Prefect CT dry-run
python -m src.monitoring.drift
python -m src.orchestration.prefect_flows --mode dry-run

# Frontend linting, strict TypeScript check, and Vite production build
cd frontend
npm run lint
npm run build
```

---

## CI/CD/CT Workflows & Registry Rollback

- [.github/workflows/ci-tests.yml](.github/workflows/ci-tests.yml) — Runs CPU-only `black --check .`, `flake8`, `pytest`, and frontend `npm ci && npm run lint && npm run build`.
- [.github/workflows/docker-build.yml](.github/workflows/docker-build.yml) — Validates `docker compose config --quiet`, builds `backend`, `frontend`, `mlflow`, and `docker/training.Dockerfile`, runs full multi-container HTTP smoke checks, and publishes SHA-tagged images to `ghcr.io/bril3d/safeguard-*` on `main` pushes.
- [.github/workflows/continuous-training.yml](.github/workflows/continuous-training.yml) — Scheduled / manual (`workflow_dispatch`) proxy drift check (`src/monitoring/drift.py`) and Prefect CT pipeline check.
- Deployment & Rollback Runbook: [docs/ops/deployment_and_rollback.md](docs/ops/deployment_and_rollback.md)

---

## Exploratory Notebooks & Documentation

- Exploratory Notebooks (calling modular `src/` helpers): [notebooks/01_eda_and_dataset_audit.ipynb](notebooks/01_eda_and_dataset_audit.ipynb), [notebooks/02_experiments_and_onnx_benchmark.ipynb](notebooks/02_experiments_and_onnx_benchmark.ipynb)
- Technical Report LaTeX Source: [rapport_technique.tex](rapport_technique.tex)
- Engineering Rules & Normative Specs: [GEMINI.md](GEMINI.md), [docs/spec/requirements-traceability.md](docs/spec/requirements-traceability.md)
- Architecture & Compliance: [docs/spec/software-architecture.md](docs/spec/software-architecture.md), [docs/architecture/tracking.md](docs/architecture/tracking.md), [docs/architecture/compliance_engine.md](docs/architecture/compliance_engine.md)
- Experiments, Dataset Audit & Benchmarks: [docs/experiments/exp-matrix-comparison.md](docs/experiments/exp-matrix-comparison.md), [reports/dataset/dataset_audit.md](reports/dataset/dataset_audit.md), [reports/benchmark/benchmark_comparison.md](reports/benchmark/benchmark_comparison.md)
