# SafeGuard — Real-Time Construction PPE Detection and Monitoring

Academic Deep Learning & MLOps Module Project (Axis 3: Real-Time Object Detection).

SafeGuard is a local decision-support prototype detecting workers and visible Personal Protective Equipment (`helmet`, `vest`, `gloves`) as well as explicit `no_helmet` cues in construction-site images and live webcam streams. It combines a DVC + MLflow experimentation lifecycle with a decoupled ONNX Runtime FastAPI backend, anonymous ByteTrack compliance engine, and React + TypeScript operations console.

> **Decision-Support Boundary**: SafeGuard surfaces potential `no_helmet` events requiring human verification. Missing positive PPE (`helmet` or `vest` undetected) never triggers an automatic violation alert, and no facial recognition, biometric templates, or persistent worker profiles are stored.

---

## Architecture Overview

```text
  Browser / Webcam ──HTTP/JSON──▶ Frontend (Nginx :3000 / Vite :5173)
                                      │ /api/* reverse proxy
                                      ▼
                                 Backend API (FastAPI + ONNX Runtime :8000)
                                      ├── /predict, /predict/visualize, /predict/frame
                                      ├── /predict/compliance (ByteTrack + Rules + Alerts)
                                      └── /compliance/alerts, /compliance/analytics
                                 MLflow Tracking Server (:5000) ◀── Persistent mlflow_data:/mlflow
```

Key architectural invariants:
- **Runtime Isolation**: `backend/Dockerfile` installs only [backend/requirements.txt](backend/requirements.txt) (ONNX Runtime, FastAPI, OpenCV headless) and excludes PyTorch, Ultralytics, DVC, MLflow, datasets, and `.env` secrets via [.dockerignore](.dockerignore).
- **Separate Service Images**:
  - [frontend/Dockerfile](frontend/Dockerfile) — Multi-stage Node 22 build (`npm ci && npm run build`) served via `nginx:1.27-alpine`.
  - [backend/Dockerfile](backend/Dockerfile) — Multi-stage Python 3.12 non-root runtime loading the exported ONNX artifact once during FastAPI lifespan startup.
  - [mlflow/Dockerfile](mlflow/Dockerfile) — Standalone non-root MLflow server backed by SQLite + artifact store on persistent named volume `mlflow_data:/mlflow`.

---

## Prerequisites

For **Docker Compose deployment** (recommended):
- Docker Engine / Docker Desktop with Docker Compose v2 (`docker compose version`)
- `4 GB+` available RAM (CPU inference; GPU is not required)
- Free local ports `3000` (Frontend), `8000` (FastAPI), and `5000` (MLflow)

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
   # Activate training environment (.venv-train)
   .\.venv-train\Scripts\python.exe -m src.models.export_onnx `
     --output models/production/safeguard_yolo11s.onnx `
     --verification-output reports/benchmark/verification_report.json
   ```
   Or rerun the reproducible DVC export stage:
   ```powershell
   dvc repro export
   ```
3. **Option B — Unloaded Cold-Start Behavior**:
   If starting from a completely fresh checkout before running model export, the backend container still starts cleanly: `GET /health` returns HTTP `200` with `"model_loaded": false`, while `/predict*` and `/model/metadata` return HTTP `503` (`model_unavailable`) until the ONNX artifact is placed in `models/production/`.

---

## Local Deployment with Docker Compose

1. *(Optional)* Copy the environment template if overriding ports, thresholds, or CORS origins:
   ```powershell
   Copy-Item .env.example .env
   ```
   Never commit `.env` or any private tokens/credentials.

2. Launch all three services (`frontend`, `backend`, `mlflow`) on bridge network `safeguard_net`:
   ```powershell
   docker compose up --build
   ```

3. Verify service endpoints once health checks pass:
   - **Frontend Operations Console**: `http://localhost:3000`
   - **Backend Health**: `http://localhost:8000/health` (also proxied at `http://localhost:3000/api/health`)
   - **Swagger / OpenAPI Docs**: `http://localhost:8000/docs`
   - **MLflow Tracking UI**: `http://localhost:5000`

4. Stop services cleanly while preserving MLflow history (`mlflow_data` volume):
   ```powershell
   docker compose down
   ```

---

## Bare-Metal Local Development & Verification

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
Open `http://localhost:5173`.

### Quality Gates & Tests (CPU-Compatible)
```powershell
# Python formatting, linting, and full pytest suite (42 unit + API tests)
black --check .
flake8
pytest

# Frontend linting, strict TypeScript check, and Vite production build
cd frontend
npm run lint
npm run build
```

---

## CI/CD Workflows

- [.github/workflows/ci-tests.yml](.github/workflows/ci-tests.yml) — Runs CPU-only `black --check .`, `flake8`, `pytest` (using `requirements-dev.txt`), and frontend `npm ci && npm run lint && npm run build` on pull requests and pushes without requiring a GPU or raw dataset files.
- [.github/workflows/docker-build.yml](.github/workflows/docker-build.yml) — Validates `docker compose config --quiet`, builds `backend`, `frontend`, and `mlflow` images, starts the stack, and runs HTTP health/proxy smoke checks before teardown.

---

## Project Documentation

- Engineering Rules & Workflow: [GEMINI.md](GEMINI.md)
- Requirements Traceability: [docs/spec/requirements-traceability.md](docs/spec/requirements-traceability.md)
- Software Architecture & Decisions: [docs/spec/software-architecture.md](docs/spec/software-architecture.md), [docs/spec/architecture-decisions.md](docs/spec/architecture-decisions.md)
- API Contract: [docs/spec/api-contract.md](docs/spec/api-contract.md)
- Anonymous Tracking & Compliance Engine: [docs/architecture/tracking.md](docs/architecture/tracking.md), [docs/architecture/compliance_engine.md](docs/architecture/compliance_engine.md)
- Dataset Audit & Experiments: [docs/spec/dataset-strategy.md](docs/spec/dataset-strategy.md), [docs/experiments/exp-matrix-comparison.md](docs/experiments/exp-matrix-comparison.md)
