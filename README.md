# SafeGuard — Real-Time Construction PPE Detection and Monitoring

Academic Deep Learning & MLOps Module Project (Axis 3: Real-Time Object Detection).

SafeGuard is a local decision-support prototype for detecting people and visible Personal Protective Equipment (PPE) / explicit missing-PPE cues in construction-site imagery and webcam/video streams, combining an auditable DVC + MLflow training lifecycle with an isolated FastAPI + ONNX Runtime + React TypeScript runtime stack.

---

## Project Documentation

- Engineering Rules & Workflow: [GEMINI.md](GEMINI.md)
- Requirements Traceability: [docs/spec/requirements-traceability.md](docs/spec/requirements-traceability.md)
- Software Architecture: [docs/spec/software-architecture.md](docs/spec/software-architecture.md)
- Architecture Decisions (ADRs): [docs/spec/architecture-decisions.md](docs/spec/architecture-decisions.md)
- API Contract: [docs/spec/api-contract.md](docs/spec/api-contract.md)
- Dataset Strategy: [docs/spec/dataset-strategy.md](docs/spec/dataset-strategy.md)
- Roadmap & Team Ownership: [docs/implementation/roadmap.md](docs/implementation/roadmap.md), [docs/implementation/team-responsibilities.md](docs/implementation/team-responsibilities.md)
- Windows Environment Audit: [docs/implementation/environment-audit.md](docs/implementation/environment-audit.md)

---

## Repository Structure (Foundation)

```text
SafeGuard/
├── .github/workflows/     # CPU-compatible CI tests & Docker build checks
├── backend/               # Isolated FastAPI runtime service + Pytest API tests
├── frontend/              # React + TypeScript (Vite) web interface
├── src/                   # Offline data/model modules + pure-Python compliance/tracking
├── tests/                 # Package & unit tests
├── configs/               # YAML configuration files (taxonomy, data, training)
├── data/                  # DVC-managed datasets (never committed to Git)
├── models/                # DVC-managed checkpoints & ONNX artifacts
├── reports/               # Dataset audits, experiment & benchmark reports
├── scripts/               # CLI utilities
├── docs/                  # Reference PDFs, specs, and implementation plans
├── docker-compose.yml     # Local stack: frontend, backend, mlflow
├── pyproject.toml         # Python package, Black, and Pytest config
└── .flake8                # Python lint rules
```

---

## Local Quickstart (Foundation Phase)

### 1. Environment Variables
Copy `.env.example` to `.env` if custom ports or settings are needed (never commit `.env`):

```powershell
Copy-Item .env.example .env
```

### 2. Backend (Python 3.12 + FastAPI)
```powershell
python -m venv .venv-backend
.\.venv-backend\Scripts\Activate.ps1
pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- Health endpoint: `http://localhost:8000/health`
- Swagger UI: `http://localhost:8000/docs`

### 3. Frontend (React + TypeScript + Vite)
```powershell
cd frontend
npm install
npm run dev
```

- Frontend dev server: `http://localhost:5173`

### 4. Verification Commands
```powershell
# Python formatting, linting, and tests
.\.venv-backend\Scripts\python.exe -m black --check .
.\.venv-backend\Scripts\python.exe -m flake8
.\.venv-backend\Scripts\python.exe -m pytest

# Frontend lint and production build
cd frontend
npm run lint
npm run build
```

### 5. Docker Compose Stack
Ensure Docker Desktop is running, then launch all local services (`frontend`, `backend`, `mlflow`):

```powershell
docker compose up --build
```
