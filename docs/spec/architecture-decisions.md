# Architecture Decisions & Constraints

**Sources**: [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (normative priority) and [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (`SG-SPEC-001` V1.1).

---

## 1. Core Architectural Decisions (Approved Baseline)

### ADR-01: Decoupled Development Lifecycle vs. Runtime Inference
- **Decision**: Strictly separate offline data/training/experimentation (`src/data/`, `src/models/`, `dvc.yaml`, MLflow) from the online inference runtime (`backend/`, `frontend/`).
- **Rationale**: Prevents training dependencies (PyTorch, Ultralytics training pipeline, raw/processed datasets) from bloating or coupling the production serving container (`SG-SPEC-001` p.4–5, p.17; Prof. Guide p.6–7).
- **Constraint**: `backend/` **must not** import from `src/data/` or `src/models/train.py`, and `backend/Dockerfile` must not copy datasets or training checkpoints (`best.pt`). It consumes only the exported, parity-verified ONNX artifact and model metadata.

### ADR-02: Single-Axis Scope — Axis 3 Real-Time Object Detection
- **Decision**: Implement Axis 3 (Real-Time Object Detection) using the Ultralytics YOLO family (`YOLO11n`, `YOLO11s`, `YOLOv8n`) for construction-site PPE monitoring (`SG-SPEC-001` p.1, p.11; Prof. Guide p.3–4).
- **Explicit Exclusions for V1** (`SG-SPEC-001` p.3, p.16):
  - No facial recognition, biometric identification, or named worker tracking.
  - No vehicle/machinery detection, hazardous-zone analysis, or person-machine proximity analysis.
  - No single-source classes in the V1 merged baseline unless a separately versioned experiment is approved.
  - No autonomous sanctions or compliance certification; all outputs are decision-support only.
  - No cloud production deployment required (local Docker Compose stack).

### ADR-03: Standardized Repository Layout
- **Decision**: Combine the professor's required directory tree (Prof. Guide p.7) with SafeGuard's data pipeline and extension modules (`SG-SPEC-001` p.17):
  - `.github/workflows/`: `ci-tests.yml`, `docker-build.yml`
  - `data/`: `raw/`, `interim/`, `processed/` (DVC-tracked only, excluded from Git)
  - `src/`: `data/`, `models/`, `tracking/`, `compliance/`, `utils/`
  - `backend/`: `Dockerfile`, `requirements.txt`, `app/` (`main.py`, `schemas.py`, `inference.py`), `tests/` (`test_api.py`)
  - `frontend/`: `Dockerfile`, `package.json`, `src/` (`components/`, `services/`, `App.tsx`)
  - `configs/`: `data.yaml`, `class_mapping.yaml`, `training.yaml`
  - `models/`: selected checkpoints & ONNX artifacts (DVC-managed)
  - `reports/`: dataset audit, experiment reports, benchmarks
  - `scripts/`: CLI utilities / ingestion / verification scripts
  - Root files: `dvc.yaml`, `params.yaml`, `.dvcignore`, `.gitignore`, `docker-compose.yml`, `README.md`, `rapport_technique.pdf`

### ADR-04: ONNX Runtime Inference Service Pattern
- **Decision**: FastAPI loads the verified ONNX model **once** during application startup/lifespan and reuses the `onnxruntime.InferenceSession` across all prediction requests (`SG-SPEC-001` p.13–14; Prof. Guide p.6).
- **Constraint**: Server latency (`preprocessing`, `inference`, `postprocessing`) must be timed separately from client end-to-end latency. If the ONNX session is uninitialized or unavailable, prediction endpoints must return `503 Service Unavailable`.

### ADR-05: Separation of Detection, Geometric Association, and Compliance Alerts
- **Decision**: Keep three distinct data layers (`SG-SPEC-001` p.4, p.13):
  1. **Raw detections** (`class_id`, `class_name`, `confidence`, `bbox_xyxy`).
  2. **Person-to-PPE geometric associations** (approximate intersection/containment and relative position; uncertain associations must be recorded as uncertain rather than attributed blindly).
  3. **Potential alert events** (timestamp, category, confidence, optional temporary track ID, evidence frame reference, review status).
- **Strict Alert Rule**: Only explicitly trained missing-PPE classes (`no_helmet`, and `no_vest` if approved) may trigger missing-PPE potential alerts. Absence of a positive `helmet` or `vest` detection alone is **never** treated as proof of missing PPE (`SG-SPEC-001` p.13).

### ADR-06: React + TypeScript Frontend & Bounded Real-Time Streaming
- **Decision**: Build the web client in React + TypeScript with a modern CSS framework (`SG-SPEC-001` p.15, p.17; Prof. Guide p.6).
- **Real-Time Constraints**:
  - Support three ingestion modes: Image upload (drag & drop / file picker), preloaded licensed Example Gallery (1-click backend call), and live Webcam/video stream.
  - Bound in-flight webcam inference requests; drop stale frames rather than queuing backlog; preserve aspect-ratio policy when resizing frames; render boxes against corresponding frames (`SG-SPEC-001` p.15).
  - Never use simulated/hardcoded detections in production UI components.

### ADR-07: Local Multi-Service Docker Compose & CPU-Compatible CI
- **Decision**: Orchestrate `frontend`, `backend`, and `mlflow` via `docker compose up --build` with multi-stage Dockerfiles, health checks, persistent volumes, and service networking (`SG-SPEC-001` p.17; Prof. Guide p.6–7).
- **CI Constraint**: GitHub Actions (`ci-tests.yml`, `docker-build.yml`) must pass on CPU runners without requiring GPU access (`SG-SPEC-001` p.17).

### ADR-08: Operational Monitoring vs. Drift Claims & Privacy Defaults
- **Decision**: Track operational metrics (model version, API error counts, latency, P95 latency, FPS, prediction volume, confidence distributions). Any distribution shift indicator without ground-truth labels must be explicitly labeled as a **proxy or statistical warning**, never as verified accuracy degradation (`SG-SPEC-001` p.16).
- **Privacy**: Raw webcam footage must **not** be retained by default. Temporary tracking IDs (if implemented) are anonymous frame-to-frame IDs only (`SG-SPEC-001` p.16).

### ADR-09: Pure-Python Domain Isolation for Compliance, Tracking, and Alerts
- **Decision**: Implement Compliance Rules (`src/compliance/rules.py`), Alert Processing (`src/compliance/alerts.py`), and optional Worker Tracking (`src/tracking/tracker.py`) as pure-Python + NumPy domain modules with **zero** dependencies on PyTorch, Ultralytics, MLflow, DVC, `src/data/`, `src/models/`, or FastAPI.
- **Rationale**: Allows both offline evaluation scripts and the online FastAPI runtime (`backend/`) to share identical compliance/alert/tracking behavior without pulling training frameworks into the inference container or violating clean separation of concerns.

### ADR-10: Separate Training vs. Runtime Dependency Manifests & Docker Contexts
- **Decision**: Maintain separate Python dependency definitions (`requirements-train.txt` for offline training/MLOps vs. `backend/requirements.txt` for the runtime inference service) and exclude training code, datasets, and PyTorch checkpoints from `backend/Dockerfile`.
- **Rationale**: Keeps the runtime backend image lightweight, fast to build in CPU CI, and strictly compliant with `SG-SPEC-001` p.4–5, p.17 and `professor_project_guide.pdf` p.6–7.

### ADR-11: Local-Only Multi-Container Orchestration (No Cloud Deployment)
- **Decision**: Scope deployment strictly to local execution via `docker compose up --build` (`frontend`, `backend`, `mlflow`), with persistent local volumes and internal Docker bridge networking.
- **Rationale**: Matches `SG-SPEC-001` p.3 ("Out of scope for V1: Cloud production deployment") and the professor's local live-demo defense requirement.

---

## 2. Unresolved Architecture & Engineering Decisions (Pending Approval)

The following items are not pinned in either reference PDF and require explicit approval before implementation in the corresponding phases:

1. **ONNX Graph & Optimization Policy**:
   - Exact ONNX `opset` version and input shape policy (static `640x640` batch=1 vs dynamic batch/spatial axes).
   - Whether NMS is embedded inside the exported ONNX graph or executed in Python/NumPy postprocessing inside `backend/app/inference.py`.
   - Exact parity tolerances (box IoU/coordinate delta, confidence delta, class-ID match) between PyTorch and ONNX Runtime.
   - Whether post-training quantization (`FP16` or `INT8`, mentioned in Prof. Guide p.10 rubric) will be evaluated after FP32 ONNX baseline parity.
2. **Frontend Stack & UI Scope**:
   - Specific React build tool (e.g., Vite) and CSS/component framework (TailwindCSS + Shadcn/UI vs Material UI).
   - Exact layout/consolidation of the 8 suggested pages in `SG-SPEC-001` p.15 (`Overview`, `Live Monitoring`, `Image Analysis`, `Example Gallery`, `Alert History`, `Analytics`, `Model Monitoring`, `Settings/Model Information`).
   - Exact delivery timing of `Extension` modules (`src/tracking/tracker.py` ByteTrack-style IDs and `Analytics`) relative to core milestones.
3. **Infrastructure & Storage**:
   - Shared DVC remote storage backend accessible to all 3 team members (e.g., Google Drive, S3/MinIO, or shared storage).
   - MLflow backend store (SQLite vs PostgreSQL container) and artifact volume layout inside `docker-compose.yml`.
   - Persistence strategy (in-memory vs lightweight local store) for operational monitoring metrics and alert review history.
   - Evidence frame retention toggle policy (default off; whether configurable local retention is needed for demo).
4. **Training Environment**:
   - Primary GPU training environment (Kaggle / Google Colab vs local GPU) and exact training hyperparameter/epoch budget for `EXP-A` through `EXP-F`.
