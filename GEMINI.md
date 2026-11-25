# SafeGuard — Engineering Rules, Architecture Constraints & Conventions

## 1. Project Identity & Normative References

- **Project**: SafeGuard — Real-Time Construction PPE Detection and Monitoring (Academic Module Project, Axis 3: Real-Time Object Detection).
- **Purpose**: Academic decision-support prototype detecting people and visible PPE / explicit missing-PPE cues in construction-site images and webcam/video streams, built with an end-to-end reproducible MLOps lifecycle and local multi-service runtime.
- **Authoritative Documents**:
  1. [`docs/reference/professor_project_guide.pdf`](docs/reference/professor_project_guide.pdf) — Mandatory academic, architectural, milestone, and defense requirements. **Takes precedence over all other specifications in case of conflict.**
  2. [`docs/reference/safeguard_specification_v1.pdf`](docs/reference/safeguard_specification_v1.pdf) — Audited V1.1 baseline functional & technical specification (`SG-SPEC-001`).

---

## 2. Technology Stack

- **ML / Computer Vision**: Python, PyTorch, Ultralytics YOLO (`YOLO11n`, `YOLO11s`, `YOLOv8n`), ONNX, ONNX Runtime
- **MLOps & Reproducibility**: DVC, MLflow (Tracking & Model Registry)
- **Backend**: FastAPI, Pydantic
- **Frontend**: React, TypeScript (with a modern UI/CSS framework such as TailwindCSS / Shadcn/UI / Material UI)
- **Testing & Quality**: Pytest, Playwright, flake8 / black, ESLint
- **Infrastructure & CI/CD**: Docker, Docker Compose, GitHub Actions

---

## 3. Global Engineering Rules (Mandatory)

1. **Never invent numbers or facts**: Never fabricate dataset counts, class IDs, metrics, model performance, or benchmark results. All figures must come from actual file inspection, verified tool output, or executed runs.
2. **Inspect before deciding**: Inspect actual files and official documentation before making implementation decisions.
3. **Preserve raw data**: Keep raw datasets immutable. Never edit original annotations or replace a pinned archive in place.
4. **Prevent data leakage**: Never train on validation or test images. Fit any learned preprocessing only on training data, apply split and leakage checks before training-only augmentation, and never use test data to select thresholds, hyperparameters, or checkpoints.
5. **Audit duplicates and leakage**: Detect exact duplicates (SHA-256) and near-duplicates (perceptual hashes / image embeddings) across both sources and within splits; manually review likely duplicate pairs before removal; group related scenes/video frames so near-identical frames cannot cross train/test boundaries; if a test image duplicates a training image, retain the test copy and remove the training copy from the eligible training pool.
6. **Document decisions**: All important architectural, taxonomy, split, threshold, and model-selection decisions must be documented with provenance.
7. **Clean architecture**: Follow clean architecture and strict separation of concerns across data, training, tracking/compliance, inference service, frontend, and operations.
8. **Code quality & safety**: Use type hints, meaningful names, and appropriate error handling everywhere.
9. **Test critical behavior**: Write automated tests for important behavior across data pipelines, compliance rules, inference wrappers, API endpoints, and UI flows.
10. **Verify before claiming completion**: Run actual verification commands after edits and report exact observed outputs before claiming any phase or task is complete.
11. **Protect approved contracts**: Do not silently change the approved architecture, canonical class taxonomy, API contracts, locked evaluation splits, or project scope.
12. **Verify library APIs**: Use current official documentation (via Context7 when available) whenever library APIs, CLI flags, or versions are uncertain.
13. **Keep Git clean of binaries**: Keep large datasets, raw/processed images, and model weights/artifacts out of Git. Track them with DVC and exclude them via `.gitignore`.
14. **Zero secrets in repo**: Never commit credentials, API keys, DVC tokens, `.env` files, or private dataset tokens in source code, Git commits, or documentation. Use environment variables and GitHub Actions secrets.
15. **TDD for core logic**: Use Test-Driven Development (red -> green -> refactor) for core data processing, API contracts, and business/compliance logic.
16. **Incremental delivery**: Implement features incrementally phase by phase. Never generate the entire project in one operation.
17. **Approval gate for high-impact changes**: Ask for explicit user approval before deleting data, changing dataset splits, changing the canonical taxonomy, or introducing a new major dependency.
18. **Simplicity over over-engineering**: Prefer readable, maintainable code over unnecessary abstractions.

---

## 4. Mandatory Pre-Implementation Workflow

Before implementing any feature or phase:
1. Read the relevant project documentation (`professor_project_guide.pdf`, `safeguard_specification_v1.pdf`, and any module docs/configs).
2. Inspect existing files in the target area.
3. Produce a short, concrete implementation plan.
4. Explicitly list the files that will be created or modified.
5. Define the acceptance tests for that phase.
6. Implement **only** the requested phase (following TDD for core data, API, and business logic).
7. Run verification tests/linters and report their actual observed results.
8. **Stop and wait for explicit user instruction before starting the next phase.**

---

## 5. Architecture Constraints

### 5.1 Decoupled Lifecycle vs. Runtime
- **Training & Experimentation** (`src/data/`, `src/models/`, `dvc.yaml`, MLflow) are strictly separated from the **Runtime Inference Service** (`backend/`).
- The FastAPI runtime **must not** import training code, training dependencies, or datasets, and training artifacts/datasets **must not** be copied into the runtime backend Docker image.
- Training produces versioned PyTorch checkpoints (`best.pt`) and an exported, parity-verified **ONNX** artifact. The backend loads that ONNX model **once** during FastAPI startup/lifespan via ONNX Runtime and reuses it across requests.

### 5.2 Target Repository Structure
Follow the standardized layout combining the professor's guide and `SG-SPEC-001`:

```text
SafeGuard/
├── .github/workflows/
│   ├── ci-tests.yml            # Linting, unit/API tests, frontend checks (CPU-compatible)
│   └── docker-build.yml        # Docker image builds & Compose config validation
├── data/                       # DVC-managed ONLY (never committed to Git)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   ├── data/                   # inspect, validate, harmonize, deduplicate, split
│   ├── models/                 # train, evaluate, benchmark, export_onnx
│   ├── tracking/               # tracker.py (temporary anonymous ByteTrack-style IDs)
│   ├── compliance/             # rules.py (person-PPE association & potential alert logic)
│   └── utils/                  # visualization, metrics
├── backend/
│   ├── Dockerfile              # Multi-stage runtime image (ONNX Runtime, no training deps)
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py             # FastAPI app, lifespan model load, routes
│   │   ├── schemas.py          # Strict Pydantic request/response models
│   │   └── inference.py        # ONNX Runtime wrapper (preprocess, infer, NMS/postprocess)
│   └── tests/
│       └── test_api.py         # Pytest suite (valid, invalid, empty, 503, boundary cases)
├── frontend/
│   ├── Dockerfile              # Multi-stage production build
│   ├── package.json
│   └── src/
│       ├── components/         # Dropzone, ExampleGallery, Webcam, BoundingBoxCanvas, Charts/Alerts
│       ├── services/           # Typed API client calls to FastAPI
│       └── App.tsx
├── configs/
│   ├── data.yaml
│   ├── class_mapping.yaml      # Single source of truth for canonical taxonomy
│   └── training.yaml
├── models/                     # Selected checkpoints & ONNX artifacts (DVC-managed)
├── reports/                    # Dataset audit, experiment reports, benchmarks
├── scripts/                    # CLI utilities / ingestion / verification scripts
├── dvc.yaml                    # Reproducible DVC pipeline
├── params.yaml
├── .dvcignore
├── .gitignore
├── docker-compose.yml          # Single-command stack: frontend, backend, mlflow
├── README.md
└── rapport_technique.pdf       # Final LaTeX-generated technical report
```

### 5.3 Dataset Sources, Taxonomy & DVC Pipeline Gates
- **Pinned Sources** (publisher-reported figures remain provisional until the archive-level audit passes):
  - **Source A — Roboflow Construction Site Safety**: `roboflow-universe-projects/construction-site-safety/30` (version 30 raw, auto-orient applied, no augmentations, CC BY 4.0).
  - **Source B — Ultralytics Construction-PPE**: November 2026 release (CC BY 4.0; no publisher-provided test split).
- **Canonical Taxonomy Gate**:
  - Provisional 5-class semantic intersection: `person`, `helmet`, `vest`, `gloves`, `no_helmet`.
  - `no_vest` is **candidate-only** until exact Roboflow v30 export YAML and visual annotation semantics are verified side-by-side against Ultralytics Construction-PPE.
  - Single-source classes (`boots`, `goggles`, `no_gloves`, `no_boots`, `no_goggle`, `forklifts`, `machinery`, `vehicles`, `masks`, `none`) are excluded from the initial merged taxonomy.
  - Final approved class list and IDs must be written to `configs/class_mapping.yaml` and never hardcoded across scripts.
- **Split Policy**:
  - Preserve Roboflow v30 test split as a locked source-specific test set (subject to duplicate-leakage checks).
  - Preserve Roboflow v30 and Ultralytics validation splits for model development/selection (not as test sets).
  - Derive Ultralytics internal holdout from its training pool only after duplicate/scene-group checks.
  - Evaluate source-specific test sets separately; report combined scores only as supplementary summaries.
- **Minimum DVC Pipeline Stages**:
  `ingest/prepare -> validate -> harmonize -> deduplicate -> merge/split -> statistics -> train -> evaluate -> export` (`dvc repro` must rerun only affected stages).

### 5.4 Experimentation, MLflow & ONNX Parity
- **Experiment Matrix**: `EXP-A` (Ultralytics baseline, YOLO11n), `EXP-B` (Roboflow baseline, YOLO11n), `EXP-C` (Harmonized merged, YOLO11n), `EXP-D` (Harmonized merged, YOLO11s), `EXP-E` (Merged + controlled tuning, best candidate), `EXP-F` (Harmonized merged, YOLOv8n).
- **Fair Controls**: Lock approved taxonomy, comparable budgets, and locked test sets; tune thresholds/hyperparameters on validation data only.
- **MLflow**: Log parameters (architecture, checkpoint, dataset version, class-map hash, epochs, batch, imgsz, optimizer, lr, augmentation, conf/IoU thresholds, seed, hardware), metrics (precision, recall, mAP@50, mAP@50:95, F1, per-class AP, training duration, latency, FPS), artifacts, and register candidates (`Staging` / `Production`).
- **ONNX Verification**: Record checkpoint hash, opset, and input shape policy; verify input/output names, dtypes, and shapes; run parity tests comparing PyTorch vs. ONNX Runtime class IDs, confidences, and boxes within documented tolerances; benchmark mean/P95 latency, FPS, memory, and artifact size.

### 5.5 Compliance Engine, Tracking & Safety Boundaries
- **Separate Representations**: Keep raw detections, person-to-PPE geometric associations, and derived alerts/events as distinct data structures.
- **Decision-Support Only**: SafeGuard does not certify workplace compliance or issue autonomous sanctions. Alerts must be labelled as **potential events** requiring human verification.
- **Strict Alert Trigger Rule**: Only explicitly trained missing-PPE classes (`no_helmet`, and `no_vest` if approved) may trigger missing-PPE alerts. Absence of a positive `helmet`/`vest` detection alone is **never** treated as proof of missing PPE. Uncertain associations or low confidence must surface uncertainty rather than assert a violation.
- **Privacy & Scope**: No facial recognition, biometric identification, re-identification, or persistent worker profiles. Tracker IDs (ByteTrack or equivalent) are temporary anonymous frame-to-frame IDs only; raw webcam footage must not be retained by default.
- **Monitoring vs. Drift**: Operational monitoring tracks model version, API errors, latency, P95 latency, FPS, prediction volume, and confidence distributions. Any distribution shift indicator without ground-truth labels must be marked as a proxy/statistical warning, not verified accuracy degradation.

### 5.6 Backend API Contract (`backend/`)
- **Mandatory Endpoints**:
  - `GET /health` — Server status, model availability, ONNX Runtime/provider info, GPU availability if detectable, loaded model version.
  - `GET /model/metadata` — Model name, architecture, version, class names, input config, validation metrics, artifact metadata.
  - `POST /predict` — Image `UploadFile` -> structured detections (`class_id`, `class_name`, `confidence`, `bbox_xyxy`), image dimensions, server-side timing.
  - `POST /predict/visualize` — Image `UploadFile` -> annotated image with bounding boxes, labels, and confidence scores.
  - `GET /classes` — Canonical class names and IDs.
  - `POST /predict/frame` — Optional webcam/video frame inference sharing the `/predict` service layer.
- **OpenAPI**: Swagger UI available at `/docs`.
- **Validation & Security**:
  - Strict Pydantic schemas for all payloads.
  - Validate MIME type **and** actual image decoding (never trust filename or `Content-Type` header alone).
  - Enforce upload size and image dimension limits.
  - Semantic HTTP status codes: `400` (invalid/unsupported image), `422` (validation failure), `503` (model not initialized/unavailable), `500` (safe structured error without leaking internal secrets/traces).
  - Measure server preprocessing/inference/postprocessing latency separately from client end-to-end latency. Use structured logs with request IDs.

### 5.7 Frontend & Real-Time Contract (`frontend/`)
- **Ingestion Modes**:
  1. Image upload (drag & drop or file picker, preview, validation, upload state, structured + visual result).
  2. Example gallery (preloaded, licensed sample images calling the real backend inference path in 1 click).
  3. Live webcam/video stream (browser permission, frame capture, controlled request rate, live bounding box overlay).
- **Real-Time Rules**:
  - Bound in-flight requests; never allow an unbounded request queue.
  - Drop stale frames rather than accumulating latency backlog.
  - Preserve aspect-ratio policy when resizing frames to approved inference resolution.
  - Render detections against corresponding frames (or documented synchronization) with distinct per-class colors, labels, confidence indicators, FPS/stream status, potential alerts with review state, and **separate** server vs. client end-to-end latency readouts.
  - Handle camera permission denial, disconnects, API timeouts, and `503` model unavailability gracefully.
  - Production components must call the real FastAPI backend (no simulated detections).

### 5.8 Deployment & CI/CD
- **Docker Compose**: Single command `docker compose up --build` launching `frontend`, `backend`, and `mlflow` with multi-stage Dockerfiles, health checks, service networking, environment variables, and persistent volumes.
- **GitHub Actions**: CPU-compatible workflows running Python linting (`flake8`, `black`), Python unit & API tests (`pytest`), frontend linting & production build, and Docker/Compose build validation.

---

## 6. Coding Conventions

### 6.1 Python (`src/`, `backend/`, `scripts/`)
- **Style & Formatting**: PEP 8 compliant, formatted with `black`, linted with `flake8`.
- **Typing**: Explicit type hints on all function parameters, return types, and class attributes (`from __future__ import annotations` where helpful). Use Pydantic v2 `BaseModel` for API schemas and config validation.
- **Docstrings & Comments**: Concise docstrings on modules, classes, and public functions explaining purpose and contracts; inline comments explain *why*, not *what*.
- **Modularity & Reuse**: Keep functions small and single-purpose with shallow nesting and early returns. Centralize class mappings (`configs/class_mapping.yaml`) and shared utilities rather than duplicating logic or constants.
- **Error Handling**: Validate inputs at system boundaries; raise explicit domain/HTTP exceptions rather than swallowing errors or returning silent fallbacks.
- **Testing (`pytest`)**: Write deterministic, fast, CPU-friendly unit and API tests (mocking ONNX/model binaries only where isolated unit tests require it, plus real ONNX parity and endpoint integration tests). Never rely on test-set data in tests.

### 6.2 TypeScript / React (`frontend/`)
- **Strict Typing**: TypeScript `strict` mode; mirror backend Pydantic schemas in dedicated API types (`frontend/src/types/` or `services/`).
- **Linting & Build**: Pass ESLint and `tsc` / production build cleanly with zero errors.
- **Component Architecture**: Separate API client/transport (`services/`), stream/frame scheduling hooks, canvas/overlay rendering, and UI presentation components. Avoid monolithic components.
- **Testing (`Playwright` / frontend checks)**: Verify critical ingestion flows, error states, and real-time overlay behavior.

### 6.3 Configuration, Commits & Traceability
- Store dataset, class mapping, and training parameters in `configs/*.yaml` and `params.yaml`.
- Keep `.dvc` files, `dvc.yaml`, manifests, and `reports/` in sync with pipeline runs.
- Every reported metric or artifact must trace back to an MLflow run ID, DVC data version, and Git revision.
