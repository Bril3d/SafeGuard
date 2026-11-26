# SafeGuard — Final Engineering Acceptance Audit Report

**Audit Date**: 2026-11-26  
**GitHub Remote**: `https://github.com/Bril3d/SafeGuard` (`main`)  
**Normative References Audited Against**:
1. [`docs/reference/professor_project_guide.pdf`](../docs/reference/professor_project_guide.pdf) (authoritative precedence)
2. [`docs/reference/safeguard_specification_v1.pdf`](../docs/reference/safeguard_specification_v1.pdf) (`SG-SPEC-001` V1.1)
3. [`docs/spec/requirements-traceability.md`](../docs/spec/requirements-traceability.md) (`M-01`–`M-14`, `SG-01`–`SG-10`, `EXT-01`–`EXT-07`)

> **Overall Audit Verdict — READY ONCE `rapport_technique.pdf` IS COMPILED**  
> All P0 engineering blockers around remote GitHub hosting, green GitHub Actions CI & Docker container smoke validation, DVC local remote/lock synchronization, team member naming, and LaTeX report authoring (`rapport_technique.tex`) are resolved. Only compiling `rapport_technique.tex` into `rapport_technique.pdf` (deferred to user per instruction due to absent local TeX installation) remains before final defense submission.

---

## 1. Prioritized Remaining Actions

| Priority | Audit Item | Status | Current State & Remaining Action |
|---|---|---|---|
| **P0 (Final Defense Step)** | **Compile LaTeX PDF (`rapport_technique.pdf`)** | **PARTIALLY IMPLEMENTED** | [`rapport_technique.tex`](../rapport_technique.tex) is authored, synced with verified figures/metrics, and committed; user will compile `rapport_technique.pdf` (MiKTeX / TeX Live / Overleaf). |
| **P1 (Optional Cross-Host)** | **Off-Host Cloud DVC Remote** | **PARTIALLY IMPLEMENTED** | Local DVC remote `localstore` (`.dvc/local_remote`, pushed 4 artifacts) and `dvc status` (`Data and pipelines are up to date.`) are clean locally; configure S3/GCS/GDrive only if multi-machine binary pulling is needed. |
| **P2 (Minor Metadata)** | **Governance Metadata Polish** | **PARTIALLY IMPLEMENTED** | `configs/class_mapping.yaml` header retains `status: "provisional_pending_user_approval"` and MLflow Registry uses alias `@production` (`current_stage="None"` under MLflow `3.18.0`). |

---

## 2. Criterion-by-Criterion Audit Matrix

### 2.1 DATA (Normative: `M-04`, `M-05`, `SG-01`, `SG-02`, `SG-03`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Correct source versions** | **PASS** | Roboflow Construction Site Safety `v30` (`717` imgs, `5,570` boxes, `CC BY 4.0`) and Ultralytics Construction-PPE (`1,000` imgs, `14,863` boxes, `CC BY 4.0`, no native test set) pinned in `configs/data.yaml`. Hard Hat Workers v10 evaluated separately and rejected for 5-class merge. | `configs/data.yaml`, `reports/dataset/dataset_audit.md`, `docs/datasets/hard_hat_workers_audit.md` |
| **Valid annotations** | **PASS** | `0` missing labels, `0` orphan labels, `0` corrupt images, `0` invalid normalized coordinates across all `1,717` raw images (`20,433` bounding boxes). Unit tests cover malformed coordinates & corrupt images. | `reports/dataset/dataset_audit.md`, `pytest tests/test_data_pipeline.py` (`6 passed`) |
| **Correct class mapping** | **PARTIALLY IMPLEMENTED** | Canonical 5-class taxonomy (`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`) is implemented identically across harmonization, training YAMLs, ONNX metadata, backend schemas, and compliance taxonomy gate, with `no_vest` held as `pending_approval` and single-source classes excluded. Header in `configs/class_mapping.yaml` still reads `status: "provisional_pending_user_approval"`. | `configs/class_mapping.yaml`, `pytest tests/test_compliance_and_tracking.py` |
| **No known split leakage** | **PASS** | `deduplication_report.json` dropped `77` images (`26` `drop_scene_group_test_leakage`, `13` `drop_test_leakage`, `18` `drop_val_leakage`, `20` `drop_within_split_duplicate`). `final_split_statistics.json` verifies `train_test_sha_overlap: 0`, `train_val_sha_overlap: 0`, `val_test_sha_overlap: 0`, and `train_test_scene_group_overlap: 0`. | `reports/dataset/deduplication_report.json`, `reports/dataset/final_split_statistics.json`, `pytest tests/test_harmonization_pipeline.py` (`1 passed`) |
| **DVC reproducibility** | **PARTIALLY IMPLEMENTED** | `dvc.yaml` (10 stages), `dvc.lock`, `.dvcignore` (including `*.cache`), and `.dvc/config` (`localstore -> local_remote`) are up to date locally (`dvc status` prints `Data and pipelines are up to date.` after `dvc push`). Remaining gap: no external cloud DVC remote (`S3`/`GDrive`) or standalone `*.dvc` files outside `dvc.lock`. | `.venv-train\Scripts\dvc.exe status`, `.venv-train\Scripts\dvc.exe push` (`4 files pushed`), `.dvc/config` |

---

### 2.2 MODEL (Normative: `M-04`, `M-07`, `SG-04`, `SG-05`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **YOLO experiments recorded** | **PASS** | `7` finished MLflow runs (`SMOKE` + `EXP-A` through `EXP-F`) trained on `NVIDIA GeForce RTX 5070 Laptop GPU` at `imgsz=640`, `seed=42`, tracked in `sqlite:///mlflow.db`. | `docs/experiments/exp-matrix-comparison.md`, MLflow `audit_mlflow.py` |
| **Baseline and advanced models compared** | **PASS** | Compares single-source baselines (`EXP-A` UL `YOLO11n`, `EXP-B` RF `YOLO11n`), merged baseline (`EXP-C` `YOLO11n`), scaled capacity (`EXP-D` `YOLO11s`), controlled hyperparameter/augmentation tuning (`EXP-E` `YOLO11s`, 60 epochs, `lr0=0.0015`), and alternative architecture (`EXP-F` `YOLOv8n`). | `docs/experiments/exp-matrix-comparison.md`, `configs/training.yaml` |
| **Per-class metrics present** | **PASS** | Per-class `P`, `R`, `mAP50`, `mAP50-95`, FP, FN, and `small_object_fn` recorded for all 5 canonical classes on both `roboflow_test` (`82` imgs, `426` instances) and `ultralytics_test` (`120` imgs, `978` instances). | `reports/experiments/EXP-A..F/eval_summary.json`, `error_analysis_*.json` |
| **Evaluation methodology documented** | **PASS** | Locked source-specific test splits (`test_roboflow.yaml`, `test_ultralytics.yaml`) kept isolated; threshold selection performed solely on `val_merged` via `val_conf_sweep.json`. | `docs/experiments/exp-matrix-comparison.md`, `reports/experiments/val_conf_sweep.json` |
| **Final model selected with justification** | **PASS** | `EXP-E` (`YOLO11s`, tuned, `60` epochs) selected: best `roboflow_test` `mAP@50` (`0.5610`), best `ultralytics_test` `mAP@50` (`0.8335`) & F1 (`0.8833`), and highest `mAP@50` across all 5 classes including `no_helmet` (`0.4706` RF / `0.5467` UL). | `models/candidates/EXP-E/EXP-E_seed42_best.pt`, `docs/experiments/exp-matrix-comparison.md` |
| **ONNX predictions checked** | **PASS** | `models/production/safeguard_yolo11s.onnx` (`36.139 MB`, opset `17`, `[1,3,640,640]` → `[1,9,8400]`) verified vs PyTorch: raw forward `max_abs_diff=0.0026245` / `min_cosine_similarity=1.0`, same-tensor NMS `N=10` match rate `1.0` (`mean_iou=0.99989`), full pipeline `N=30` match rate `0.9766` (`bounds_violations=0`), `0` class mismatches, plus CPU/GPU latency benchmark (`49.70 ms` ONNX CPU vs `100.01 ms` PyTorch CPU). | `reports/benchmark/verification_report.json`, `reports/benchmark/benchmark_report.json`, `pytest tests/test_onnx_optimization.py` |

---

### 2.3 MLOPS (Normative: `M-05`, `M-06`, `M-11`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **MLflow runs accessible** | **PASS** | Local `mlflow.db` contains `7` `FINISHED` runs (`SMOKE`, `EXP-A`..`EXP-F`, 34–39 params, 28–66 metrics each); containerized MLflow server (`mlflow/Dockerfile`) responds HTTP `200 OK` on `/health`. | `audit_mlflow.py`, `curl http://127.0.0.1:5000/health` |
| **Model Registry populated** | **PARTIALLY IMPLEMENTED** | Registered model `SafeGuard-PPE-Detector` exists with versions `1` and `2` (`status: READY`, `run_id: 8209a2eee1c4468da88f6616ab700138`) and alias `production -> 2`. MLflow `3.18.0` sets `current_stage="None"` (legacy stage transitions deprecated in favor of aliases). | `audit_mlflow.py` (`client.search_registered_models()`) |
| **DVC model and dataset versions available** | **PARTIALLY IMPLEMENTED** | `dvc.lock` tracks `data/raw/*` MD5 hashes and stage outputs; local remote `localstore` (`.dvc/local_remote`) stores pushed DVC objects; `.gitignore` keeps Git clean of large binaries. No external cloud remote configured for separate machines. | `.venv-train\Scripts\dvc.exe remote list`, `dvc push` |
| **Pipeline reproducible** | **PASS** | `dvc.yaml`, `dvc.lock`, `params.yaml`, and `.dvcignore` are fully synchronized (`dvc status` prints `Data and pipelines are up to date.`). | `.venv-train\Scripts\dvc.exe status` |

---

### 2.4 BACKEND (Normative: `M-08`, `SG-07`, `EXT-03`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **All mandatory endpoints work** | **PASS** | `GET /health`, `GET /model/metadata`, `POST /predict`, `POST /predict/visualize` plus `GET /classes`, `POST /predict/frame`, `POST /predict/compliance`, `GET /compliance/alerts`, `PATCH /compliance/alerts/{event_id}`, `GET /compliance/analytics`, and `POST /compliance/reset` pass tests & live Docker Compose checks. | `.venv-backend\Scripts\pytest.exe backend/tests -q` (`12 passed`), live `/api/predict` (`200`, `12` detections) |
| **Pydantic schemas validated** | **PASS** | All schemas inherit `StrictSchema` (`ConfigDict(extra="forbid")`), rejecting invalid types/ranges with semantic `422`. | `backend/app/schemas.py`, `backend/tests/test_api.py` |
| **Swagger accessible** | **PASS** | `GET /docs` and `GET /openapi.json` return HTTP `200` in both `TestClient` and live Docker containers (`docker-build.yml`). | `test_openapi_docs_and_paths_accessible`, `http://127.0.0.1:8000/docs` (`200`) |
| **Error handling tested** | **PASS** | `400` (`invalid_image`, bad MIME, empty upload, invalid alert status), `404` (`http_404`), `422` (`validation_error`), and `503` (`model_unavailable`) verified. | `backend/tests/test_api.py`, `backend/tests/test_inference_service.py` |

---

### 2.5 FRONTEND (Normative: `M-09`, `SG-08`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Image upload works** | **PASS** | Drag/drop & file input, preview, XHR upload progress, `/predict` canvas, and `/predict/visualize` OpenCV toggle verified in Chrome Playwright. | `frontend/tests/safeguard.spec.ts` (`1 passed`), `02_image_analysis_canvas.png` |
| **Example gallery works** | **PASS** | 6 CC BY 4.0 validation images (`frontend/public/examples/`) call live `/predict` in 1 click. Verified in Playwright. | `frontend/tests/safeguard.spec.ts`, `03_example_gallery.png` |
| **Webcam works** | **PASS** | `LiveMonitoringPage.tsx` + `useWebcamMonitor.ts` capture webcam frames with single-in-flight cap and stale-frame dropping against `/predict/compliance`. | `frontend/tests/safeguard.spec.ts`, `04_live_monitoring.png` |
| **Bounding boxes display correctly** | **PASS** | Canvas overlays & OpenCV render distinct per-class colors, labels, confidence indicators, and track tags cleanly. | `02_image_analysis_canvas.png`, `04_live_monitoring.png` |
| **Latency and FPS are shown** | **PASS** | Displays separate `Server Inference` vs `Client E2E` latency in `ms` plus live webcam `FPS`. | `npm --prefix frontend run lint`, `npm --prefix frontend run build`, Playwright |

---

### 2.6 PRODUCT (Normative: `SG-06`, `SG-09`, `EXT-01`, `EXT-02`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Tracking works** | **PASS** | `ByteTrackTracker` tracks anonymous temporary IDs (`active`, `tentative`, `lost`, `retired`), recovers occluded workers via Stage-2 low-conf matching, expires tracks past `max_age` without ID reuse, and retains zero biometrics/PII. | `pytest tests/test_compliance_and_tracking.py` (`12 passed`) |
| **Alerts behave correctly** | **PASS** | Enforces approved taxonomy only (`no_vest` rejected), distinguishes `unknown`, `uncertain`, `insufficient_visibility`, confirms consecutive frames, suppresses cooldown duplicates, captures `evidence_frame_b64`, and tracks review status. | `tests/test_compliance_and_tracking.py`, `backend/tests/test_api.py` |
| **Analytics display actual recorded data** | **PASS** | `AnalyticsPage.tsx` renders Recharts charts strictly from real FastAPI responses (`useSessionTelemetry`), backed separately by `GET /compliance/analytics`. | `05_analytics.png`, `GET /compliance/analytics` |
| **Monitoring displays actual system metrics** | **PASS** | `ModelMonitoringPage.tsx` tracks live `/health`, `EXP-E-yolo11s-onnx`, `CPUExecutionProvider`, GPU status, mean/P95 latency, FPS, error rate, and request audit table. | `06_model_monitoring.png`, `frontend/tests/safeguard.spec.ts` |

---

### 2.7 INFRASTRUCTURE (Normative: `M-10`, `M-11`, `M-12`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Docker Compose starts all required services** | **PASS** | `docker compose up -d --build --wait` built and ran `frontend` (`79.9MB`), `backend` (`723MB`, zero training deps), and `mlflow` (`1.3GB`) cleanly (`Healthy`); proxied `POST /api/predict` returned `200` (`12` detections). | `docker compose config --quiet`, `docker compose up --build`, `docker_smoke.py` |
| **Persistent volumes work** | **PASS** | `mlflow_data:/mlflow` persists after `docker compose down`; `./models/production:/app/models/production:ro` mounts ONNX artifact cleanly. | `docker volume ls --filter name=safeguard_mlflow_data` |
| **GitHub Actions passes** | **PASS** | Live workflows on `https://github.com/Bril3d/SafeGuard` (`main` push `23db775`): `CI Tests` (run `45841262202`, `1m10s`) and `Docker Build & Smoke Validation` (run `45841262369`, `2m8s`) both finished **`completed / success`**. | `gh run list --repo Bril3d/SafeGuard`, `gh run watch` |
| **README startup instructions work** | **PASS** | Verified locally and in clean fresh-checkout `git clone` (`docker compose config --quiet` + `40 passed, 2 skipped` CPU `pytest`). | `README.md`, `docker compose up --build` |

---

### 2.8 DOCUMENTATION (Normative: `M-01`, `M-03`, `M-13`, `SG-02`, `SG-04`, `SG-05`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **LaTeX technical report** | **PARTIALLY IMPLEMENTED** | [`rapport_technique.tex`](../rapport_technique.tex) written and committed with full verified tables, architecture, experiment results, benchmark comparison, critical limitations, and embedded report charts. Remaining user step: compile into `rapport_technique.pdf` (`pdflatex` twice via MiKTeX/TeX Live/Overleaf). | `Test-Path rapport_technique.tex` (`True`), `Test-Path rapport_technique.pdf` (`False`) |
| **Architecture diagrams** | **PASS** | Mermaid diagrams in `docs/spec/software-architecture.md` + detailed specs in `docs/architecture/tracking.md` & `compliance_engine.md`. | `docs/spec/software-architecture.md` |
| **Dataset audit** | **PASS** | Complete audit (`reports/dataset/dataset_audit.md`), 12 visual sheets, plots, deduplication report, and Hard Hat Workers v10 rejection report. | `reports/dataset/dataset_audit.md`, `docs/datasets/hard_hat_workers_audit.md` |
| **Experimental comparison** | **PASS** | `EXP-A`–`EXP-F` comparison matrix, val confidence sweep, per-class test tables, error analysis, and candidate justification. | `docs/experiments/exp-matrix-comparison.md` |
| **Benchmark report** | **PASS** | Same-hardware PyTorch CPU vs ONNX Runtime CPU vs PyTorch GPU benchmark & parity tables + chart. | `reports/benchmark/benchmark_comparison.md` |
| **Individual team responsibilities** | **PASS** | `docs/implementation/team-responsibilities.md` assigns **Khaled Briki** (`Member 1`), **Mohamed Amine Guesmi** (`Member 2`), and **Oussama Ferchichi** (`Member 3`) to explicit domains and defense roles. | `docs/implementation/team-responsibilities.md` |

---

## 3. Updated Summary Counts Across Audited Criteria

- **Total Audited Criteria**: `38`
  - **PASS**: `33` *(including live `Bril3d/SafeGuard` GitHub Actions runs, synchronized DVC pipeline state `Data and pipelines are up to date.`, and assigned team responsibilities)*
  - **PARTIALLY IMPLEMENTED**: `5` *(`Correct class mapping` header, `DVC reproducibility` cloud remote vs `localstore`, `DVC model/dataset versions` off-host remote, `MLflow Model Registry` alias vs legacy stage, `LaTeX technical report` `.tex` committed pending user `.pdf` compilation)*
  - **NOT TESTED**: `0`
  - **FAIL**: `0`
