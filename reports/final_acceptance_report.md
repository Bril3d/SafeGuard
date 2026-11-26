# SafeGuard — Final Engineering Acceptance Audit Report

**Audit Date**: 2026-11-26  
**Repository Revision**: `main` (`796b25e`)  
**Normative References Audited Against**:
1. [`docs/reference/professor_project_guide.pdf`](../docs/reference/professor_project_guide.pdf) (authoritative precedence)
2. [`docs/reference/safeguard_specification_v1.pdf`](../docs/reference/safeguard_specification_v1.pdf) (`SG-SPEC-001` V1.1)
3. [`docs/spec/requirements-traceability.md`](../docs/spec/requirements-traceability.md) (`M-01`–`M-14`, `SG-01`–`SG-10`, `EXT-01`–`EXT-07`)

> **Overall Audit Verdict — NOT YET READY FOR FINAL DEFENSE SUBMISSION**  
> Core ML pipelines, ONNX optimization, FastAPI runtime, React + TypeScript frontend, anonymous ByteTrack/compliance engine, and local Docker Compose stack are implemented and verified. However, **mandatory academic and MLOps deliverables remain incomplete or unverified on remote runners**—most critically the missing LaTeX technical report (`rapport_technique.pdf`, **FAIL**), unpushed GitHub remote / unverified live GitHub Actions execution (**NOT TESTED**), and incomplete DVC remote / stale `dvc.lock` state (**PARTIALLY IMPLEMENTED**).

---

## 1. Prioritized Blockers & Remediation Queue

| Priority | Audit Item | Status | Impact vs. Normative Spec | Concrete Fix Required |
|---|---|---|---|---|
| **P0 (Blocker)** | **LaTeX Technical Report (`rapport_technique.pdf`)** | **FAIL** | Mandatory deliverable `M-13`, §6 tree, and §8 grading (`2.0 pts`). Zero `.tex` or compiled `.pdf` report files exist in repo. | Author `rapport_technique.tex` covering dataset audit, methodology, `EXP-A`–`EXP-F`, ONNX benchmark, FastAPI/React/Compliance architecture, and critical analysis; compile to `rapport_technique.pdf`. |
| **P0 (Blocker)** | **GitHub Actions Remote Execution (`ci-tests.yml`, `docker-build.yml`)** | **NOT TESTED** | `M-11` / `CA-08` requires green workflow runs on GitHub; defense §9 reviews GitHub Actions UI status. `git remote -v` is currently empty. | Configure GitHub `origin` remote, push `main`, verify both workflows pass cleanly on `ubuntu-latest` runners. |
| **P1 (High)** | **DVC Remote Storage & `dvc.lock` Cleanliness** | **PARTIALLY IMPLEMENTED** | `M-05` / `CA-06` expects `dvc status` clean and pullable `.dvc` artifacts/remote for clean-machine reproduction; currently `.dvc/config` has no remote and `dvc status` flags `split`, `export`, `benchmark`. | Add a DVC remote (`dvc remote add`), track raw/model `.dvc` artifacts or clean cache flags, and refresh `dvc repro` / `dvc push`. |
| **P1 (Medium)** | **Team Member Identity Mapping** | **PARTIALLY IMPLEMENTED** | `M-01` / §2 requires mapping the 3 student identities to `Member 1/2/3` domains. | Update Section 3 of `docs/implementation/team-responsibilities.md` with actual team member names. |
| **P2 (Minor)** | **Governance Metadata Alignment** | **PARTIALLY IMPLEMENTED** | `configs/class_mapping.yaml` still carries `status: "provisional_pending_user_approval"`, MLflow uses alias `@production` with `current_stage="None"` (MLflow 3.18 deprecation), and frontend Alert/Analytics pages use session memory rather than syncing backend `GET /compliance/alerts` & `GET /compliance/analytics`. | Optionally mark `class_mapping.yaml` approved, document MLflow alias convention for defense, and optionally wire frontend initial hydration to `/compliance/alerts` & `/compliance/analytics`. |

---

## 2. Criterion-by-Criterion Audit Matrix

### 2.1 DATA (Normative: `M-04`, `M-05`, `SG-01`, `SG-02`, `SG-03`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Correct source versions** | **PASS** | Roboflow Construction Site Safety `v30` (`717` imgs, `5,570` boxes, `CC BY 4.0`) and Ultralytics Construction-PPE (`1,000` imgs, `14,863` boxes, `CC BY 4.0`, no native test set) pinned in `configs/data.yaml`. Hard Hat Workers v10 evaluated separately and rejected for 5-class merge. | `configs/data.yaml`, `reports/dataset/dataset_audit.md`, `docs/datasets/hard_hat_workers_audit.md` |
| **Valid annotations** | **PASS** | `0` missing labels, `0` orphan labels, `0` corrupt images, `0` invalid normalized coordinates across all `1,717` raw images (`20,433` bounding boxes). Unit tests cover malformed coordinates & corrupt images. | `reports/dataset/dataset_audit.md`, `pytest tests/test_data_pipeline.py` (`6 passed`) |
| **Correct class mapping** | **PARTIALLY IMPLEMENTED** | Canonical 5-class taxonomy (`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`) is implemented identically across harmonization, training YAMLs, ONNX metadata, backend schemas, and compliance taxonomy gate, with `no_vest` held as `pending_approval` and single-source classes excluded. **Gap**: top-level metadata in `configs/class_mapping.yaml` still reads `status: "provisional_pending_user_approval"`. | `configs/class_mapping.yaml`, `pytest tests/test_compliance_and_tracking.py` |
| **No known split leakage** | **PASS** | `deduplication_report.json` dropped `77` images (`26` `drop_scene_group_test_leakage`, `13` `drop_test_leakage`, `18` `drop_val_leakage`, `20` `drop_within_split_duplicate`). `final_split_statistics.json` verifies `train_test_sha_overlap: 0`, `train_val_sha_overlap: 0`, `val_test_sha_overlap: 0`, and `train_test_scene_group_overlap: 0`. (`train_val_scene_group_overlap: 6` is tracked and acceptable because val is non-test.) | `reports/dataset/deduplication_report.json`, `reports/dataset/final_split_statistics.json`, `pytest tests/test_harmonization_pipeline.py` (`1 passed`) |
| **DVC reproducibility** | **PARTIALLY IMPLEMENTED** | `dvc.yaml` (10 stages: `ingest`→`validate`→`harmonize`→`deduplicate`→`split`→`statistics`→`train`→`evaluate`→`export`→`benchmark`) and `dvc.lock` exist locally; `.dvcignore` exists. **Gap**: `.dvc/config` has no remote (`dvc remote list` is empty), zero `.dvc` files are committed to Git, and `dvc status` / `dvc repro --dry` reports `split`, `export`, and `benchmark` as out-of-sync (`.cache` files inside `data/processed/` and post-lock edits). | `.venv-train\Scripts\dvc.exe status`, `.venv-train\Scripts\dvc.exe remote list`, `.dvc/config`, `dvc.lock` |

---

### 2.2 MODEL (Normative: `M-04`, `M-07`, `SG-04`, `SG-05`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **YOLO experiments recorded** | **PASS** | `7` finished MLflow runs (`SMOKE` + `EXP-A` through `EXP-F`) trained on `NVIDIA GeForce RTX 5070 Laptop GPU` at `imgsz=640`, `seed=42`, tracked in `sqlite:///mlflow.db`. | `docs/experiments/exp-matrix-comparison.md`, MLflow `audit_mlflow.py` |
| **Baseline and advanced models compared** | **PASS** | Compares single-source baselines (`EXP-A` UL `YOLO11n`, `EXP-B` RF `YOLO11n`), merged baseline (`EXP-C` `YOLO11n`), scaled capacity (`EXP-D` `YOLO11s`), controlled hyperparameter/augmentation tuning (`EXP-E` `YOLO11s`, 60 epochs, `lr0=0.0015`), and alternative architecture (`EXP-F` `YOLOv8n`). | `docs/experiments/exp-matrix-comparison.md`, `configs/training.yaml` |
| **Per-class metrics present** | **PASS** | Per-class `P`, `R`, `mAP50`, `mAP50-95`, FP, FN, and `small_object_fn` recorded for all 5 canonical classes on both `roboflow_test` (`82` imgs, `426` instances) and `ultralytics_test` (`120` imgs, `978` instances). | `reports/experiments/EXP-A..F/eval_summary.json`, `error_analysis_*.json` |
| **Evaluation methodology documented** | **PASS** | Locked source-specific test splits (`test_roboflow.yaml`, `test_ultralytics.yaml`) kept isolated; threshold selection performed solely on `val_merged` via `val_conf_sweep.json`. | `docs/experiments/exp-matrix-comparison.md`, `reports/experiments/val_conf_sweep.json` |
| **Final model selected with justification** | **PASS** | `EXP-E` (`YOLO11s`, tuned, `60` epochs) selected: best `roboflow_test` `mAP@50` (`0.5610` vs `0.5338` `EXP-D`, `0.5089` `EXP-C`), best `ultralytics_test` `mAP@50` (`0.8335`) & F1 (`0.8833`), and highest `mAP@50` across all 5 classes including `no_helmet` (`0.4706` RF / `0.5467` UL). | `models/candidates/EXP-E/EXP-E_seed42_best.pt`, `docs/experiments/exp-matrix-comparison.md` |
| **ONNX predictions checked** | **PASS** | `models/production/safeguard_yolo11s.onnx` (`36.139 MB`, opset `17`, input `[1,3,640,640]`, output `[1,9,8400]`) verified vs PyTorch on `val` images: raw forward `max_abs_diff=0.0026245` / `min_cosine_similarity=1.0`, same-tensor NMS `N=10` match rate `1.0` (`mean_iou=0.99989`), full pipeline `N=30` match rate `0.9766` (`bounds_violations=0`), `0` class mismatches, plus CPU/GPU latency benchmark (`49.70 ms` ONNX CPU infer vs `100.01 ms` PyTorch CPU infer). | `reports/benchmark/verification_report.json`, `reports/benchmark/benchmark_report.json`, `pytest tests/test_onnx_optimization.py` (`6 passed`) |

---

### 2.3 MLOPS (Normative: `M-05`, `M-06`, `M-11`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **MLflow runs accessible** | **PASS** | Local `mlflow.db` contains `7` `FINISHED` runs (`SMOKE`, `EXP-A`..`EXP-F`, 34–39 params, 28–66 metrics each); containerized MLflow server (`mlflow/Dockerfile`) responds HTTP `200 OK` on `/health`. | `audit_mlflow.py`, `curl http://127.0.0.1:5000/health` |
| **Model Registry populated** | **PARTIALLY IMPLEMENTED** | Registered model `SafeGuard-PPE-Detector` exists with versions `1` and `2` (`status: READY`, `run_id: 8209a2eee1c4468da88f6616ab700138`, `EXP-E_seed42`) and alias `production -> 2`. **Gap**: `current_stage` is `"None"` because MLflow `3.18.0` deprecated legacy stage transitions in favor of aliases; verify whether defense rubric expects alias explanation or explicit stage field. | `audit_mlflow.py` (`client.search_registered_models()`) |
| **DVC model and dataset versions available** | **PARTIALLY IMPLEMENTED** | Local `.dvc/cache` and `dvc.lock` track raw directories (`data/raw/*` MD5 hashes) and manifests, and `.gitignore` excludes binaries cleanly (`0` `.onnx`/`.pt`/raw images tracked by Git). **Gap**: no `.dvc` pointer files (`*.dvc`) are committed for `models/candidates/`, `models/production/`, or `data/raw/`, and no DVC remote is configured for `dvc pull`. | `git ls-files "*.dvc"` (empty), `dvc remote list` (empty), `dvc.lock` |
| **Pipeline reproducible** | **PARTIALLY IMPLEMENTED** | Reproducible scripts and `dvc.yaml` exist, and stages `ingest`..`evaluate` report unchanged, but `split`, `export`, and `benchmark` report modified deps/outs under `dvc repro --dry`. | `.venv-train\Scripts\dvc.exe repro --dry` |

---

### 2.4 BACKEND (Normative: `M-08`, `SG-07`, `EXT-03`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **All mandatory endpoints work** | **PASS** | `GET /health`, `GET /model/metadata`, `POST /predict`, `POST /predict/visualize` plus `GET /classes`, `POST /predict/frame`, `POST /predict/compliance`, `GET /compliance/alerts`, `PATCH /compliance/alerts/{event_id}`, `GET /compliance/analytics`, and `POST /compliance/reset` all pass `TestClient` integration tests and live Docker Compose checks. | `.venv-backend\Scripts\pytest.exe backend/tests -q` (`12 passed`), live `/api/predict` (`200`, `12` detections) |
| **Pydantic schemas validated** | **PASS** | All request/response schemas in `backend/app/schemas.py` inherit `StrictSchema` (`ConfigDict(extra="forbid")`) with numeric/range validators tested via `422` failure cases. | `backend/app/schemas.py`, `backend/tests/test_api.py` |
| **Swagger accessible** | **PASS** | `GET /docs` and `GET /openapi.json` return HTTP `200` in both mocked `TestClient` and live running Docker container. | `test_openapi_docs_and_paths_accessible`, `http://127.0.0.1:8000/docs` (`200`) |
| **Error handling tested** | **PASS** | `400` (`invalid_image`, corrupt payload, unsupported MIME, empty file, invalid alert status), `404` (`http_404` unknown alert ID), `422` (`validation_error`), and `503` (`model_unavailable` on unloaded or corrupt ONNX file) verified via automated tests. | `backend/tests/test_api.py`, `backend/tests/test_inference_service.py` |

---

### 2.5 FRONTEND (Normative: `M-09`, `SG-08`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Image upload works** | **PASS** | `ImageAnalysisPage.tsx` supports drag/drop & file input, preview, XHR upload progress, `/predict` canvas rendering, and `/predict/visualize` annotated toggle. Verified in real Chrome via Playwright (`run_pw_e2e.py`). | `frontend/tests/safeguard.spec.ts` (`1 passed`), `02_image_analysis_canvas.png` |
| **Example gallery works** | **PASS** | `ExampleGalleryPage.tsx` serves 6 real CC BY 4.0 validation images (`frontend/public/examples/`) and runs 1-click live `/predict` inference. Verified in Playwright. | `frontend/tests/safeguard.spec.ts`, `03_example_gallery.png` |
| **Webcam works** | **PASS** | `LiveMonitoringPage.tsx` + `useWebcamMonitor.ts` capture browser webcam frames cleanly, limit concurrent requests (single in-flight + stale-frame drop), and hit `/predict/compliance`. Verified in Playwright with Chrome fake media stream (`>= 2` frames completed cleanly). | `frontend/tests/safeguard.spec.ts`, `04_live_monitoring.png` |
| **Bounding boxes display correctly** | **PASS** | `BoundingBoxCanvas.tsx` and Live Canvas draw distinct per-class colors, labels, confidence percentages, and track tags aligned to image coordinates; visually confirmed on captured Playwright screenshots. | `02_image_analysis_canvas.png`, `04_live_monitoring.png` |
| **Latency and FPS are shown** | **PASS** | Displays separate `Server Inference` vs `Client E2E` latency in `ms` plus live webcam `FPS` across top telemetry strip, pages, and `ModelMonitoringPage`. | `npm --prefix frontend run lint` (`0`), `npm --prefix frontend run build` (`0`), Playwright screenshots |

---

### 2.6 PRODUCT (Normative: `SG-06`, `SG-09`, `EXT-01`, `EXT-02`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Tracking works** | **PASS** | `ByteTrackTracker` tracks anonymous temporary `track_id`s across frames (`active`, `tentative`, `lost`, `retired`), recovers occluded detections via Stage-2 low-conf matching, expires tracks past `max_age` without ID reuse, and retains zero biometrics/PII. | `pytest tests/test_compliance_and_tracking.py` (`12 passed`) |
| **Alerts behave correctly** | **PASS** | `AlertManager` & `ComplianceInterpreter` enforce approved taxonomy only (`no_vest` ignored), never alert on missing positive PPE, distinguish `unknown`, `uncertain`, and `insufficient_visibility`, require `confirm_frames` consecutive active detections, suppress cooldown duplicates, capture annotated `evidence_frame_b64`, and manage `event_status` transitions. | `tests/test_compliance_and_tracking.py`, `backend/tests/test_api.py` (`PATCH /compliance/alerts/{id}`) |
| **Analytics display actual recorded data** | **PASS** | Frontend `AnalyticsPage.tsx` renders Recharts class-count bars and server-vs-client latency lines strictly from real session responses recorded in `useSessionTelemetry.ts` (`05_analytics.png`), and backend exposes `GET /compliance/analytics`. *(Minor note: UI session charts track client session calls rather than fetching backend persistent stream analytics on page reload.)* | `05_analytics.png`, `frontend/tests/safeguard.spec.ts`, `GET /compliance/analytics` (`test_api.py`) |
| **Monitoring displays actual system metrics** | **PASS** | `ModelMonitoringPage.tsx` reports live `/health` status, `EXP-E-yolo11s-onnx`, `CPUExecutionProvider`, GPU status, mean/P95 response time, last server inference `ms`, live `FPS`, API error count, and per-request telemetry table (`06_model_monitoring.png`). | `06_model_monitoring.png`, `frontend/tests/safeguard.spec.ts` |

---

### 2.7 INFRASTRUCTURE (Normative: `M-10`, `M-11`, `M-12`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Docker Compose starts all required services** | **PASS** | `docker compose up -d --build --wait` built and launched `safeguard-frontend` (`79.9MB`), `safeguard-backend` (`723MB`, zero PyTorch/Ultralytics/MLflow), and `safeguard-mlflow` (`1.3GB`) on `safeguard_net`; all 3 passed healthchecks (`Healthy`), and proxied `POST http://127.0.0.1:3000/api/predict` returned `200` with `12` detections (`121.005 ms` ONNX CPU inference). | `docker compose config --quiet`, `docker compose up -d --build --wait`, `docker_smoke.py` |
| **Persistent volumes work** | **PASS** | Named volume `mlflow_data:/mlflow` persisted `local safeguard_mlflow_data` after `docker compose down`; `./models/production:/app/models/production:ro` cleanly serves read-only ONNX artifact. | `docker volume ls --filter name=safeguard_mlflow_data` |
| **GitHub Actions passes** | **NOT TESTED** *(Workflows implemented)* | Both `.github/workflows/ci-tests.yml` and `.github/workflows/docker-build.yml` are committed and validated locally (CPU `pytest` `42 passed` in `.venv-train`, `40 passed, 2 skipped` in clean fresh checkout, `black`/`flake8`/`eslint`/`tsc`/`docker compose build` + smoke checks). **Blocker**: `git remote -v` is empty—no remote GitHub repository is linked or pushed yet, so hosted GitHub runner execution is **NOT TESTED**. | `git remote -v` (empty), `.github/workflows/*.yml`, local `fresh_checkout` test |
| **README startup instructions work** | **PASS** | Tested `README.md` workflows: bare-metal checks + clean `git clone` (`docker compose config --quiet` + CPU `pytest`) + `docker compose up --build` all succeeded without hidden steps; cold-start unloaded model behavior documented. | `README.md`, `docker compose up --build`, clean clone check |

---

### 2.8 DOCUMENTATION (Normative: `M-01`, `M-03`, `M-13`, `SG-02`, `SG-04`, `SG-05`)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **LaTeX technical report** | **FAIL** | **Missing deliverable**: `Test-Path rapport_technique.pdf` is `False`; `Test-Path rapport_technique.tex` is `False`; zero `.tex` files exist in the repository. Required by Prof. Guide (`M-13`, §6, §8) and `GEMINI.md`. | `Test-Path rapport_technique.pdf` (`False`), recursive `*.tex` search (empty) |
| **Architecture diagrams** | **PASS** | `docs/spec/software-architecture.md` includes Mermaid subsystem, offline data-flow, and runtime architecture diagrams; detailed tracking & compliance docs in `docs/architecture/tracking.md` and `docs/architecture/compliance_engine.md`. | `docs/spec/software-architecture.md`, `docs/architecture/*.md` |
| **Dataset audit** | **PASS** | Comprehensive audit (`reports/dataset/dataset_audit.md`), 12 visual annotation sheets, distribution/bbox plots, `deduplication_report.json`, and Hard Hat Workers v10 rejection audit (`docs/datasets/hard_hat_workers_audit.md`). | `reports/dataset/dataset_audit.md`, `docs/datasets/hard_hat_workers_audit.md` |
| **Experimental comparison** | **PASS** | `EXP-A`–`EXP-F` comparison matrix with validation sweep, locked source-specific test metrics, per-class tables, error analysis, and candidate selection documented cleanly. | `docs/experiments/exp-matrix-comparison.md`, `exp-a-review.md`, `exp-b-c-comparison.md` |
| **Benchmark report** | **PASS** | `reports/benchmark/benchmark_comparison.md`, `benchmark_report.json`, `verification_report.json`, and `latency_comparison.png` report same-hardware PyTorch CPU vs ONNX CPU vs PyTorch GPU performance & parity. | `reports/benchmark/benchmark_comparison.md` |
| **Individual team responsibilities** | **PARTIALLY IMPLEMENTED** | `docs/implementation/team-responsibilities.md` clearly defines `Member 1` / `Member 2` / `Member 3` domains and zero-slide technical defense format, but actual student names remain unassigned under Section 3 (`Unresolved Team Decisions`). | `docs/implementation/team-responsibilities.md` |

---

## 3. Summary Counts Across Audited Criteria

- **Total Audited Criteria**: `31`
  - **PASS**: `24`
  - **PARTIALLY IMPLEMENTED**: `5` (`Class mapping approval header`, `DVC reproducibility / remote & .dvc pointers`, `MLflow Model Registry stage vs alias`, `Pipeline reproducibility dry-run`, `Team responsibilities student names`)
  - **NOT TESTED**: `1` (`GitHub Actions hosted runner execution` — workflows written & locally verified, blocked only on configuring/pushing GitHub remote)
  - **FAIL**: `1` (`LaTeX technical report rapport_technique.pdf` missing)

---

## 4. Recommended Next Actions Before Defense

1. **Resolve Blocker 1 (`FAIL`)**: Author `rapport_technique.tex` (pulling exact verified tables/figures from `reports/dataset/`, `docs/experiments/exp-matrix-comparison.md`, `reports/benchmark/`, `docs/spec/`, and `docs/architecture/`) and compile `rapport_technique.pdf` at repo root.
2. **Resolve Blocker 2 (`NOT TESTED`)**: Attach the team's GitHub remote (`git remote add origin ...`), push `main`, and verify `.github/workflows/ci-tests.yml` and `.github/workflows/docker-build.yml` pass green on GitHub Actions.
3. **Close P1 Gaps (`PARTIALLY IMPLEMENTED`)**: Configure a DVC remote (local/GDrive/S3) and refresh `dvc.lock` / commit `.dvc` pointers so `dvc status` is clean, finalize `configs/class_mapping.yaml` approval status, and fill real team member names into `docs/implementation/team-responsibilities.md`.
