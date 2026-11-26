# SafeGuard — Final Dual-Specification Engineering Audit Report

**Audit Date**: 2026-11-26  
**GitHub Remote**: `https://github.com/Bril3d/SafeGuard` (`main`)  
**Normative References Audited Against**:
1. [`docs/reference/professor_project_guide.pdf`](../docs/reference/professor_project_guide.pdf) (authoritative academic & defense precedence — Deep Learning Avancé 2026, 0-slide 100% technical defense)
2. [`docs/reference/Cahier des Charges Projet MLOps.pdf`](../docs/reference/Cahier%20des%20Charges%20Projet%20MLOps.pdf) (full MLOps lifecycle: DVC, MLflow, Prefect, CI/CD/CT, Prometheus, Grafana, Evidently AI, notebooks, rollback)
3. [`docs/reference/safeguard_specification_v1.pdf`](../docs/reference/safeguard_specification_v1.pdf) (`SG-SPEC-001` V1.1 baseline) and [`docs/spec/requirements-traceability.md`](../docs/spec/requirements-traceability.md)

> **Overall Audit Verdict — DUAL-SPECIFICATION READY FOR `rapport_technique.pdf` COMPILATION & LIVE DEFENSE**  
> All code, DVC reproducibility (`dvc status` clean), MLflow experiments/registry, ONNX parity, FastAPI + React + ByteTrack product stack, 6-service Docker Compose runtime (`backend`, `frontend`, `mlflow`, `prefect`, `prometheus`, `grafana`), GitHub Actions CI/CD/CT workflows (green on `main`), Playwright E2E screenshots (`README.md`), and `rapport_technique.tex` source are verified passing. Compiling `rapport_technique.tex` into `rapport_technique.pdf` (deferred to a machine with LaTeX installed) and completing the live instructor defense remain before final academic submission.

---

## 1. Prioritized Remaining Actions & Dual-Spec Notes

| Priority | Audit Item | Status | Current State & Remaining Action |
|---|---|---|---|
| **P0 (Final Defense Deliverable)** | **Compile `rapport_technique.pdf`** | **PARTIALLY IMPLEMENTED** | [`rapport_technique.tex`](../rapport_technique.tex) exists (`True`) with verified tables, metrics, architecture, and figures; compile via `pdflatex` / Overleaf (`rapport_technique.pdf` currently `False`). |
| **P0 (Academic Evaluation Event)** | **Live Technical Defense (15–20 min, Zero Slides)** | **NOT TESTED** | Authoritative [`professor_project_guide.pdf`](../docs/reference/professor_project_guide.pdf) mandates 0 slides + live demo, MLOps UI walkthrough, and IDE Q&A (takes precedence over `Cahier des Charges` slide mention). |
| **P1 (Optional Cross-Host)** | **Off-Host Cloud DVC Remote** | **PARTIALLY IMPLEMENTED** | Local remote `localstore` (`.dvc/local_remote`) passes `dvc status` (`Data and pipelines are up to date.`); configure S3/GCS/GDrive only if syncing binaries across separate hosts. |
| **P2 (Governance / Lifecycle Polish)** | **Metadata, Branching & Live Schedules** | **PARTIALLY IMPLEMENTED** | (a) `configs/class_mapping.yaml` header retains `status: "provisional_pending_user_approval"`; (b) MLflow `3.18.0` uses `@production -> v2` alias (`current_stage="None"`); (c) single `main` branch locally (`ci-tests`/`docker-build` target `main`/`develop`); (d) Prefect CT runs via GitHub Actions cron + CLI (`prefect deploy` optional to schedule inside local Prefect UI); (e) Jupyter notebooks have clean source cells calling `src/` without pre-rendered `.ipynb` outputs. |

---

## 2. Criterion-by-Criterion Audit Matrix

### 2.1 DATA & EXPLORATION (Prof. Guide + MLOps Étapes 1–2)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Correct source versions** | **PASS** | Roboflow Construction Site Safety `v30` (`717` imgs, `5,570` boxes, `CC BY 4.0`) and Ultralytics Construction-PPE (`1,000` imgs, `14,863` boxes, `CC BY 4.0`) pinned in `configs/data.yaml`. | `configs/data.yaml`, `reports/dataset/dataset_audit.md` |
| **Valid annotations** | **PASS** | `0` missing labels, `0` orphan labels, `0` corrupt images, `0` invalid normalized coordinates across `1,717` raw images (`20,433` boxes). | `pytest` (`test_data_pipeline.py`) |
| **Correct class mapping** | **PARTIALLY IMPLEMENTED** | Canonical 5 classes (`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`) enforced identically across harmonization, training, ONNX metadata, FastAPI, and compliance; file header still states `status: "provisional_pending_user_approval"`. | `configs/class_mapping.yaml`, `tests/test_compliance_and_tracking.py` |
| **No known split leakage** | **PASS** | Deduplication dropped `77` images; `final_split_statistics.json` verifies `train_test_sha_overlap: 0`, `train_val_sha_overlap: 0`, `val_test_sha_overlap: 0`, `train_test_scene_group_overlap: 0`. | `reports/dataset/deduplication_report.json`, `final_split_statistics.json` |
| **Versioned EDA & benchmark notebooks** | **PARTIALLY IMPLEMENTED** | `notebooks/01_eda_and_dataset_audit.ipynb` and `notebooks/02_experiments_and_onnx_benchmark.ipynb` provide reproducible EDA/benchmark entry points calling modular `src/` code; cell outputs are empty until executed in Jupyter. | `notebooks/01_eda_and_dataset_audit.ipynb`, `02_experiments_and_onnx_benchmark.ipynb` |
| **DVC reproducibility & local remote** | **PARTIALLY IMPLEMENTED** | 10-stage `dvc.yaml` + `dvc.lock` + `params.yaml` + `.dvcignore` + `.dvc/config` (`localstore`); `dvc status` prints `Data and pipelines are up to date.` Off-host S3/GCS remote not configured. | `.venv-train\Scripts\python.exe -m dvc status` |

---

### 2.2 MODELING & ONNX OPTIMIZATION (Prof. Guide Axe 3 + MLOps Étape 3)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **YOLO experiments recorded** | **PASS** | `7` finished MLflow runs (`SMOKE` + `EXP-A` through `EXP-F`) trained at `imgsz=640`, `seed=42`, tracked in `mlflow.db`. | `docs/experiments/exp-matrix-comparison.md`, `mlflow.db` |
| **Baseline & advanced comparison** | **PASS** | Compares `EXP-A`/`B` single-source `YOLO11n`, `EXP-C` merged `YOLO11n`, `EXP-D` `YOLO11s`, `EXP-E` tuned `YOLO11s` (`60` epochs, `lr0=0.0015`), and `EXP-F` `YOLOv8n`. | `docs/experiments/exp-matrix-comparison.md`, `configs/training.yaml` |
| **Per-class metrics & evaluation discipline** | **PASS** | Per-class `P`, `R`, `mAP50`, `mAP50-95`, FP/FN reported on isolated `roboflow_test` and `ultralytics_test`; confidence thresholds swept solely on `val_merged`. | `reports/experiments/EXP-A..F/`, `val_conf_sweep.json` |
| **Final model selected with justification** | **PASS** | `EXP-E` (`YOLO11s`) selected: highest `roboflow_test` `mAP@50` (`0.5610`), `ultralytics_test` `mAP@50` (`0.8335`), F1 (`0.8833`), and best `no_helmet` AP. | `models/candidates/EXP-E/EXP-E_seed42_best.pt` |
| **ONNX export parity & benchmarks** | **PASS** | `models/production/safeguard_yolo11s.onnx` (`36.139 MB`, opset `17`, `[1,3,640,640]` -> `[1,9,8400]`) verified vs PyTorch (`max_abs_diff=0.0026245`, `min_cosine_similarity=1.0`, NMS `N=10` `mean_iou=0.99989`, full pipeline `N=30` match `0.9766`, CPU latency `49.70 ms` ONNX vs `100.01 ms` PyTorch). | `reports/benchmark/verification_report.json`, `benchmark_report.json`, `test_onnx_optimization.py` |

---

### 2.3 MLOPS, ORCHESTRATION, MONITORING & DRIFT (MLOps Étapes 4, 6, 9)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **MLflow Tracking & Model Registry** | **PARTIALLY IMPLEMENTED** | `mlflow` service runs via `mlflow/Dockerfile` (HTTP `200 /health`, persistent `mlflow_data`). Registry `SafeGuard-PPE-Detector` v1/v2 `READY` with alias `@production -> 2` (`current_stage="None"` under MLflow `3.18.0`). | `docker compose`, `mlflow.db` |
| **Training container & DVC stages** | **PASS** | `docker/training.Dockerfile` packages training/DVC/MLflow/Prefect/Evidently (`training` Compose profile); CI builds it cleanly; `dvc.yaml` covers all 10 stages via `params.yaml`. | `docker/training.Dockerfile`, `dvc.yaml`, `params.yaml` |
| **Prefect workflow orchestration** | **PARTIALLY IMPLEMENTED** | `prefect` service (`prefecthq/prefect:3.9.5-python3.12`, `:4200`) healthy in Compose; `src/orchestration/prefect_flows.py` executes `safeguard-data-pipeline` & `safeguard-training-ct-pipeline` cleanly via CLI dry-run + `pytest`. Cron deployment registration in live UI server via `prefect deploy` is manual/optional. | `python -m src.orchestration.prefect_flows --mode dry-run`, `test_orchestration_and_drift.py` |
| **Structured prediction logging & Prometheus** | **PASS** | `backend/app/monitoring.py` exposes `safeguard_inference_requests_total`, `safeguard_prediction_errors_total`, `safeguard_inference_latency_seconds`, `safeguard_detections_by_class_total`, `safeguard_prediction_confidence` on `/metrics` and writes anonymized JSONL logs. | `GET /metrics`, `backend/tests/test_metrics_and_monitoring.py`, live Prometheus target `up` |
| **Prometheus alerts & Grafana provisioning** | **PASS** | `monitoring/prometheus/prometheus.yml` scrapes `backend:8000/metrics`; `alerts.yml` defines `SafeGuardBackendDown`, `SafeGuardHighErrorRate`, `SafeGuardHighInferenceLatency`; `monitoring/grafana/provisioning/` auto-provisions Prometheus datasource and 5-panel dashboard (`uid: safeguard-ops`). | `docker compose up -d --build --wait` (`prometheus`, `grafana` healthy) |
| **Evidently AI proxy drift detection** | **PASS** | `src/monitoring/drift.py` runs `evidently.presets.DataDriftPreset` on `class_name`, `confidence`, and `bbox_area_norm`, labeling findings strictly `proxy_statistical_only_without_ground_truth`, emitting JSON + HTML reports and triggering Prefect CT conditionally. | `python -m src.monitoring.drift`, `reports/monitoring/drift_report.json` |

---

### 2.4 BACKEND, FRONTEND & COMPLIANCE PRODUCT (Prof. Guide 5.1–5.2 + SG-SPEC-001)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **FastAPI endpoints, schemas & Swagger** | **PASS** | `/health`, `/model/metadata`, `/predict`, `/predict/visualize`, `/classes`, `/predict/frame`, `/predict/compliance`, `/compliance/*`, `/metrics`, plus `/docs` OpenAPI UI; strict Pydantic v2 validation (`400`, `404`, `422`, `503`). | `pytest -q` (`45 passed`), `backend/app/main.py` |
| **React + TypeScript + Playwright E2E** | **PASS** | All 8 pages (`Overview`, `Live Monitoring`, `Image Analysis`, `Example Gallery`, `Alert History`, `Analytics`, `Model Monitoring`, `Settings & Model`), drag/drop + `.txt` validation guard, Canvas & OpenCV views, curated gallery, live webcam ByteTrack stream, Recharts, and review queue verified end-to-end with 10 screenshots and 0 console/page errors. | `npm --prefix frontend run lint`, `npm --prefix frontend run build` (`10.01s`), `npm --prefix frontend run test:e2e` (`1 passed, 11.7s`), `docs/assets/screenshots/` |
| **ByteTrack & decision-support boundaries** | **PASS** | Anonymous temporary track IDs only (no PII/biometrics), explicit `no_helmet` confirmation window (`3f`) + cooldown duplicate suppression, `unknown` when positive `helmet`/`vest` absent. | `tests/test_compliance_and_tracking.py` |

---

### 2.5 INFRASTRUCTURE, CI/CD/CT & DOCUMENTATION (Prof. Guide 5.3/6/8 + MLOps Étapes 7–8)

| Criterion | Status | Exact Observed Evidence | Verification Command / Artifact |
|---|---|---|---|
| **Full Docker Compose stack & volumes** | **PASS** | `docker compose up -d --build --wait` verified all 6 services (`backend`, `frontend`, `mlflow`, `prefect`, `prometheus`, `grafana`) `healthy` on Docker Desktop (`29.2.0`); all 8 HTTP `200` checks passed; proxied `/api/predict` returned `12` detections; all 5 named volumes preserved on `docker compose down`. | `docker compose config --quiet`, `docker volume ls --filter name=safeguard_` |
| **GitHub Actions CI / CD / CT** | **PASS** | `CI Tests` (`45845184497`), `Continuous Training (CT) & Drift Check` (`45845184120`), and `Docker Build, Smoke Test & CD Publish` (`45845183928`, pushing `backend`/`frontend`/`mlflow` images to `ghcr.io/bril3d/safeguard-*`) all `completed / success` on `main`. `docker-build.yml` + `deployment_and_rollback.md` cover rollback procedures. | `gh run list --repo Bril3d/SafeGuard`, `docs/ops/deployment_and_rollback.md` |
| **README, code quality & Git hygiene** | **PASS** | `README.md` documents prerequisites, ONNX bootstrap, Docker/bare-metal commands, Prometheus/Evidently/Prefect workflow, and embeds all 10 Playwright screenshots. `black --check .` (`40 files`), `flake8` (`0`), ESLint (`0`), `.gitignore` excludes binaries/`.env`. | `black --check .`, `flake8`, `git status -s` |
| **LaTeX technical report & team charter** | **PARTIALLY IMPLEMENTED** | `rapport_technique.tex` complete; `docs/implementation/team-responsibilities.md` assigns **Khaled Briki**, **Mohamed Amine Guesmi**, **Oussama Ferchichi**. Pending: `pdflatex` compilation to `rapport_technique.pdf`. | `Test-Path rapport_technique.tex` (`True`), `Test-Path rapport_technique.pdf` (`False`) |

---

## 3. Summary Counts Across Audited Criteria

- **Total Audited Criteria**: `25` consolidated dual-specification areas (mapping all `M-01`–`M-14`, `SG-01`–`SG-10`, and `Cahier des Charges` Étapes 1–9 + deliverables)
  - **PASS**: `17`
  - **PARTIALLY IMPLEMENTED**: `7` *(`rapport_technique.pdf` compilation, off-host cloud DVC remote, `class_mapping.yaml` header string, pre-rendered notebook cell outputs, MLflow 3 alias vs legacy stage naming, local Prefect UI cron deployment registration, single local `main` branch)*
  - **NOT TESTED**: `1` *(Live 15–20 min zero-slides instructor technical defense)*
  - **FAIL**: `0`
