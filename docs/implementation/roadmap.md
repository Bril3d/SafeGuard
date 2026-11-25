# Implementation Roadmap & Milestones

**Sources**: [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (Section 7, p.8; Section 9, p.11) and [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (Sections 7, 13, 14, p.11, p.19–20).

---

## 1. Academic Milestones (`Jalons`)

The professor's guide defines 4 mandatory milestones (`p.8`) with explicit validation checkpoints:

| Milestone | Deadline | Mandatory Objectives & Deliverables | Professor Validation | Linked Acceptance Criteria |
|---|---|---|---|---|
| **Jalon 1 — Specification & Kickoff** | End of Week 2 | Axis 3 PPE subject confirmed; GitHub repository initialized with standardized structure; pinned datasets downloaded, audited, and validated; DVC tracking & remote configured. | GitHub Repository | `CA-06`, `CA-07` |
| **Jalon 2 — Modeling & Experimentation** | End of Week 4 | Clean, leakage-checked data pipeline (`dvc.yaml` stages); source baselines (`EXP-A`, `EXP-B`) and fusion/comparison runs (`EXP-C`..`EXP-F`) trained and tracked in MLflow; validation metrics & NMS/threshold experiments recorded; **early FastAPI & React scaffolding started** (mock responses only if needed during early UI dev, clearly labeled). | MLflow Runs | `CA-05`, `CA-06`, `CA-07` |
| **Jalon 3 — FastAPI Backend & Containerization** | End of Week 6 | Selected checkpoint exported to ONNX & parity/benchmark verified; MLflow Model Registry candidate promoted; operational async FastAPI service with `/docs` Swagger, Pydantic schemas, ONNX Runtime inference; Pytest unit/API suite passing; multi-stage backend `Dockerfile` ready. | API Demo | `CA-02`, `CA-04`, `CA-08` |
| **Jalon 4 — Full-Stack Integration, CI/CD & Technical Defense** | Final Session | React + TypeScript frontend connected to real FastAPI backend (upload, example gallery, bounded real-time webcam, compliance alerts, latency/FPS); `docker compose up --build` starts `frontend`, `backend`, `mlflow`; GitHub Actions CI green; `rapport_technique.pdf` (LaTeX) and `README.md` finalized; 15–20 min no-slides live defense. | Technical Defense | `CA-01`..`CA-09` |

---

## 2. Incremental Implementation Phases (SafeGuard Execution Plan)

Per `GEMINI.md` Rule 16 and the Mandatory Pre-Implementation Workflow, features are implemented incrementally phase by phase with explicit user approval between phases:

1. **Phase 0 — Architecture & Spec Reconciliation (Current)**:
   - Produce `GEMINI.md`, specification traceability, architecture decisions, API contract, dataset strategy, roadmap, team responsibilities, and risks.
   - Resolve open architecture/infrastructure/taxonomy-gate decisions.
2. **Phase 1 — Repository Scaffolding & MLOps Foundation (`Jalon 1`)**:
   - Initialize directory layout, `.gitignore`, `.dvcignore`, DVC remote config, `configs/`, `params.yaml`, and baseline CI lint/test workflow skeleton.
3. **Phase 2 — Dataset Ingestion, Audit & DVC Pipeline (`Jalon 1` -> `Jalon 2` Gate)**:
   - Implement TDD-tested `src/data/` modules (`inspect`, `validate`, `harmonize`, `deduplicate`, `split`, `statistics`) and `dvc.yaml`.
   - Execute archive audit & 7-step Taxonomy Approval Gate (`configs/class_mapping.yaml`), freeze locked splits (`Appendix A`).
4. **Phase 3 — YOLO Experimentation, MLflow & Early API/UI Scaffolding (`Jalon 2`)**:
   - Run `EXP-A` through `EXP-F` (`YOLO11n`, `YOLO11s`, `YOLOv8n`), log all params/metrics/artifacts to MLflow, tune confidence/IoU thresholds on validation data only, evaluate locked test sets, and register candidate model.
   - In parallel (avoiding "Notebook-Only Syndrome"), scaffold `backend/` Pydantic contracts and `frontend/` layout.
5. **Phase 4 — ONNX Export, Parity, Compliance Engine & FastAPI Backend (`Jalon 3`)**:
   - Export selected checkpoint to ONNX (`src/models/export_onnx.py`), run parity & runtime benchmarks (`src/models/benchmark.py`), implement TDD-tested `src/compliance/rules.py`, complete `backend/app/` (`main.py`, `schemas.py`, `inference.py`), `backend/tests/test_api.py`, and `backend/Dockerfile`.
6. **Phase 5 — React Frontend, Real-Time Webcam & Operational Monitoring (`Jalon 4`)**:
   - Implement upload, preloaded licensed example gallery, bounded real-time webcam scheduling, bounding-box canvas, animated confidence indicators, potential alerts, separate client/server latency, and operational monitoring views.
   - Replace any early scaffolding mocks completely with real FastAPI/ONNX inference; verify with Playwright.
7. **Phase 6 — Docker Compose, CI/CD Hardening & Extensions (`Jalon 4`)**:
   - Finalize `frontend/Dockerfile`, `docker-compose.yml` (`frontend`, `backend`, `mlflow`), and `.github/workflows/` (`ci-tests.yml`, `docker-build.yml`).
   - Implement approved optional extensions (`src/tracking/tracker.py`, `Analytics` views) only after core is green.
8. **Phase 7 — LaTeX Technical Report & Defense Readiness (`Jalon 4`)**:
   - Compile `rapport_technique.pdf` in LaTeX with verified MLflow/DVC/benchmark figures, finalize `README.md`, and rehearse the 15–20 min zero-slides technical defense.

---

## 3. Mandatory Acceptance Criteria (`CA-01` to `CA-09`)

| ID | Acceptance Criterion (`SG-SPEC-001` p.20) | Required Evidence |
|---|---|---|
| `CA-01` | `docker compose up --build` starts the expected local services (`frontend`, `backend`, `mlflow`) | Clean-start log and passing container health checks |
| `CA-02` | `/docs` exposes the API and image prediction works | Swagger UI check and verified `POST /predict` & `/predict/visualize` responses |
| `CA-03` | Webcam shows refreshed bounding boxes and confidence scores | Live browser demonstration with separate server/client latencies |
| `CA-04` | Backend demonstration actually uses ONNX Runtime | Runtime configuration/startup log, `GET /health`, and PyTorch-vs-ONNX parity test |
| `CA-05` | MLflow shows source baselines (`EXP-A`/`B`) and harmonized-fusion experiments (`EXP-C`..`F`) | MLflow UI run comparison & registered model record |
| `CA-06` | DVC identifies data versions and reproduces pipeline stages | `dvc status` and `dvc repro` execution evidence |
| `CA-07` | Final classes match the approved semantic intersection | `configs/class_mapping.yaml`, dataset audit, and annotation gallery |
| `CA-08` | GitHub Actions workflow passes | Green `ci-tests.yml` and `docker-build.yml` runs |
| `CA-09` | Each team member explains their implementation and limitations | Individual code review and Q&A during defense |

---

## 4. Unresolved Roadmap Decisions (Pending Approval)

1. **Exact Calendar Dates for Jalons 1–4**: Mapping Week 2 / Week 4 / Week 6 / Final Session to actual academic semester dates.
2. **Extension Gate Timing**: Confirming which optional portfolio extensions (`EXT-01` ByteTrack tracking, `EXT-02` Analytics, `EXT-06` FP16/INT8 quantization) are targeted after Phase 5 core completion.
