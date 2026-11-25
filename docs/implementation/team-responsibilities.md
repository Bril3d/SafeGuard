# Team Responsibilities & Defense Accountability

**Sources**: [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (Section 2, p.2; Section 9, p.11) and [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (Section 13, p.19; Section 15, p.21).

---

## 1. Three-Member Role Matrix

Per both normative documents, the project is executed by a **three-person team** (`trinôme`). Primary ownership organizes delivery without removing **collective solidarity and individual accountability**: every member must understand the full data-to-demo pipeline and answer individual technical questions during the code review.

| Role | Title | Primary Ownership | Shared & Cross-Cutting Responsibilities |
|---|---|---|---|
| **Member 1** | **Lead Deep Learning / Research Scientist** | - Raw dataset download audit & `Appendix A` freeze checklist<br>- 7-step Taxonomy Approval Gate (`configs/class_mapping.yaml`)<br>- Data quality, validation, harmonization, deduplication & leakage-aware split scripts (`src/data/`)<br>- Training & evaluation (`EXP-A`..`EXP-F`: `YOLO11n`, `YOLO11s`, `YOLOv8n`) in `src/models/`<br>- Validation-only threshold/NMS tuning & error analysis<br>- ONNX export, parity verification & runtime benchmarking (`export_onnx.py`, `benchmark.py`)<br>- Compliance rule semantics (`src/compliance/rules.py`) | - Co-define inference & model metadata contract with Member 3<br>- Coordinate MLflow run logging & artifact schemas with Member 2<br>- Author methodology, dataset audit, and experiment sections of `rapport_technique.pdf`<br>- Defend modeling choices, dataset ethics/leakage, and limitations |
| **Member 2** | **Lead MLOps / DevOps Engineer** | - Standardized Git repo setup, branch rules & `.gitignore` / `.dvcignore` hygiene<br>- DVC remote configuration, `dvc.yaml` pipeline & `params.yaml`<br>- MLflow tracking server & Model Registry setup (persistent storage & promotion workflow)<br>- Multi-stage Dockerfiles (`backend/Dockerfile`, `frontend/Dockerfile`) & `docker-compose.yml`<br>- GitHub Actions CI/CD (`ci-tests.yml`, `docker-build.yml`)<br>- Inference latency profiling & operational monitoring metrics design | - Integrate training & export scripts into `dvc repro` with Member 1<br>- Ensure runtime `backend/` container excludes training deps & datasets<br>- Author MLOps, reproducibility, and deployment sections of `rapport_technique.pdf`<br>- Lead the 5-minute MLOps defense review (MLflow, DVC, GitHub Actions) |
| **Member 3** | **Lead Full-Stack Software Engineer** | - FastAPI REST API (`backend/app/main.py`, `schemas.py`, `inference.py`)<br>- ONNX Runtime session loading & MIME/image validation + security limits<br>- Mandatory endpoints (`/health`, `/model/metadata`, `/predict`, `/predict/visualize`) & permitted extensions (`/classes`, `/predict/frame`)<br>- Backend Pytest suite (`backend/tests/test_api.py`)<br>- React + TypeScript UI (`frontend/src/`: upload, preloaded example gallery, bounded real-time webcam, canvas boxes, confidence bars, separate client/server latencies, potential alerts)<br>- Frontend ESLint/build checks & Playwright E2E tests | - Start API/UI scaffolding early (Jalon 2) and eliminate mocks before Jalon 3/4<br>- Co-define ONNX wrapper interface & compliance payload with Member 1<br>- Author architecture, API, UI, and real-time trade-offs in `rapport_technique.pdf`<br>- Lead the 5-minute live application defense demo |

---

## 2. Technical Defense Structure (15–20 Minutes, Zero Slides)

Per [`professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (p.10–11) and [`safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (p.21):

- **Strict Rule**: **Zero slides / PowerPoint**. All preparation effort goes into working software, MLOps traceability, automated tests, and the LaTeX report.
- **Session Breakdown**:
  1. **Live Application Demo (5 minutes)**: Start clean via `docker compose up --build`; demonstrate image upload, example gallery, and live webcam inference with bounding boxes, confidence indicators, alerts, and separate server/client latencies.
  2. **MLOps Review (5 minutes)**: Present MLflow UI (experiment runs `EXP-A`..`EXP-F`, parameters, metrics, artifacts, Model Registry status), DVC versioning (`dvc status`, `dvc repro`), and GitHub Actions workflow runs.
  3. **IDE Code Review & Individual Questions (5–10 minutes)**: Instructor inspects source code directly in the IDE and questions each member individually on architecture, design choices, and system limitations.

---

## 3. Unresolved Team Decisions (Pending Approval)

1. **Member Name Assignment**: Mapping the three team members' actual names/Git identities to `Member 1`, `Member 2`, and `Member 3`.
2. **LaTeX Report Lead Editor**: Confirming which member coordinates final LaTeX compilation (`rapport_technique.pdf`) while all three contribute sections.
