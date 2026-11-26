# SafeGuard — CI/CD/CT Promotion & Rollback Runbook

## 1. Environments & Promotion Flow

1. **Development (`local / PR`)**:
   - Triggered on every pull request targeting `main` or `develop`.
   - Runs CPU-only `.github/workflows/ci-tests.yml` (`black`, `flake8`, `pytest`, `eslint`, `tsc`/`vite build`) and `.github/workflows/docker-build.yml` (`docker compose config --quiet`, image builds, container health/smoke tests).
2. **Container Registry (`ghcr.io/bril3d/safeguard-*`)**:
   - On pushes to `main`, `docker-build.yml` publishes immutable SHA-tagged (`sha-<short_sha>`) and `latest` images for `backend`, `frontend`, and `mlflow` to GitHub Container Registry (GHCR).
3. **Model Promotion (`MLflow Model Registry`)**:
   - Candidate checkpoints (`EXP-A`–`EXP-F`) are logged in `SafeGuard-PPE-Detector`.
   - Promoted weights export to `models/production/safeguard_yolo11s.onnx` after passing `src/models/export_onnx.py` numerical parity checks and are tagged with alias `@production`.

## 2. Continuous Training (CT) Workflow

- `.github/workflows/continuous-training.yml` (plus `src/orchestration/prefect_flows.py`) runs scheduled weekly or manual (`workflow_dispatch`) checks:
  1. Executes `src/monitoring/drift.py` against prediction logs vs. validation reference baseline.
  2. Flags proxy/statistical drift when share of drifted features or confidence shift exceeds threshold (`drift_detected=true`).
  3. Triggers Prefect `training_ct_flow` (`dvc repro --dry` in CPU CI / full `dvc repro` on GPU training host via `docker/training.Dockerfile`) and prevents unverified promotion unless validation metrics and ONNX parity pass.

## 3. Rollback Strategy on Failure

1. **Model Artifact Rollback (Zero Image Rebuild)**:
   - Reassign `@production` alias in MLflow to the previous verified model version (e.g., `version 1`) and restore the matching `.onnx` + `.metadata.json` in `models/production/`, then restart backend (`docker compose restart backend`).
2. **Container Image Rollback**:
   - Pull the previous green SHA tag from GHCR (`ghcr.io/bril3d/safeguard-backend:sha-<prev>`) and `ghcr.io/bril3d/safeguard-frontend:sha-<prev>` and redeploy via `docker compose up -d`.
3. **Health Gate Verification**:
   - Confirm `GET /health` returns `"model_loaded": true` and Prometheus target `up{job="safeguard-backend"} == 1` before routing traffic.
