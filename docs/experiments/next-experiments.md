# Controlled Next-Experiment Plan (Following EXP-A Review)

**Basis**: Empirical findings from `EXP-A` ([exp-a-review.md](exp-a-review.md), MLflow Run `fedb2987682e438982dd7a0cbdacb676`).  
**Discipline**: Do not modify datasets; do not tune on locked test sets; prioritize controlled experiments that answer one clear research or engineering question at a time.

---

## Priority 1 — `EXP-B` (Source-A Baseline) & `EXP-C` (Harmonized Merged `YOLO11n`)

**Targeted Limitation**: Cross-source generalization gap (`EXP-A` `mAP50 = 0.8106` on `ultralytics_test` vs. `0.3526` on `roboflow_test`).

- **Experiment 1 (`EXP-B`)**: Train `YOLO11n` on `data/processed/roboflow_only.yaml` (`50` epochs, `imgsz=640`, `batch=16`, `seed=42`, `baseline` augmentation).
  - *Question Answered*: What is the Source-A (`roboflow_only`) baseline on both `roboflow_test` and `ultralytics_test`?
- **Experiment 2 (`EXP-C`)**: Train `YOLO11n` on `data/processed/merged.yaml` (`50` epochs, `imgsz=640`, `batch=16`, `seed=42`, `baseline` augmentation).
  - *Question Answered*: Does multi-source semantic fusion (`EXP-C` vs. `EXP-A` and `EXP-B`) significantly lift cross-domain `roboflow_test` mAP50 (especially `person`, `helmet`, `vest`, `gloves`) without degrading `ultralytics_test`, and how does merging affect `no_helmet` given the cross-source box-extent mismatch?

---

## Priority 2 — `EXP-D` (`YOLO11s` Capacity Trade-off on Merged Data)

**Targeted Limitation**: Small-object misses (`97.8%` of `ultralytics_test` FN have `area_norm < 0.01`) and `no_helmet` recall (`0.5172` in `EXP-A`).

- **Experiment 3 (`EXP-D`)**: Train `YOLO11s` on `data/processed/merged.yaml` (`50` epochs, `imgsz=640`, `batch=12`, `seed=42`, `baseline` augmentation) and compare against `EXP-C` (`YOLO11n`).
  - *Question Answered*: Does scaling backbone/neck capacity from `YOLO11n` to `YOLO11s` reduce small-object false negatives (`helmet`, `gloves`, `no_helmet`) enough to justify the higher ONNX inference latency and model size?

---

## Priority 3 — `EXP-E` (Controlled Small-Object & Tuning Experiment) & `EXP-F` (`YOLOv8n`)

**Targeted Limitation**: `no_helmet` low recall (`0.5172`) & missed tiny PPE objects, plus architecture comparison.

- **Experiment 4 (`EXP-E` — Controlled Tuning on Best Merged Candidate)**: Using the winning `EXP-C`/`EXP-D` architecture on `merged.yaml`, run a single controlled comparison testing:
  1. Higher input resolution or controlled `copy_paste` / `scale` / `mosaic` tuning (`controlled_tuned` preset, **one factor group at a time**, keeping `seed=42`) — *only* to test if small-object (`gloves`, `no_helmet`) recall improves on **validation**.
  2. Validation-only confidence threshold sweep (`conf` in `[0.15, 0.20, 0.25, 0.30]`) on `val_merged` to measure whether `no_helmet` recall can be lifted cleanly without unacceptable false-positive growth before locking final thresholds.
- **Experiment 5 (`EXP-F`)**: Train `YOLOv8n` on `data/processed/merged.yaml` (`50` epochs, `imgsz=640`, `batch=16`, `seed=42`, `baseline` augmentation) to complete the cross-generation architecture comparison.

---

## Explicit Non-Goals (Deferred / Rejected for Now)

- **Do NOT** stack heavy color/geometric augmentations (`flipud`, large `degrees`, `erasing`, `mixup`) all at once — `EXP-A` showed zero overfitting on validation, so aggressive regularization would slow convergence without solving domain shift or small-object resolution.
- **Do NOT** modify `configs/class_mapping.yaml` or `data/processed/` mid-matrix (`EXP-A`–`EXP-F` must share the exact locked canonical dataset for fair comparison).

---

## Recommended Immediate Next Step

Approve running **`EXP-B`** followed by **`EXP-C`** (`YOLO11n`, `50` epochs each) so we can evaluate the central multi-source fusion hypothesis against `EXP-A` before deciding on `EXP-D` / `EXP-E`.
