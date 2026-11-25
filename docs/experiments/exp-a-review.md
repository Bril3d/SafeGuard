# EXP-A Baseline Review — Ultralytics Source Only (`YOLO11n`)

**Date**: 2026-11-25  
**MLflow Run ID**: `fedb2987682e438982dd7a0cbdacb676` (`sqlite:///mlflow.db`, Experiment: `SafeGuard-PPE-Detection`)  
**Checkpoint**: `models/candidates/EXP-A/EXP-A_seed42_best.pt` (`sha256`: `6d2611c077d3e96c8612e5966615d23adc85e92f1ea5dcbaf9f2dc0e77ecedba`)  
**Git Revision at Launch**: `6dc541b`

---

## 1. Preflight Verification Summary

All six preflight checks passed without modifying the dataset:

1. **Dataset Config**: `data/processed/ultralytics_only.yaml` verified (`nc=5`, `{0: person, 1: helmet, 2: vest, 3: gloves, 4: no_helmet}`).
2. **Split & Annotation Integrity**:
   - `train_ultralytics`: `679` images, `679` labels, `5,920` valid boxes (`0` corrupt/invalid)
   - `val_ultralytics`: `200` images, `200` labels, `1,675` valid boxes (`0` corrupt/invalid)
   - `test_ultralytics`: `120` images, `120` labels, `978` valid boxes (`0` corrupt/invalid)
3. **Class IDs**: Every box in all three splits matched `0..4`.
4. **Leakage Check**: `0` SHA-256 overlap and `0` scene-group overlap across `train_ultralytics`, `val_ultralytics`, and `test_ultralytics`.
5. **DVC Revision**: `dvc.lock` SHA-256 `7ae1205d7ce5bf2d6219b2de06394053dd084a8b86fb72a35912d7a0547e0b13`, `split_manifest.json` SHA-256 `743f497da198edf23926a8c86fdefb097109ff631dbeb21322e97eb58d0a7962` (note: `dvc status` only flagged Ultralytics' `labels.cache` files generated in `test_*` during prior `SMOKE` evaluation).
6. **Configuration & Software Versions**:
   - Model: `YOLO11n` (`yolo11n.pt`, pretrained COCO weights; `2,590,815` train params)
   - Hyperparameters: `epochs=50`, `batch_size=16`, `imgsz=640`, `optimizer=AdamW`, `lr0=0.002`, `lrf=0.01`, `weight_decay=0.0005`, `seed=42`, `amp=True`, `patience=15`, `baseline` augmentation preset (`mosaic=1.0`, `close_mosaic=10`, `mixup=0.0`, `fliplr=0.5`)
   - Runtime: Python `3.12.10`, PyTorch `2.11.0+cu128`, CUDA `12.8`, GPU `NVIDIA GeForce RTX 5070 Laptop GPU` (`8151 MiB`), Ultralytics `8.4.218`, MLflow `3.18.0`, DVC `3.67.1`

---

## 2. Training Dynamics & Overfitting Analysis

- **Duration & GPU Memory**: `568.41 s` (~9.5 min); peak GPU memory `2,306.26 MB` allocated / `2,994.0 MB` reserved.
- **Loss Progression (`runs/detect/runs/train/EXP-A_seed42/results.csv`)**:
  - `train/box_loss`: `1.20345` (ep 1) → `0.74693` (ep 40) → `0.62737` (ep 50)
  - `train/cls_loss`: `1.57272` (ep 1) → `0.47994` (ep 40) → `0.39464` (ep 50)
  - `train/dfl_loss`: `1.20402` (ep 1) → `0.94549` (ep 40) → `0.88762` (ep 50)
  - `val/box_loss`: `1.51707` (ep 1) → `0.75459` (ep 40) → `0.70473` (ep 50)
  - `val/cls_loss`: `18.0302` (ep 1) → `0.49306` (ep 40) → `0.45261` (ep 50)
  - `val/dfl_loss`: `1.38377` (ep 1) → `0.94885` (ep 40) → `0.92788` (ep 50)
- **Validation Metrics (Best Checkpoint)**: `val_precision = 0.9223`, `val_recall = 0.7640`, `val_mAP50 = 0.8316`, `val_mAP50-95 = 0.6256`.
- **Overfitting Verdict**: **No overfitting** is observed. All three validation losses reach their lowest values at Epoch 50 alongside steady `val mAP50-95` gains (`0.6247` at ep 50), and in-domain test performance (`mAP50=0.8106`, `mAP50-95=0.6321`) matches validation closely.

---

## 3. Locked Test-Set Results (`conf=0.25`, `iou=0.50`)

### 3.1 `ultralytics_test` (In-Domain Holdout: 120 images, 978 boxes)
| Class | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| **All** | **0.9358** | **0.8031** | **0.8644** | **0.8106** | **0.6321** |
| `person` | 0.9677 | 0.9485 | 0.9580 | 0.9545 | 0.8209 |
| `helmet` | 0.9386 | 0.8225 | 0.8768 | 0.8229 | 0.6622 |
| `vest` | 0.9576 | 0.9248 | 0.9409 | 0.9354 | 0.7520 |
| `gloves` | 0.9395 | 0.8025 | 0.8656 | 0.8188 | 0.6241 |
| `no_helmet` | 0.8752 | **0.5172** | 0.6502 | **0.5215** | **0.3013** |

### 3.2 `roboflow_test` (Cross-Source Locked Test: 82 images, 426 boxes)
| Class | Precision | Recall | F1 | mAP@50 | mAP@50:95 |
|---|---:|---:|---:|---:|---:|
| **All** | **0.5776** | **0.3919** | **0.4669** | **0.3526** | **0.2045** |
| `person` | 0.8198 | 0.5230 | 0.6386 | 0.5072 | 0.3264 |
| `helmet` | 0.8889 | 0.5818 | 0.7033 | 0.5636 | 0.3659 |
| `vest` | 0.6596 | 0.5082 | 0.5741 | 0.4686 | 0.2308 |
| `gloves` | 0.3478 | 0.2000 | 0.2540 | 0.1421 | 0.0808 |
| `no_helmet` | 0.1716 | 0.1463 | 0.1580 | 0.0813 | 0.0188 |

---

## 4. Confusion Matrix, Error Analysis & Small-Object Findings

| Split (`error_analysis_*.json`) | TP | FP | FN | Small-Object FN (`area_norm < 0.01`) | % of FN that are Small |
|---|---:|---:|---:|---:|---:|
| `val_ultralytics` (200 imgs) | 1,364 | 151 | 311 | 297 | **95.5%** |
| `test_ultralytics` (120 imgs) | 841 | 84 | 137 | 134 | **97.8%** |
| `test_roboflow` (82 imgs) | 199 | 85 | 227 | 163 | **71.8%** |

- **Confusion Matrix (`ultralytics_test`)**: Inter-class confusion is minimal (`0.06` of true `no_helmet` confused as `helmet`, `0.01` of `helmet` as `no_helmet`). The primary failure mode is **missed detections into background**: `39%` of `no_helmet` (`40` FN), `18%` of `gloves` (`38` FN), `17%` of `helmet` (`33` FN).
- **Confusion Matrix (`roboflow_test`)**: Background misses dominate every class (`83%` `no_helmet`, `80%` `gloves`, `49%` `vest`, `48%` `person`, `36%` `helmet`), alongside elevated `no_helmet` background FP (`27` FP, `P=0.1716`).
- **Class Imbalance**: In `train_ultralytics`, `no_helmet` has only `547` instances (`289` images) vs. `1,689` `person`, `1,333` `gloves`, `1,256` `vest`, and `1,095` `helmet`, while also having the smallest median normalized box area (`0.00068`).

---

## 5. Three Most Important Limitations & Baseline Verdict

1. **Cross-Source Domain & Annotation Gap**: `EXP-A` drops from `0.8106` mAP50 (`ultralytics_test`) to `0.3526` (`roboflow_test`), with `no_helmet` collapsing to `0.0813` mAP50 (`0.0188` mAP50:95) due to real-world visual domain shift and the previously audited `NO-Hardhat` (hairline strip) vs `no_helmet` (full head) box-extent mismatch.
2. **Low `no_helmet` Recall (`0.5172` even in-domain)**: Combined class imbalance (`547` instances) and tiny head sizes (`median area 0.00068`) cause `39%` false negatives into background, directly limiting safety-alert reliability.
3. **Small-Object Recall Ceiling (`97.8%` of `ultralytics_test` FN have `area_norm < 0.01`)**: `YOLO11n` at `640x640` misses distant helmets, gloves, and unhelmeted heads (`134/137` FN).

**Baseline Reliability Verdict**: `EXP-A` is a **reliable, clean, leakage-free Source-B baseline** (no overfitting, strong convergence on `person`/`vest`/`helmet`/`gloves`), but it is **insufficient as a final model** due to weak cross-domain generalization and low `no_helmet` recall.
