# SafeGuard Experiment Matrix Comparison (`EXP-A`, `EXP-B`, `EXP-C`, `EXP-D`, `EXP-F`)

**Date**: 2026-11-26  
**Canonical Taxonomy**: 5 classes (`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`) from `configs/class_mapping.yaml`  
**Shared Evaluation Protocol**: `imgsz=640`, `conf=0.25`, `iou=0.50`, `seed=42`, `AdamW` (`lr0=0.002`), `baseline` augmentation, evaluated on locked source-specific test sets:
- `roboflow_test` (`data/processed/test_roboflow.yaml`): `82` images, `426` ground-truth boxes
- `ultralytics_test` (`data/processed/test_ultralytics.yaml`): `120` images, `978` ground-truth boxes

---

## 1. Training & Validation Summary

All five runs trained on `NVIDIA GeForce RTX 5070 Laptop GPU` (`50` epochs, `imgsz=640`, `seed=42`, tracked in `sqlite:///mlflow.db`).

| Run | Architecture | Data Source (Train / Val) | Batch | MLflow Run ID | Duration (s) | Peak GPU Alloc / Res (MB) | Val P | Val R | Val mAP50 | Val mAP50-95 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `EXP-A` | `YOLO11n` | `ultralytics_only` (`679` / `200`) | 16 | `fedb2987682e438982dd7a0cbdacb676` | 568.41 | 2,306.26 / 2,994.0 | 0.9223 | 0.7640 | 0.8316 | 0.6256 |
| `EXP-B` | `YOLO11n` | `roboflow_only` (`452` / `107`) | 16 | `85f96581e4ca4f75b89da422f2bbc447` | 402.14 | 2,467.76 / 3,088.0 | 0.7110 | 0.4733 | 0.5404 | 0.3324 |
| `EXP-C` | `YOLO11n` | `merged` (`1,131` / `307`) | 16 | `c732b372be954446acad5c30e007c20c` | 793.18 | 2,369.49 / 3,126.0 | **0.9116** | 0.7276 | 0.8059 | 0.5914 |
| **`EXP-D`** | **`YOLO11s`** | `merged` (`1,131` / `307`) | 12 | `a5ed0c1d3b4c4a0288c627fddc56f686` | 1,191.36 | 3,150.65 / 4,560.0 | 0.8959 | **0.7472** | **0.8142** | **0.6154** |
| `EXP-F` | `YOLOv8n` | `merged` (`1,131` / `307`) | 16 | `42f61ec04a0d41f784068c0a25097645` | 691.35 | 2,089.38 / 2,586.0 | 0.8932 | 0.7258 | 0.7977 | 0.5911 |

> Bolded validation scores compare the three `merged`-dataset models (`EXP-C`, `EXP-D`, `EXP-F`). Both `EXP-D` (`runs/detect/runs/train/EXP-D_seed42/results.png`) and `EXP-F` (`runs/detect/runs/train/EXP-F_seed42/results.png`) show monotonically decreasing validation `box_loss`, `cls_loss`, and `dfl_loss` through epoch 50 without U-shaped overfitting.

Checkpoints saved (gitignored binaries, DVC/MLflow-tracked):
- `EXP-D`: `models/candidates/EXP-D/EXP-D_seed42_best.pt` (SHA-256: `e7b3c83e57f79d697dd84775036533a540e88b3a508ed247b210b8a196ade871`)
- `EXP-F`: `models/candidates/EXP-F/EXP-F_seed42_best.pt` (SHA-256: `c81729bf8af9d980562920fb0d169be8ba9d12ca58733c78070b7e9d318bf57c`)

---

## 2. Locked Source-Specific Test Results

### 2.1 `roboflow_test` (`82` images, `426` instances)

| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP | FP | FN | Small-Object FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`YOLO11n`, UL) | 0.5776 | 0.3919 | 0.4669 | 0.3526 | 0.2045 | 199 | 85 | 227 | — |
| `EXP-B` (`YOLO11n`, RF) | 0.6642 | 0.4868 | 0.5618 | 0.4479 | 0.2834 | 245 | 74 | 181 | — |
| `EXP-C` (`YOLO11n`, Merged) | 0.6676 | **0.5618** | 0.6101 | 0.5089 | 0.3231 | **264** | 90 | **162** | **128** |
| **`EXP-D` (`YOLO11s`, Merged)** | **0.8114** | 0.5486 | **0.6546** | **0.5338** | **0.3426** | **264** | **73** | **162** | **128** |
| `EXP-F` (`YOLOv8n`, Merged) | 0.7526 | 0.5349 | 0.6254 | 0.5170 | 0.3209 | 259 | 85 | 167 | 133 |

#### Per-Class Breakdown on `roboflow_test` (`EXP-C` vs. `EXP-D` vs. `EXP-F`)

| Class | `EXP-C` (`YOLO11n`) P / R / mAP50 / mAP50-95 | `EXP-D` (`YOLO11s`) P / R / mAP50 / mAP50-95 | `EXP-F` (`YOLOv8n`) P / R / mAP50 / mAP50-95 |
|---|---|---|---|
| `person` | 0.8333 / **0.6322** / 0.6198 / 0.4552 | **0.8930** / 0.6234 / **0.6418** / **0.4749** | 0.8426 / 0.6152 / 0.6298 / 0.4721 |
| `helmet` | 0.8478 / **0.7091** / **0.6948** / **0.4555** | **1.0000** / 0.6675 / 0.6845 / 0.4434 | 0.9649 / 0.6545 / 0.6508 / 0.4195 |
| `vest` | 0.7167 / **0.7049** / **0.6740** / **0.3880** | 0.7811 / 0.6885 / 0.6573 / 0.3768 | **0.8189** / 0.6670 / 0.6635 / 0.3816 |
| `gloves` | 0.4400 / 0.2750 / 0.1798 / 0.1142 | **0.7463** / **0.3000** / **0.2982** / **0.2067** | 0.6037 / 0.2500 / 0.2146 / 0.1308 |
| `no_helmet` | 0.5000 / **0.4878** / 0.3763 / 0.2024 | **0.6366** / 0.4634 / 0.3873 / **0.2111** | 0.5331 / **0.4878** / **0.4262** / 0.2004 |

Per-class FP / FN counts at `conf=0.25` (`roboflow_test`):
- `EXP-D`: FP `{person: 31, helmet: 5, vest: 15, gloves: 10, no_helmet: 12}`, FN `{person: 61, helmet: 33, vest: 20, gloves: 27, no_helmet: 21}` (`128/162` FN are small objects).
- `EXP-F`: FP `{person: 29, helmet: 5, vest: 14, gloves: 13, no_helmet: 24}`, FN `{person: 62, helmet: 37, vest: 18, gloves: 30, no_helmet: 20}` (`133/167` FN are small objects).

---

### 2.2 `ultralytics_test` (`120` images, `978` instances)

| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP | FP | FN | Small-Object FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`YOLO11n`, UL) | **0.9358** | 0.8031 | 0.8644 | 0.8106 | 0.6321 | 841 | 84 | 137 | — |
| `EXP-B` (`YOLO11n`, RF) | 0.6954 | 0.5208 | 0.5956 | 0.5032 | 0.3296 | 589 | 131 | 389 | — |
| `EXP-C` (`YOLO11n`, Merged) | 0.9172 | 0.8231 | 0.8676 | 0.8097 | 0.6271 | 842 | 62 | 136 | 133 |
| **`EXP-D` (`YOLO11s`, Merged)** | 0.9334 | 0.8230 | **0.8747** | **0.8211** | **0.6574** | **853** | 68 | **125** | **121** |
| `EXP-F` (`YOLOv8n`, Merged) | 0.9211 | **0.8244** | 0.8701 | 0.8111 | 0.6305 | 846 | **59** | 132 | 129 |

#### Per-Class Breakdown on `ultralytics_test` (`EXP-C` vs. `EXP-D` vs. `EXP-F`)

| Class | `EXP-C` (`YOLO11n`) P / R / mAP50 / mAP50-95 | `EXP-D` (`YOLO11s`) P / R / mAP50 / mAP50-95 | `EXP-F` (`YOLOv8n`) P / R / mAP50 / mAP50-95 |
|---|---|---|---|
| `person` | 0.9526 / **0.9596** / 0.9545 / 0.8095 | **0.9727** / 0.9559 / **0.9548** / **0.8344** | 0.9520 / 0.9485 / 0.9445 / 0.8022 |
| `helmet` | 0.9512 / 0.8387 / 0.8317 / 0.6587 | **0.9571** / **0.8441** / **0.8406** / **0.6882** | 0.9157 / 0.8172 / 0.8097 / 0.6604 |
| `vest` | 0.9528 / 0.9182 / 0.9110 / 0.7384 | 0.9623 / 0.9227 / **0.9306** / **0.7705** | **0.9716** / **0.9318** / 0.9290 / 0.7484 |
| `gloves` | 0.8990 / 0.8357 / 0.8286 / 0.6297 | **0.9236** / 0.8404 / **0.8394** / **0.6712** | 0.9188 / **0.8498** / 0.8388 / 0.6339 |
| `no_helmet` | 0.8305 / 0.5632 / 0.5228 / 0.2991 | **0.8516** / 0.5517 / **0.5403** / **0.3225** | 0.8475 / **0.5747** / 0.5335 / 0.3073 |

Per-class FP / FN counts at `conf=0.25` (`ultralytics_test`):
- `EXP-D`: FP `{person: 9, helmet: 12, vest: 11, gloves: 24, no_helmet: 12}`, FN `{person: 11, helmet: 29, vest: 15, gloves: 33, no_helmet: 37}` (`121/125` FN are small objects).
- `EXP-F`: FP `{person: 10, helmet: 14, vest: 7, gloves: 16, no_helmet: 12}`, FN `{person: 13, helmet: 35, vest: 16, gloves: 30, no_helmet: 38}` (`129/132` FN are small objects).

---

## 3. Architecture & Capacity Findings

1. **`EXP-D` (`YOLO11s`) is the strongest candidate across every aggregate split** (`val_merged`, `roboflow_test`, and `ultralytics_test`):
   - Over `EXP-C` (`YOLO11n`), `YOLO11s` improves `roboflow_test` precision by `+0.1438` (`0.6676 → 0.8114`), F1 by `+0.0445` (`0.6101 → 0.6546`), `mAP@50` by `+0.0249` (`0.5089 → 0.5338`), and `mAP@50:95` by `+0.0195` (`0.3231 → 0.3426`), cutting false positives from `90` to `73`.
   - On `roboflow_test` `gloves`, `YOLO11s` lifts `mAP@50` from `0.1798` (`EXP-C`) / `0.2146` (`EXP-F`) to **`0.2982`** (`+11.84` points over `EXP-C`), with precision `0.4400 → 0.7463`.
   - On `ultralytics_test`, `YOLO11s` achieves the best F1 (`0.8747`), highest `mAP@50` (`0.8211`), highest `mAP@50:95` (`0.6574`, `+3.03` points over `EXP-C`), fewest total FN (`125` vs `136`), fewest small-object FN (`121` vs `133`), and ranks #1 on `mAP@50` and `mAP@50:95` for **all 5 classes**.
2. **`EXP-F` (`YOLOv8n`) vs. `EXP-C` (`YOLO11n`)**:
   - `YOLOv8n` (`3.13M` params, `8.6` GFLOPs) trains slightly faster (`691.35`s vs `793.18`s) and scores slightly higher on both test sets than `YOLO11n` (`0.5170` vs `0.5089` RF mAP50; `0.8111` vs `0.8097` UL mAP50), with slightly lower validation `mAP@50` (`0.7977` vs `0.8059`).
   - However, on `roboflow_test` `YOLOv8n` produces twice as many `no_helmet` false positives as `YOLO11s` (`24` vs `12`), lowering `no_helmet` precision (`0.5331` vs `0.6366`). In a safety decision-support UI where false missing-PPE alarms erode operator trust, `YOLO11s`'s higher precision is a decisive advantage.
3. **Remaining Bottlenecks for `EXP-E`**:
   - **Underfitting at epoch 50**: Both `EXP-D` and `EXP-F` still show downward validation loss curves at epoch 50.
   - **Small-object recall**: `128/162` (`79.0%`) of `EXP-D` misses on `roboflow_test` and `121/125` (`96.8%`) on `ultralytics_test` are small objects (`area_norm < 0.01`).
   - **`no_helmet` recall (`0.4634` RF / `0.5517` UL)** vs. high `no_helmet` precision (`0.6366` RF / `0.8516` UL) suggests room for controlled longer/tuned training plus validation-only confidence threshold calibration.

---

## 4. Recommended Next Step (`EXP-E` — Controlled Tuning on `YOLO11s`)

Run `EXP-E` using the already configured `configs/training.yaml` candidate (`YOLO11s`, `merged.yaml`, `epochs=60`, `batch_size=12`, `imgsz=640`, `lr0=0.0015`, `seed=42`, `controlled_tuned` preset: `hsv_s=0.6`, `degrees=5.0`, `mosaic=0.8`, `mixup=0.1`), followed by:
1. Locked test evaluation (`evaluate.py`) and `error_analysis.py` comparing `EXP-E` directly against `EXP-D`.
2. Validation-only confidence threshold sweep (`val_merged`) to verify whether adjusting `conf` (or per-class safety trigger guidance) improves `no_helmet` recall without excessive false alarms.
3. Selecting the final winner (`EXP-D` vs `EXP-E`) for `export_onnx.py` + `benchmark.py` (CPU & GPU ONNX parity + latency/FPS/memory verification) and registering the winning artifact in MLflow.
