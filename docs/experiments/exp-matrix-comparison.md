# SafeGuard Full Experiment Matrix Comparison (`EXP-A`–`EXP-F`)

**Date**: 2026-11-26  
**Canonical Taxonomy**: 5 classes (`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`) from `configs/class_mapping.yaml`  
**Shared Evaluation Protocol**: `imgsz=640`, `conf=0.25`, `iou=0.50`, `seed=42`, `AdamW`, evaluated on locked source-specific test sets:
- `roboflow_test` (`data/processed/test_roboflow.yaml`): `82` images, `426` ground-truth boxes
- `ultralytics_test` (`data/processed/test_ultralytics.yaml`): `120` images, `978` ground-truth boxes

---

## 1. Training & Validation Summary

All six runs trained on `NVIDIA GeForce RTX 5070 Laptop GPU` (`imgsz=640`, `seed=42`, tracked in `sqlite:///mlflow.db`). `EXP-A`–`EXP-D` and `EXP-F` used `50` epochs, `lr0=0.002`, and `baseline` augmentation; `EXP-E` used `60` epochs, `lr0=0.0015`, and `controlled_tuned` augmentation (`hsv_s=0.6`, `degrees=5.0`, `mosaic=0.8`, `mixup=0.1`).

| Run | Architecture | Data Source (Train / Val) | Epochs / Batch | MLflow Run ID | Duration (s) | Peak GPU Alloc / Res (MB) | Val P | Val R | Val mAP@50 | Val mAP@50:95 |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `EXP-A` | `YOLO11n` | `ultralytics_only` (`679` / `200`) | 50 / 16 | `fedb2987682e438982dd7a0cbdacb676` | 568.41 | 2,306.26 / 2,994.0 | 0.9223 | 0.7640 | 0.8316 | 0.6256 |
| `EXP-B` | `YOLO11n` | `roboflow_only` (`452` / `107`) | 50 / 16 | `85f96581e4ca4f75b89da422f2bbc447` | 402.14 | 2,467.76 / 3,088.0 | 0.7110 | 0.4733 | 0.5404 | 0.3324 |
| `EXP-C` | `YOLO11n` | `merged` (`1,131` / `307`) | 50 / 16 | `c732b372be954446acad5c30e007c20c` | 793.18 | 2,369.49 / 3,126.0 | **0.9116** | 0.7276 | 0.8059 | 0.5914 |
| `EXP-D` | `YOLO11s` | `merged` (`1,131` / `307`) | 50 / 12 | `a5ed0c1d3b4c4a0288c627fddc56f686` | 1,191.36 | 3,150.65 / 4,560.0 | 0.8959 | 0.7472 | 0.8142 | **0.6154** |
| **`EXP-E`** | **`YOLO11s`** | `merged` (`1,131` / `307`) | **60 / 12** | `8209a2eee1c4468da88f6616ab700138` | 1,129.34 | 3,129.35 / 4,638.0 | 0.9018 | **0.7564** | **0.8191** | 0.6136 |
| `EXP-F` | `YOLOv8n` | `merged` (`1,131` / `307`) | 50 / 16 | `42f61ec04a0d41f784068c0a25097645` | 691.35 | 2,089.38 / 2,586.0 | 0.8932 | 0.7258 | 0.7977 | 0.5911 |

> Bolded `merged`-dataset scores show `EXP-E` leads on `Val R` (`0.7564`) and `Val mAP@50` (`0.8191`), with `EXP-E` [results.png](../../runs/detect/runs/train/EXP-E_seed42/results.png) showing monotonic validation loss reduction through epoch 60 without overfitting.

Selected candidate checkpoints (gitignored binaries, DVC/MLflow-tracked):
- `EXP-D`: `models/candidates/EXP-D/EXP-D_seed42_best.pt` (`sha256: e7b3c83e57f79d697dd84775036533a540e88b3a508ed247b210b8a196ade871`)
- `EXP-E`: `models/candidates/EXP-E/EXP-E_seed42_best.pt` (`sha256: f867ce81ab2374f3f25fed97766c0898f4776ce0a4dcfab071411c08897bd091`)
- `EXP-F`: `models/candidates/EXP-F/EXP-F_seed42_best.pt` (`sha256: c81729bf8af9d980562920fb0d169be8ba9d12ca58733c78070b7e9d318bf57c`)

---

## 2. Validation-Only Confidence Threshold Sweep (`EXP-D` vs. `EXP-E` on `val_merged`)

Executed strictly on `val` split (`307` images, `2,033` instances; locked test sets untouched) in [val_conf_sweep.json](../../reports/experiments/val_conf_sweep.json). Ultralytics reports `P`/`R` at the max-F1 point along the retained PR curve above `conf`.

| Model | `conf` | Precision | Recall | F1 | mAP@50 | mAP@50:95 | `no_helmet` P / R / mAP@50 |
|---|---:|---:|---:|---:|---:|---:|---|
| `EXP-D` | 0.15 | 0.8948 | 0.7527 | 0.8176 | 0.7795 | 0.5963 | 0.8114 / 0.5391 / 0.5657 |
| `EXP-D` | 0.20 | 0.8948 | 0.7527 | 0.8176 | 0.7663 | 0.5917 | 0.8114 / 0.5391 / 0.5514 |
| `EXP-D` | 0.25 | 0.8948 | 0.7527 | 0.8176 | 0.7596 | 0.5881 | 0.8114 / 0.5391 / 0.5437 |
| `EXP-D` | 0.30 | 0.9067 | 0.7476 | 0.8195 | 0.7424 | 0.5812 | 0.8316 / 0.5282 / 0.5117 |
| **`EXP-E`** | 0.15 | **0.9171** | 0.7557 | 0.8286 | **0.7796** | 0.5930 | **0.8270** / 0.5432 / **0.5673** |
| **`EXP-E`** | 0.20 | **0.9171** | 0.7557 | 0.8286 | 0.7717 | 0.5897 | **0.8270** / 0.5432 / 0.5528 |
| **`EXP-E`** | 0.25 | **0.9171** | 0.7557 | 0.8286 | 0.7613 | 0.5852 | **0.8270** / 0.5432 / 0.5450 |
| **`EXP-E`** | 0.30 | 0.9047 | **0.7650** | **0.8290** | 0.7562 | 0.5838 | 0.8148 / **0.5556** / 0.5290 |

- `EXP-E` outperforms `EXP-D` in F1 at every swept validation confidence threshold (`0.8286–0.8290` vs. `0.8176–0.8195`).
- The max-F1 operating region sits cleanly in the `[0.25, 0.30]` range, supporting `conf=0.25` as the default inference threshold while reserving lower-confidence outputs (`0.15–0.25`) strictly for uncertain/review UI states if enabled later.

---

## 3. Locked Source-Specific Test Results (`conf=0.25`, `iou=0.50`)

### 3.1 `roboflow_test` (`82` images, `426` instances)

| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP | FP | FN | Small-Object FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`YOLO11n`, UL) | 0.5776 | 0.3919 | 0.4669 | 0.3526 | 0.2045 | 199 | 85 | 227 | — |
| `EXP-B` (`YOLO11n`, RF) | 0.6642 | 0.4868 | 0.5618 | 0.4479 | 0.2834 | 245 | 74 | 181 | — |
| `EXP-C` (`YOLO11n`, Merged) | 0.6676 | 0.5618 | 0.6101 | 0.5089 | 0.3231 | 264 | 90 | 162 | 128 |
| `EXP-D` (`YOLO11s`, Merged) | **0.8114** | 0.5486 | 0.6546 | 0.5338 | 0.3426 | 264 | 73 | 162 | 128 |
| **`EXP-E` (`YOLO11s` Tuned, Merged)** | 0.7544 | **0.5998** | **0.6683** | **0.5610** | **0.3498** | **274** | **59** | **152** | **120** |
| `EXP-F` (`YOLOv8n`, Merged) | 0.7526 | 0.5349 | 0.6254 | 0.5170 | 0.3209 | 259 | 85 | 167 | 133 |

#### Per-Class Breakdown on `roboflow_test` (`EXP-C` vs. `EXP-D` vs. `EXP-E` vs. `EXP-F`)

| Class | `EXP-C` (`YOLO11n`) P / R / mAP50 | `EXP-D` (`YOLO11s`) P / R / mAP50 | **`EXP-E` (`YOLO11s` Tuned) P / R / mAP50 / mAP50-95** | `EXP-F` (`YOLOv8n`) P / R / mAP50 |
|---|---|---|---|---|
| `person` | 0.8333 / 0.6322 / 0.6198 | **0.8930** / 0.6234 / 0.6418 | 0.8112 / **0.6667** / **0.6537** / 0.4693 | 0.8426 / 0.6152 / 0.6298 |
| `helmet` | 0.8478 / **0.7091** / **0.6948** | **1.0000** / 0.6675 / 0.6845 | 0.9625 / 0.7000 / 0.6936 / **0.4702** | 0.9649 / 0.6545 / 0.6508 |
| `vest` | 0.7167 / 0.7049 / 0.6740 | 0.7811 / 0.6885 / 0.6573 | 0.8148 / **0.7213** / **0.6779** / **0.3943** | **0.8189** / 0.6670 / 0.6635 |
| `gloves` | 0.4400 / 0.2750 / 0.1798 | **0.7463** / 0.3000 / 0.2982 | 0.6087 / **0.3500** / **0.3094** / 0.1823 | 0.6037 / 0.2500 / 0.2146 |
| `no_helmet` | 0.5000 / 0.4878 / 0.3763 | **0.6366** / 0.4634 / 0.3873 | 0.5750 / **0.5610** / **0.4706** / **0.2331** | 0.5331 / 0.4878 / 0.4262 |

Per-class FP / FN at `conf=0.25` (`roboflow_test`) for `EXP-E`:
- FP `{person: 25, helmet: 1, vest: 11, gloves: 9, no_helmet: 13}` (`59` total vs. `73` in `EXP-D`, `85` in `EXP-F`, `90` in `EXP-C`).
- FN `{person: 60, helmet: 32, vest: 16, gloves: 25, no_helmet: 19}` (`152` total vs. `162` in `EXP-D`, with small-object FN reduced from `128` to `120`).

---

### 3.2 `ultralytics_test` (`120` images, `978` instances)

| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP | FP | FN | Small-Object FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`YOLO11n`, UL) | **0.9358** | 0.8031 | 0.8644 | 0.8106 | 0.6321 | 841 | 84 | 137 | — |
| `EXP-B` (`YOLO11n`, RF) | 0.6954 | 0.5208 | 0.5956 | 0.5032 | 0.3296 | 589 | 131 | 389 | — |
| `EXP-C` (`YOLO11n`, Merged) | 0.9172 | 0.8231 | 0.8676 | 0.8097 | 0.6271 | 842 | 62 | 136 | 133 |
| `EXP-D` (`YOLO11s`, Merged) | 0.9334 | 0.8230 | 0.8747 | 0.8211 | **0.6574** | 853 | 68 | 125 | 121 |
| **`EXP-E` (`YOLO11s` Tuned, Merged)** | 0.9285 | **0.8423** | **0.8833** | **0.8335** | 0.6421 | **869** | 65 | **109** | **107** |
| `EXP-F` (`YOLOv8n`, Merged) | 0.9211 | 0.8244 | 0.8701 | 0.8111 | 0.6305 | 846 | **59** | 132 | 129 |

#### Per-Class Breakdown on `ultralytics_test` (`EXP-C` vs. `EXP-D` vs. `EXP-E` vs. `EXP-F`)

| Class | `EXP-C` (`YOLO11n`) P / R / mAP50 | `EXP-D` (`YOLO11s`) P / R / mAP50 | **`EXP-E` (`YOLO11s` Tuned) P / R / mAP50 / mAP50-95** | `EXP-F` (`YOLOv8n`) P / R / mAP50 |
|---|---|---|---|---|
| `person` | 0.9526 / **0.9596** / 0.9545 | **0.9727** / 0.9559 / 0.9548 | 0.9669 / **0.9596** / **0.9638** / 0.8076 | 0.9520 / 0.9485 / 0.9445 |
| `helmet` | 0.9512 / 0.8387 / 0.8317 | **0.9571** / 0.8441 / 0.8406 | 0.9214 / **0.8656** / **0.8572** / 0.6823 | 0.9157 / 0.8172 / 0.8097 |
| `vest` | 0.9528 / 0.9182 / 0.9110 | 0.9623 / 0.9227 / 0.9306 | 0.9536 / **0.9500** / **0.9386** / 0.7667 | **0.9716** / 0.9318 / 0.9290 |
| `gloves` | 0.8990 / 0.8357 / 0.8286 | 0.9236 / 0.8404 / 0.8394 | **0.9388** / **0.8638** / **0.8612** / 0.6663 | 0.9188 / 0.8498 / 0.8388 |
| `no_helmet` | 0.8305 / 0.5632 / 0.5228 | 0.8516 / 0.5517 / 0.5403 | **0.8616** / 0.5725 / **0.5467** / 0.2878 | 0.8475 / **0.5747** / 0.5335 |

Per-class FP / FN at `conf=0.25` (`ultralytics_test`) for `EXP-E`:
- FP `{person: 11, helmet: 16, vest: 11, gloves: 15, no_helmet: 12}` (`65` total vs. `68` in `EXP-D`).
- FN `{person: 10, helmet: 25, vest: 9, gloves: 29, no_helmet: 36}` (`109` total vs. `125` in `EXP-D`, with small-object FN reduced from `121` to `107`).

---

## 4. Matrix Conclusions & Candidate Selection

1. **`EXP-E` (`YOLO11s`, `controlled_tuned`, `60` epochs, `lr0=0.0015`) is the unambiguous overall winner**:
   - Improves `roboflow_test` `mAP@50` from `0.5338` (`EXP-D`) to **`0.5610`** (`+2.72` pts; `+20.84` pts over `EXP-A` and `+11.31` pts over `EXP-B`), lifts `roboflow_test` F1 to **`0.6683`**, cuts total false positives (`73 → 59`, only `1` helmet FP), and boosts `no_helmet` `mAP@50` by **`+8.33` points** (`0.3873 → 0.4706`) with recall jumping from `0.4634` to `0.5610`.
   - Improves `ultralytics_test` `mAP@50` from `0.8211` (`EXP-D`) to **`0.8335`**, lifts F1 from `0.8747` to **`0.8833`**, increases true positives (`853 → 869`), reduces missed detections (`125 → 109`), reduces small-object misses (`121 → 107`), and achieves the highest `mAP@50` on **every single class** (`person 0.9638`, `helmet 0.8572`, `vest 0.9386`, `gloves 0.8612`, `no_helmet 0.5467`).
   - Only slight trade-off: `ultralytics_test` `mAP@50:95` is `0.6421` vs. `EXP-D`'s `0.6574` (`no_helmet` box tightness `0.2878` vs. `0.3225`), attributable to mild `mixup=0.1` + `degrees=5.0`, while `roboflow_test` `mAP@50:95` improved (`0.3426 → 0.3498`).
2. **All Three Hypotheses Confirmed Across `EXP-A`–`EXP-F`**:
   - **Multi-source fusion (`EXP-C` vs. `EXP-A`/`EXP-B`)**: Essential for cross-domain generalization (`roboflow_test` `0.3526`/`0.4479 → 0.5089`).
   - **Model scaling (`EXP-D` `YOLO11s` vs. `EXP-C` `YOLO11n` / `EXP-F` `YOLOv8n`)**: Higher precision and better small-object `gloves` & localization.
   - **Controlled tuning (`EXP-E` vs. `EXP-D`)**: Longer schedule + lower initial LR + mild controlled augmentations deliver substantial gains on small PPE and `no_helmet` recall without overfitting.

---

## 5. Recommended Next Step (ONNX Export, Verification & Production Candidate Registration)

1. Export `EXP-E` (`models/candidates/EXP-E/EXP-E_seed42_best.pt`) to ONNX (`models/production/safeguard_yolo11s.onnx`) with opset `17` and fixed `640×640` input shape.
2. Verify PyTorch-vs-ONNX Runtime numerical parity (class IDs, confidences, and bounding boxes within documented tolerances) and benchmark CPU + CUDA ONNX Runtime mean/P95 latency, FPS, memory, and artifact size via `src/models/benchmark.py`.
3. Register the verified `EXP-E` model candidate in MLflow Model Registry (`Production` / `Staging`), wire the FastAPI ONNX Runtime inference service (`backend/`), and verify end-to-end tests before proceeding to compliance/tracking and frontend integration.
