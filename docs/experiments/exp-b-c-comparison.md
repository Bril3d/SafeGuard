# Multi-Source Fusion Comparison: `EXP-A` vs. `EXP-B` vs. `EXP-C` (`YOLO11n`)

**Date**: 2026-11-25  
**Controlled Setup**: `YOLO11n` (`yolo11n.pt`), `50` epochs, `batch_size=16`, `imgsz=640`, `AdamW` (`lr0=0.002`), `seed=42`, `baseline` augmentation, identical 5-class canonical taxonomy (`configs/class_mapping.yaml`), evaluated at `conf=0.25, iou=0.50` on locked `roboflow_test` (`82` images, `426` boxes) and `ultralytics_test` (`120` images, `978` boxes).

---

## 1. Training & Validation Summary

| Run | Data Source | MLflow Run ID | Duration (s) | Peak GPU Alloc / Res (MB) | Val P | Val R | Val mAP50 | Val mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **`EXP-A`** | `ultralytics_only` (`679` tr / `200` val) | `fedb2987682e438982dd7a0cbdacb676` | 568.41 | 2,306.26 / 2,994.0 | 0.9223 | 0.7640 | 0.8316 | 0.6256 |
| **`EXP-B`** | `roboflow_only` (`452` tr / `107` val) | `85f96581e4ca4f75b89da422f2bbc447` | 402.14 | 2,467.76 / 3,088.0 | 0.7110 | 0.4733 | 0.5404 | 0.3324 |
| **`EXP-C`** | `merged` (`1,131` tr / `307` val) | `c732b372be954446acad5c30e007c20c` | 793.18 | 2,369.49 / 3,126.0 | 0.9116 | 0.7276 | 0.8059 | 0.5914 |

---

## 2. Locked Source-Specific Test Comparison

### 2.1 `roboflow_test` (82 images, 426 instances)
| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`ultralytics_only`) | 0.5776 | 0.3919 | 0.4669 | 0.3526 | 0.2045 | 199 / 85 / 227 |
| `EXP-B` (`roboflow_only`) | 0.6642 | 0.4868 | 0.5618 | 0.4479 | 0.2834 | 245 / 74 / 181 |
| **`EXP-C` (`merged`)** | **0.6676** | **0.5618** | **0.6101** | **0.5089** | **0.3231** | **264 / 90 / 162** |

Per-class `mAP@50` on `roboflow_test`:
- `person`: `EXP-A` `0.5072` | `EXP-B` `0.6149` | **`EXP-C` `0.6198`**
- `helmet`: `EXP-A` `0.5636` | `EXP-B` `0.6325` | **`EXP-C` `0.6948`**
- `vest`: `EXP-A` `0.4686` | `EXP-B` `0.5094` | **`EXP-C` `0.6740`**
- `gloves`: `EXP-A` `0.1421` | `EXP-B` `0.1574` | **`EXP-C` `0.1798`**
- `no_helmet`: `EXP-A` `0.0813` | `EXP-B` `0.3253` | **`EXP-C` `0.3763`**

### 2.2 `ultralytics_test` (120 images, 978 instances)
| Run | Precision | Recall | F1 | mAP@50 | mAP@50:95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| `EXP-A` (`ultralytics_only`) | **0.9358** | 0.8031 | 0.8644 | **0.8106** | **0.6321** | 841 / 84 / 137 |
| `EXP-B` (`roboflow_only`) | 0.6954 | 0.5208 | 0.5956 | 0.5032 | 0.3296 | 589 / 131 / 389 |
| **`EXP-C` (`merged`)** | 0.9172 | **0.8231** | **0.8676** | 0.8097 | 0.6271 | **842 / 62 / 136** |

Per-class `mAP@50` on `ultralytics_test`:
- `person`: `EXP-A` `0.9545` | `EXP-B` `0.9103` | **`EXP-C` `0.9545`**
- `helmet`: `EXP-A` `0.8229` | `EXP-B` `0.7500` | **`EXP-C` `0.8317`**
- `vest`: **`EXP-A` `0.9354`** | `EXP-B` `0.7365` | `EXP-C` `0.9110`
- `gloves`: `EXP-A` `0.8188` | `EXP-B` `0.1106` | **`EXP-C` `0.8286`**
- `no_helmet`: `EXP-A` `0.5215` | `EXP-B` `0.0087` | **`EXP-C` `0.5228`** (Recall `0.5172` → **`0.5632`**, F1 `0.6502` → **`0.6712`**)

---

## 3. Research Conclusion (`EXP-A` vs. `EXP-B` vs. `EXP-C`)

The central SafeGuard research question is **empirically confirmed**:
1. **Multi-source fusion (`EXP-C`) outperforms both single-source baselines on `roboflow_test` across every metric and every class** (`mAP50`: `0.5089` vs. `0.4479` `EXP-B` and `0.3526` `EXP-A`; `mAP50-95`: `0.3231` vs. `0.2834` / `0.2045`; `F1`: `0.6101` vs. `0.5618` / `0.4669`).
2. **Simultaneously, `EXP-C` preserves `ultralytics_test` accuracy (`0.8097` vs. `0.8106` mAP50)** while improving overall recall (`0.8031 → 0.8231`), F1 (`0.8644 → 0.8676`), false positives (`84 → 62`), and `no_helmet` recall (`0.5172 → 0.5632`).
3. **Remaining Bottlenecks**: `no_helmet` recall (`0.5632` UL / `0.4878` RF) and small-object misses (`133/136` FN on `ultralytics_test`, `128/162` FN on `roboflow_test`) remain the primary targets for **`EXP-D` (`YOLO11s`)** and **`EXP-F` (`YOLOv8n`)** / **`EXP-E` (controlled tuning)**.
