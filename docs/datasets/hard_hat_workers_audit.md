# Hard Hat Workers Dataset Integration Audit

**Date**: 2026-11-26  
**Status**: Read-only audit complete (no existing datasets, splits, mappings, or `EXP-E` artifacts modified)  
**Evaluated Target**: Integration into SafeGuard's canonical 5-class unified detector (`EXP-E`, `YOLO11s`, `data/processed/merged.yaml`)  
**Decision**: **REJECT** (for direct merging into SafeGuard's 5-class dataset; optional non-merge transfer/auxiliary alternatives documented in Section 5)

---

## 1. Phase 1 — Source Provenance, Version & License

| Field | Observed Value |
|---|---|
| **Roboflow Universe Source** | `https://universe.roboflow.com/joseph-nelson/hard-hat-workers/dataset/10` (`public.roboflow.com/object-detection/hard-hat-workers/`) |
| **Original Creator** | Northeastern University – China (Harvard Dataverse DOI: `10.7910/DVN/7CBGOS`) |
| **Downloaded Version** | `v10` (`raw_AllClasses`, exported `June 8, 2023 at 10:31 AM GMT`, YOLOv8 format) |
| **License** | **Public Domain** (permissive; compatible with SafeGuard CC BY 4.0 sources) |
| **Preprocessing / Augmentations** | Auto-orientation with EXIF stripping only; **no augmentations applied** |
| **Local Raw Storage** | `data/raw/hard_hat_workers/` (preserved read-only; gitignored via `data/raw/*`) |
| **Declared Classes (`nc: 3`)** | `0: head`, `1: helmet`, `2: person` (`data/raw/hard_hat_workers/data.yaml`) |

---

## 2. Phase 2 — Dataset Inspection & Integrity

Executed across all `7,035` images and `7,035` label files using SafeGuard's `src/data/inspect_dataset.py` and `src/data/validate_annotations.py` helpers (recorded verbatim in [audit_metrics.json](../../reports/dataset/hard_hat_workers/audit_metrics.json)).

### 2.1 Counts & Integrity Checks

| Metric | Value |
|---|---:|
| Total images | `7,035` (`train`: `4,916`, `valid`: `1,413`, `test`: `706`) |
| Total label files | `7,035` |
| Total bounding-box annotations | `27,039` |
| Missing labels / Orphan labels | `0` / `0` |
| Empty label files | `0` |
| Corrupted / unreadable images | `0` |
| Invalid bounding boxes | `0` (`100%` valid YOLO coordinates within `[0, 1]`) |
| Image format & color mode | `JPEG`: `7,035` (`100%`), `RGB`: `7,035` (`100%`) |
| Image width (min / p10 / median / p90 / max) | `167` / `400` / `500` / `500` / `1,100` px |
| Image height (min / p10 / median / p90 / max) | `126` / `282` / `333` / `436` / `965` px |
| Most common resolutions | `500×333` (`1,365`), `500×375` (`635`), `500×334` (`320`), `500×332` (`303`), `450×300` (`200`) |

### 2.2 Class Distribution & Split Breakdown

| Source Class | `train` Inst / Imgs | `valid` Inst / Imgs | `test` Inst / Imgs | Total Instances | Images Containing Class (% of 7,035) |
|---|---:|---:|---:|---:|---:|
| `0: head` | `4,612` / `911` | `1,339` / `256` | `726` / `140` | `6,677` | `1,307` (`18.58%`) |
| `1: helmet` | `13,919` / `4,503` | `3,913` / `1,297` | `1,915` / `636` | `19,747` | `6,436` (`91.49%`) |
| `2: person` | `450` / `154` | `101` / `32` | `64` / `23` | `615` | `209` (**`2.97%`**) |
| **Total** | `18,981` / `4,916` | `5,353` / `1,413` | `2,705` / `706` | **`27,039`** | `7,035` (`100%`) |

Per-image class co-occurrence across the `7,035` images:
- `helmet` only: `5,543` (`78.79%`)
- `head + helmet`: `684` (`9.72%`)
- `head` only: `599` (`8.51%`)
- `helmet + person`: `185` (`2.63%`)
- `head + helmet + person`: `24` (`0.34%`)

### 2.3 Bounding-Box Geometry vs. Existing SafeGuard Sources

Comparing normalized box geometry against [audit_summary.json](../../reports/dataset/audit_summary.json):

| Dataset / Class | Count | Median W | Median H | Median Area | P10–P90 Area | Median AR (`w/h`) | Small (`<0.01`) |
|---|---:|---:|---:|---:|---|---:|---:|
| **HHW `head`** | `6,677` | `0.0580` | `0.1042` | `0.00603` | `0.00218 – 0.01961` | **`0.560`** | `71.62%` (`40.89%` `<0.005`) |
| RF v30 `NO-Hardhat` | `402` | `0.0743` | `0.0565` | `0.00413` | `0.00012 – 0.07451` | **`1.264`** | — |
| UL PPE `no_helmet` | `813` | `0.0219` | `0.0319` | **`0.00068`** | `0.00007 – 0.02227` | `0.667` | — |
| **HHW `helmet`** | `19,747` | `0.0710` | `0.1182` | **`0.00829`** | `0.00222 – 0.04855` | **`0.584`** | `55.60%` (`32.28%` `<0.005`) |
| RF v30 `Hardhat` | `574` | `0.0430` | `0.0490` | `0.00240` | `0.00013 – 0.02634` | **`0.898`** | — |
| UL PPE `helmet` | `1,576` | `0.0466` | `0.0574` | `0.00273` | `0.00013 – 0.02125` | **`0.828`** | — |
| **HHW `person`** | `615` | `0.1560` | `0.4926` | `0.07826` | `0.01540 – 0.33346` | `0.358` | `6.02%` |
| RF v30 `Person` | `1,148` | `0.1152` | `0.3391` | `0.03869` | `0.00093 – 0.46907` | `0.396` | — |
| UL PPE `Person` | `2,443` | `0.0920` | `0.3125` | `0.02877` | `0.00085 – 0.38976` | `0.298` | — |

---

## 3. Phase 3 — Semantic Compatibility & Visual Inspection

Visual contact sheets rendered in [reports/dataset/hard_hat_workers/](../../reports/dataset/hard_hat_workers/):
- [head_no_person_samples.png](../../reports/dataset/hard_hat_workers/head_no_person_samples.png)
- [head_and_helmet_samples.png](../../reports/dataset/hard_hat_workers/head_and_helmet_samples.png)
- [person_annotated_vs_missing.png](../../reports/dataset/hard_hat_workers/person_annotated_vs_missing.png)
- [helmet_samples.png](../../reports/dataset/hard_hat_workers/helmet_samples.png)
- [cross_dataset_duplicates.png](../../reports/dataset/hard_hat_workers/cross_dataset_duplicates.png)

### 3.1 `person` Class — Severely Incompletely Annotated (CRITICAL DEFECT)

- Only **`209` of `7,035` images (`2.97%`)** contain any `person` annotation (`615` `person` instances total vs. `26,424` `head`+`helmet` worker instances — **`97.67%` of visible workers lack a `person` box**).
- **`6,826` images (`97.03%`)** contain labeled `head` or `helmet` workers with **zero** `person` annotations.
- Even inside the `209` images that *do* have `person` labels, `21` (`10.0%`) contain more `head+helmet` boxes than `person` boxes (incomplete per-image coverage).
- Confirmed visually in [person_annotated_vs_missing.png](../../reports/dataset/hard_hat_workers/person_annotated_vs_missing.png): multi-worker construction scenes (`3–15` labeled heads/helmets per frame) have zero `person` boxes. Training YOLO on these images for canonical class `0: person` would systematically treat `~97%` of visible people as negative background.

### 3.2 `head` Class vs. Canonical `no_helmet`

- **Mutual exclusivity with `helmet`**: Across all `7,332` `head`–`helmet` box pairs checked in co-occurring images, **only `4` images (`0.05%`) had any `head`–`helmet` box overlap with `IoU >= 0.30`** (`003385` `IoU=0.874`, `002686` `IoU=0.867`, `002109` `IoU=0.748`, `005098` `IoU=0.303`, all inspected in the top row of [head_and_helmet_samples.png](../../reports/dataset/hard_hat_workers/head_and_helmet_samples.png)). Thus `head` does **not** generally denote "any head including helmeted heads"; in `~99.9%` of annotations it denotes an **uncovered head**.
- **Domain / Scene Contamination**: Visual inspection of [head_no_person_samples.png](../../reports/dataset/hard_hat_workers/head_no_person_samples.png) reveals substantial non-construction / non-PPE imagery where `head` labels indoor conference attendees (`002519`), indoor banner group photos (`002002`), and outdoor casual/politician crowd visits (`006751`, `001226`, `004470`).
- **Box Extent Difference**: `head` encloses the entire head + face (`median_ar = 0.560`, `median_area = 0.00603`), differing from both `roboflow_v30` `NO-Hardhat` (often crown/hairline-only, `median_ar = 1.264`) and `ultralytics_ppe` `no_helmet` (`9×` smaller median area `0.00068`).

### 3.3 `helmet` Class vs. Canonical `helmet` (CRITICAL BOX GEOMETRY MISMATCH)

- Confirmed visually in [helmet_samples.png](../../reports/dataset/hard_hat_workers/helmet_samples.png) and side-by-side on identical shared images in [cross_dataset_duplicates.png](../../reports/dataset/hard_hat_workers/cross_dataset_duplicates.png):
  - **`hard_hat_workers` `helmet` boxes frame the FULL HEAD AND FACE (top of helmet down to the chin/jaw)** (`median_ar = 0.584`, `median_area = 0.00829`).
  - **SafeGuard (`roboflow_v30` & `ultralytics_ppe`) `helmet` boxes tightly frame the HARDHAT SHELL ONLY** (`median_ar = 0.898` RF / `0.828` UL, `median_area = 0.00240` RF / `0.00273` UL — **`3.0–3.5×` smaller**).
- Mixing `19,747` full-head `helmet` boxes with `2,150` shell-only SafeGuard `helmet` boxes (`9.2:1` ratio favoring the incompatible convention!) would corrupt SafeGuard's bounding-box regression targets and degrade `mAP@50:95`.

### 3.4 Missing Canonical PPE Classes (`vest`, `gloves`)

- `vest` (`0` annotations) and `gloves` (`0` annotations) are completely unannotated across all `7,035` images, even though workers visibly wear high-visibility vests and work gloves in these photos (see [cross_dataset_duplicates.png](../../reports/dataset/hard_hat_workers/cross_dataset_duplicates.png) Row 4). Under unified 5-class YOLO loss, all `7,035` images act as unmasked false negatives against `EXP-E`'s `vest` (`0.9386` UL mAP50) and `gloves` (`0.8612` UL mAP50) classes.

### 3.5 Class Mapping Assessment

| Source Class | Instances | Proposed Direct Mapping to 5-Class `merged.yaml` | Compatibility Verdict |
|---|---:|---|---|
| `0: head` | `6,677` | `no_helmet` *(Rejected for direct merge)* | Uncovered head in ~99.9% of boxes, but full-head box extent & non-construction conference/crowd scenes (`002519`, `006751`) + unannotated PPE context. |
| `1: helmet` | `19,747` | `helmet` *(Rejected for direct merge)* | Semantic hard hat, but **full head+face** box (`AR 0.584`, area `0.00829`) conflicts with SafeGuard tight-shell `helmet` (`AR 0.828–0.898`, area `0.00240–0.00273`). |
| `2: person` | `615` | `person` *(Rejected)* | **Missing in `97.03%` (`6,826/7,035`) of images**; severe background-negative corruption if merged. |
| *(missing)* | `0` | `vest`, `gloves` | **100% unannotated** despite visible vests/gloves in scenes. |

---

## 4. Phase 4 — Duplicate & Leakage Audit

Checked across all `7,035` `hard_hat_workers` images, both raw sources (`roboflow_v30`: `717`, `ultralytics_ppe`: `1,000` labeled split images), and all processed splits in `data/processed/`.

| Check | Exact SHA-256 Matches | Perceptual Near-Duplicates (`pHash Hamming <= 6`) |
|---|---:|---:|
| **Within `hard_hat_workers` (`7,035` images)** | `0` groups | **`168` clusters** (`192` pairs), **`79` cross-split clusters** (`train`/`valid`/`test` leakage inside HHW) |
| **HHW vs. `ultralytics_ppe` (`1,000` images)** | `0` | **`0`** |
| **HHW vs. `roboflow_v30` (`717` images)** | **`1`** | **`31`** (`24` at `Hamming = 0`, `2` at `d=2`, `5` at `d=4–6`) |

Cross-dataset leakage breakdown against SafeGuard's processed splits (`32` total overlaps with `roboflow_v30`, verified visually in [cross_dataset_duplicates.png](../../reports/dataset/hard_hat_workers/cross_dataset_duplicates.png)):
- **Locked `test_roboflow` overlap (`7` images, ALL `Hamming = 0` exact visual copies)**:
  - `hhw:train` (`5`): `000005`, `002551`, `003357`, `004063`, `004763`
  - `hhw:valid` (`2`): `006463`, `006672`
- **`val_roboflow` overlap (`4` images)**:
  - `hhw:train` (`3`): `000133` (`d=4`), `004424` (`d=0`), `004631` (`d=4`)
  - `hhw:test` (`1`): `005559` (`d=0`)
- **`train_roboflow` overlap (`21` images)**:
  - Exact SHA-256 (`1`): `hhw:train` `000415`
  - Near-duplicate (`20`): `9` from `hhw:train`, `7` from `hhw:valid`, `4` from `hhw:test`

`roboflow_v30` visibly re-used these exact Northeastern University images and **re-annotated them under SafeGuard-compatible multi-PPE rules** (tight `Hardhat` shell, `Person`, `Safety Vest`, `Gloves`, `NO-Safety Vest`), confirming the `helmet` bounding-box discrepancy directly on identical pixel inputs.

---

## 5. Phase 5 — Benefits, Risks, Integration Strategy & Final Decision

### Potential Benefits
- Adds `6,677` uncovered-head (`head`) and `19,747` helmeted-head (`helmet`) examples at zero licensing cost (`Public Domain`).
- Could theoretically enrich head-vs-helmet appearance diversity across varied lighting/angles if isolated from 5-class localization.

### Critical Risks
1. **Severe Regression on `person`, `vest`, and `gloves`**: Adding `7,035` images (`6.2×` larger than SafeGuard's `1,131`-image training set) with `97.03%` missing `person` labels and `100%` missing `vest`/`gloves` labels would destroy recall on three of SafeGuard's five core classes.
2. **Box Localization Corruption for `helmet` and `no_helmet`**: Mixing `19,747` full-head+face `helmet` boxes (`AR 0.584`) into `EXP-E` (`2,150` shell-tight `helmet` boxes, `AR 0.828–0.898`) degrades `mAP@50:95` and compliance association geometry.
3. **Domain Noise**: Non-construction conference rooms (`002519`) and civilian crowd visits (`006751`, `001226`) introduce off-domain `no_helmet` priors.
4. **Test/Val Leakage Risk**: `32` images overlap `roboflow_v30` (including `7` identical visual matches inside locked `test_roboflow`) and `79` internal pHash clusters cross HHW's own `train`/`valid`/`test` boundaries.

### Explicit Decision: **REJECT**

**REJECT direct integration/merging of `hard_hat_workers` into SafeGuard's canonical 5-class training dataset (`data/processed/merged.yaml`) and do NOT launch a merged `EXP-G`.**

### Recommended Strategy Moving Forward
1. **Proceed with `EXP-E` as the winning 5-class detector**: `EXP-E` already achieved `0.8335` mAP@50 (`0.8833` F1) on `ultralytics_test` and `0.5610` mAP@50 (`0.6683` F1) on `roboflow_test` across all 5 canonical classes without partial-label corruption; advance `EXP-E` to ONNX export, parity verification, benchmarking, and FastAPI integration.
2. **Keep `data/raw/hard_hat_workers/` untouched / unused in `dvc.yaml`** (or archive out of the active pipeline).
3. **If higher `no_helmet` recall is specifically required in future MLOps iterations**, do **not** merge partial 2/3-class datasets into YOLO's 5-class head; instead choose one of:
   - *Option A (Preferred)*: Source a **full-PPE construction dataset** that annotates `person`, tight-shell `helmet`, `vest`, `gloves`, and `no_helmet` jointly.
   - *Option B (Non-merge Transfer Learning)*: Strip `person`, filter out the `32` `roboflow_v30` overlaps (`7` test matches) + `79` cross-split pHash clusters, pretrain a 2-class (`head`, `helmet`) `YOLO11s` backbone on HHW, **discard the detection head**, and fine-tune exclusively on SafeGuard's unmodified 5-class `merged.yaml`.
   - *Option C (Decoupled Head Verifier)*: Train a lightweight cropped-patch classifier (`helmet` vs. `no_helmet`) using filtered construction crops to second-check low-confidence `person` heads in `src/compliance/`.
