# SafeGuard Dataset Inspection, Validation & Semantic Audit Report

**Audit Date**: 2026-11-25  
**Mode**: Strictly read-only on `data/raw/` (raw archives and annotations untouched)  
**Generated Artifacts**:
- [class_distribution.csv](class_distribution.csv)
- [class_distribution.png](class_distribution.png)
- [bbox_analysis.png](bbox_analysis.png)
- [audit_summary.json](audit_summary.json)
- `annotation_gallery/` (12 visual sample sheets across both sources)
- [class_mapping.yaml](../../configs/class_mapping.yaml)

---

## 1. Source Verification & Original Split Assignments

### Source A — Roboflow Construction Site Safety (`v30`)
- **Path**: `data/raw/roboflow/construction-site-safety-v30/data.yaml`
- **Export Metadata** (`README.roboflow.txt`): Exported 2023-04-19 in YOLOv8 format; auto-orient with EXIF stripping; no image augmentations applied; License: `CC BY 4.0`.
- **Verified Counts**: **717 images** and **717 label files**, **5,570 valid bounding boxes** (`521` train / `114` valid / `82` test). Matches publisher-reported totals (`717 = 521/114/82`).
- **Original Class Count**: **25 classes** (`0..24`).

### Source B — Ultralytics Construction-PPE (`November 2026 Release`)
- **Path**: `data/raw/ultralytics/construction-ppe/data.yaml`
- **Release Metadata** (`README.md`, `LICENSE.txt`, `manifest.jsonl`): Open-sourced November 2026 by Ultralytics; combines human-collected and generated images; License: `CC BY 4.0`.
- **Verified Counts**: **1,000 images** and **1,000 label files**, **14,863 valid bounding boxes** (`800` train / `200` val; **no publisher-provided test split**). Matches publisher-reported totals (`1,000 = 800/200`, `14,863` boxes).
- **Original Class Count**: **12 classes** (`0..11`).

---

## 2. Annotation Integrity & Coordinate Validation

| Check | Roboflow v30 | Ultralytics Construction-PPE |
|---|---:|---:|
| Total images | 717 | 1,000 |
| Total `.txt` label files | 717 | 1,000 |
| Missing label files | 0 | 0 |
| Orphan label files | 0 | 0 |
| Corrupted / unreadable images | 0 | 0 |
| Empty label files (0 boxes) | 52 | 0 |
| Valid YOLO bounding boxes | 5,570 | 14,863 |
| Invalid coordinates / unknown IDs / bad boxes | 0 | 0 |

All `1,717` images decode cleanly and all `20,433` bounding boxes pass normalized `[0, 1]` bounds, positive dimension, column-count, and valid-class-ID checks. Note that Roboflow v30 contains `52` empty label files (background/negative scenes).

---

## 3. Class Distribution & Bounding-Box Geometry

Full per-source, per-split counts are recorded in [class_distribution.csv](class_distribution.csv) and plotted in [class_distribution.png](class_distribution.png) and [bbox_analysis.png](bbox_analysis.png).

### Candidate Intersection Class Counts & Bounding-Box Statistics

| Candidate Canonical Class | Roboflow v30 Class (ID) | RF Instances (Images: Train / Valid / Test) | RF Median W × H (Area, AR) | Ultralytics Class (ID) | UL Instances (Images: Train / Val) | UL Median W × H (Area, AR) |
|---|---|---:|---:|---|---:|---:|
| `person` | `Person` (8) | 1,148 (`372` / `84` / `59`) | `0.115 × 0.339` (`0.0387`, AR `0.396`) | `Person` (5) | 2,443 (`791` / `200`) | `0.092 × 0.313` (`0.0288`, AR `0.298`) |
| `helmet` | `Hardhat` (2) | 574 (`159` / `42` / `30`) | `0.043 × 0.049` (`0.00240`, AR `0.898`) | `helmet` (0) | 1,576 (`613` / `146`) | `0.047 × 0.057` (`0.00273`, AR `0.828`) |
| `vest` | `Safety Vest` (11) | 424 (`138` / `28` / `22`) | `0.060 × 0.120` (`0.00693`, AR `0.505`) | `vest` (2) | 1,816 (`644` / `157`) | `0.064 × 0.131` (`0.00831`, AR `0.507`) |
| `gloves` | `Gloves` (1) | 296 (`75` / `11` / `10`) | `0.027 × 0.041` (`0.00104`, AR `0.565`) | `gloves` (1) | 1,929 (`546` / `136`) | `0.050 × 0.071` (`0.00394`, AR `0.696`) |
| `no_helmet` | `NO-Hardhat` (5) | 402 (`171` / `37` / `25`) | `0.074 × 0.057` (`0.00413`, AR `1.264`) | `no_helmet` (6) | 813 (`337` / `91`) | `0.022 × 0.032` (`0.00068`, AR `0.667`) |
| `no_vest` *(candidate)* | `NO-Safety Vest` (7) | 582 (`214` / `56` / `36`) | `0.119 × 0.184` (`0.02345`, AR `0.596`) | `no_vest` (11) | 567 (`300` / `84`) | `0.109 × 0.198` (`0.02499`, AR `0.583`) |

---

## 4. Duplicate & Split-Leakage Audit

- **Exact Duplicates (`SHA-256`)**: **0** exact duplicate groups across all `1,717` images.
- **Near-Duplicates (`pHash`, Hamming distance ≤ 6)**: **34 clusters** detected:
  - **Cross-source clusters**: **0** (Roboflow v30 and Ultralytics Construction-PPE have zero exact or near-duplicate overlap).
  - **Ultralytics internal clusters**: **1** (`Group 6`, Hamming=6, `train` vs `val`).
  - **Roboflow v30 internal clusters**: **33** clusters, dominated by sequential video frames (`youtube-*`, `IMG_0871_mp4-*`, `IMG_3103_mp4-*`, `Movie-on-10-31-22-at-10_08-AM_mov-*`, `autox4_mp4-*`).
  - **Critical Train/Test Leakage in Roboflow v30**: **7 clusters** (`Groups 4, 5, 16, 18, 23, 29, 34`) span `train` and `test` (including `Group 4` at Hamming distance `0`, `IMG_0871_mp4-23`, and `Group 5` spanning `train`, `valid`, and `test`). Additional clusters span `train` and `valid`.
  - **Mandatory Remediation (Pending Split Phase)**: Retain locked Roboflow `test` copies and remove leaked `train` frames (and group related video frames by video stem) before building the merged training dataset.

---

## 5. Visual & Semantic Equivalence Findings (`annotation_gallery/`)

All 12 generated gallery sheets (`person`, `helmet`, `vest`, `gloves`, `no_helmet`, `no_vest` × both sources) were visually inspected:

1. **`person` (`Person` ↔ `Person`) — Compatible**: Both label full visible human bodies.
2. **`helmet` (`Hardhat` ↔ `helmet`) — Compatible**: Both tightly bound worn safety helmets with similar aspect ratios (`0.898` vs `0.828`).
3. **`vest` (`Safety Vest` ↔ `vest`) — Compatible**: Both bound torso high-visibility vests (`AR 0.505` vs `0.507`). Minor Roboflow edge case observed (`005139`: orange work jacket labeled `Safety Vest`).
4. **`gloves` (`Gloves` ↔ `gloves`) — Compatible (Small-Object & Support Warning)**: Both bound individual gloved hands, but objects are very small (`median area 0.00104` in RF) and Roboflow `valid`/`test` only contain `11`/`10` images with `Gloves`.
5. **`no_helmet` (`NO-Hardhat` ↔ `no_helmet`) — SEMANTIC BOX-EXTENT DISCREPANCY**:
   - Visually verified in `no_helmet__roboflow_v30.png` vs `no_helmet__ultralytics_ppe.png`: Roboflow `NO-Hardhat` boxes frequently mark only a **wide horizontal strip over the top of the head/hairline** (`median AR = 1.264`, area `0.00413`), whereas Ultralytics `no_helmet` marks the **full head/face** (`median AR = 0.667`, area `0.00068`).
6. **`no_vest` (`NO-Safety Vest` ↔ `no_vest`) — STRONG CANDIDATE (Pending Approval)**:
   - Visually verified in `no_vest__roboflow_v30.png` vs `no_vest__ultralytics_ppe.png`: Both label the **upper torso lacking a high-visibility vest** with nearly identical aspect ratios (`0.596` RF vs `0.583` UL), areas (`0.0235` vs `0.0250`), and balanced counts (`582` RF vs `567` UL). Remains `pending_approval` in `configs/class_mapping.yaml` until approved.
7. **Roboflow Border-Reflection Padding**:
   - Visually observed in `-1670-`, `-1680-`, `-2252-`, `-2297-`, `-2406-`: several Roboflow v30 images have **mirror-padded borders** with annotations also placed on mirrored person fragments along the edges.

---

## 6. Unresolved Decisions & Approval Gate

No merged training dataset has been generated and no model has been trained. Please review and approve:

1. **Include `no_vest` as class ID `5`?** (Recommended: **Yes**, given compatible torso semantics, matching geometry, and 582/567 balanced support; currently `pending_approval` in `configs/class_mapping.yaml`).
2. **How to handle `no_helmet` box-extent mismatch?** (Keep both sources as `no_helmet` and measure cross-source vs single-source impact in `EXP-A`/`B`/`C`, or apply a different policy).
3. **Approve duplicate & video-scene leakage pruning policy** for Roboflow v30 (`train`/`valid`/`test` frame groups) and Ultralytics (`Group 6`) before creating `data/processed/`.
4. **Roboflow mirror-padded images** (retain as-is in source baseline vs flag/document as known source noise).
