# Dataset Strategy, Taxonomy & DVC Pipeline

**Sources**: [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (Sections 4–6, Appendices A–C, p.6–10, p.22) and [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (Sections 4.3, 5.3, 9, p.4–5, p.6–7, p.11).

> **CRITICAL PROVISIONAL NOTICE**: All publisher-reported image counts, split counts, and candidate class mappings below are **provisional** until the downloaded archives and YAML files pass the mandatory dataset audit (`Appendix A` checklist). Never treat provisional counts or guessed class IDs as training facts.

---

## 1. Pinned Data Sources

### Source A — Roboflow Construction Site Safety (v30)
- **Project / Version**: `roboflow-universe-projects/construction-site-safety/30` (raw, non-augmented version 30 — **not** augmented version 27).
- **URL**: `https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30`
- **Publisher-Reported Figures (Provisional)**: 717 total images (521 train / 114 validation / 82 test); auto-orient applied; no augmentations applied.
- **License**: CC BY 4.0 (must verify and preserve license text with downloaded export).
- **Authoritative Artifact**: Downloaded version-30 archive and its exported YAML/labels.

### Source B — Ultralytics Construction-PPE (November 2026 Release)
- **Release**: November 2026 Construction-PPE release (do **not** use older 1,416-image / 11-class figures).
- **URL**: `https://docs.ultralytics.com/datasets/detect/construction-ppe`
- **Publisher-Reported Figures (Provisional)**: 1,000 total images (800 train / 200 validation); 12 classes; 14,863 reviewed bounding boxes; **no separate publisher-provided test split**; combination of human-collected and generated images.
- **License**: CC BY 4.0.
- **Authoritative Artifact**: Exact downloaded archive, `construction-ppe.yaml`, labels, and license.

### Source Versioning & Immutability Rules
- Pin exact project/version, download date, export format, archive SHA-256, and license in `source_manifest.json`.
- Store original YAML files and source manifests alongside DVC-managed references.
- Never edit raw images or raw annotations; all transformations write to `data/interim/` or `data/processed/`.
- Never replace a pinned archive in place.

---

## 2. Canonical Taxonomy & Semantic Intersection Gate

### 2.1 Provisional Candidate Intersection (`SG-SPEC-001` p.8)
| Canonical Class (Provisional) | Roboflow v30 Candidate | Ultralytics Candidate | Gate Status |
|---|---|---|---|
| `person` | `Person` | `Person` | Provisional retain |
| `helmet` | `Hardhat` | `helmet` | Provisional retain |
| `vest` | `Safety Vest` | `vest` | Provisional retain |
| `gloves` | `Gloves` | `gloves` | Provisional retain |
| `no_helmet` | `NO-Hardhat` | `no_helmet` | Provisional retain |
| `no_vest` | `NO-Safety Vest`? | `no_vest` (Ultralytics class 11) | **Candidate only** — verify exact v30 export YAML & visual annotation semantics |

### 2.2 Excluded Single-Source Classes
Classes present in only one source (`boots`, `goggles`, `no_gloves`, `no_boots`, `no_goggle`, `forklifts`, `machinery`, `vehicles`, `masks`, generic `none`) are **excluded** from the initial merged taxonomy to prevent contradictory background supervision (`SG-SPEC-001` p.8).

### 2.3 Mandatory 7-Step Taxonomy Approval Gate (`SG-SPEC-001` p.8)
1. Extract exact class list and IDs from each downloaded source YAML (never infer from screenshots or memory).
2. Count instances and distinct images per class in each source split.
3. Render representative labeled samples for each candidate class from both sources.
4. Check annotation target semantics (verify boxes refer to comparable visual objects/body regions).
5. Review edge cases (occlusion, out-of-frame body parts, partially visible PPE, ambiguous missing-PPE labels — noting Ultralytics' rule that occluded/out-of-frame body parts do not establish missing equipment).
6. Approve only classes with compatible semantics and sufficient evidence in both sources.
7. Write the final canonical order, source IDs/names, inclusion rationale, evidence paths, and audit date to `configs/class_mapping.yaml` (single source of truth; never hardcode class IDs across scripts).

---

## 3. Deduplication, Split Policy & Leakage Prevention

### 3.1 Deduplication Rules (`SG-SPEC-001` p.9)
- Compute **SHA-256** hashes for exact file duplicates across both sources and within splits.
- Compute **perceptual hashes and/or image embeddings** to flag likely near-duplicates.
- Manually review likely duplicate pairs before removing records.
- **Test Protection Rule**: If a test image duplicates a training image, **retain the test copy** and remove the training copy from the eligible training pool.
- **Scene Grouping Rule**: Group video frames or related scene captures so near-identical frames cannot cross train/validation/test boundaries.

### 3.2 Split Policy (`SG-SPEC-001` p.9)
- **Roboflow v30 Test Split**: Preserve as a locked source-specific evaluation set (subject to duplicate-leakage checks).
- **Roboflow v30 & Ultralytics Validation Splits**: Preserve for model development/selection and threshold tuning only (**not** as test sets).
- **Ultralytics Internal Holdout**: Derive from Ultralytics' original training pool only after duplicate/scene-group checks (`10–15%` is a planning range, not a fixed fact).
- **Merged Training**: Use only eligible training partitions; never mix test images into training or tuning.
- **Evaluation Reporting**: Evaluate source-specific test sets separately; report combined scores only as supplementary summaries, never as replacements.
- **Leakage Rule**: Split and leakage checks must execute **before** any training-only augmentation. Validation and test images must never be augmented into training data or used to pick thresholds, hyperparameters, or checkpoints.

---

## 4. Minimum DVC Pipeline Stages (`SG-SPEC-001` p.9–10)

```text
ingest/prepare -> validate -> harmonize -> deduplicate -> merge/split -> statistics -> train -> evaluate -> export
```

| Stage | Required Operations | Evidence / Output Artifact |
|---|---|---|
| `ingest/prepare` | Place pinned archives in `data/raw/`; record source, license, date, hashes | `source_manifest.json`, archive SHA-256 hashes |
| `validate` | Check image readability, label/image pairing, class IDs, normalized coords, box geometry, unknown labels | Validation report, invalid-record list |
| `harmonize` | Map verified source labels to canonical IDs; filter non-intersection labels without altering raw data | `configs/class_mapping.yaml`, harmonized labels (`data/interim/`) |
| `deduplicate` | Exact (SHA-256) and near-duplicate (pHash/embeddings) detection across sources & splits | Duplicate report, duplicate groups |
| `merge/split` | Combine eligible images/labels preserving provenance; create leakage-aware partitions with locked test sets | Merged manifest, split manifest, split statistics (`data/processed/`) |
| `statistics` | Report class counts, images/class, boxes/class, source proportions, bbox size distributions | CSV / JSON / plots in `reports/` |
| `train` -> `evaluate` -> `export` | Train experiments, evaluate locked splits, export verified ONNX | MLflow runs, `models/`, benchmark reports |

The pipeline must support `dvc status` and `dvc repro`, rerunning only stages affected by changed dependencies.

---

## 5. Unresolved Dataset & Taxonomy Decisions (Pending Approval)

1. **Archive Verification (`GATE`)**: Actual file counts, class IDs, YAML schemas, and license files from the downloaded Roboflow v30 and Ultralytics Nov 2026 archives have not yet been audited.
2. **Candidate Class `no_vest` (`GATE`)**: Inclusion or exclusion of `no_vest` is unresolved pending visual and semantic comparison of Roboflow v30 against Ultralytics class `no_vest`.
3. **Final Canonical Taxonomy (`GATE`)**: Exact class count (5 vs 6) and canonical integer ID ordering in `configs/class_mapping.yaml` require user approval after the audit.
4. **Ultralytics Internal Holdout Size**: Exact percentage/count (within the `10–15%` planning range) to be determined after duplicate and scene-group auditing.
5. **Independent Real-World Evaluation Set**: Decision on whether collecting/annotating an independent real-world test set (`SG-SPEC-001` p.9, "if feasible") is in scope for V1.
6. **Near-Duplicate Thresholds & Removal Approval**: Similarity thresholds (perceptual hash / embedding distance) and manual review sign-off before removing any duplicate records from the training pool (`GEMINI.md` Rule 17).
