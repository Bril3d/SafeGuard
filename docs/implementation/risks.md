# Risk Register & Engineering Mitigations

**Sources**: [`docs/reference/safeguard_specification_v1.pdf`](../reference/safeguard_specification_v1.pdf) (Section 15, p.21; Sections 4–11) and [`docs/reference/professor_project_guide.pdf`](../reference/professor_project_guide.pdf) (Section 9 "Les 5 Erreurs Critiques", p.11).

---

## 1. Technical, Scientific & Academic Risks

| ID | Risk | Source Reference | Impact | Mitigation & Gate | Owner |
|---|---|---|---|---|---|
| **R-01** | **Class semantics differ between Roboflow v30 and Ultralytics Construction-PPE** (especially `no_vest` and partial/occluded PPE) | `SG-SPEC-001` p.8, p.21 | Contradictory supervision where unlabeled objects are treated as background, corrupting fusion training | Enforce 7-step Taxonomy Approval Gate before training: inspect exported YAMLs, render side-by-side bounding-box samples, exclude uncertain/single-source classes, keep `no_vest` gated until proven compatible | Member 1 |
| **R-02** | **Duplicate or near-identical frames cross train/validation/test splits** | `SG-SPEC-001` p.9, p.21; Prof. Guide p.11 | Data leakage causing artificially inflated test metrics and academic penalty | Run SHA-256 exact dedup + perceptual-hash/embedding near-dedup + scene/video grouping *before* split and *before* training-only augmentation; retain test copy and remove train copy on collision | Member 1 |
| **R-03** | **Insufficient support for missing-PPE classes** (`no_helmet`, candidate `no_vest`) | `SG-SPEC-001` p.13, p.21 | Low per-class recall and noisy/unstable potential alerts | Report per-class instance/image support and AP/recall separately; tune confidence/cooldown thresholds on validation set only; never infer missing PPE from absent positive `helmet`/`vest` detections; document limitations | Member 1 |
| **R-04** | **Generated images in Ultralytics dataset dominate training distribution** | `SG-SPEC-001` p.6, p.21 | Domain gap and weaker generalization to real construction/webcam scenes | Record image provenance; evaluate Roboflow v30 locked test split separately from Ultralytics holdout; collect independent real-world evaluation set if feasible | Member 1 |
| **R-05** | **Small or occluded PPE objects (e.g., `gloves`, distant `helmet`)** | `SG-SPEC-001` p.11, p.21 | Missed detections in complex or cluttered scenes | Conduct per-class error analysis (confusion matrix, missed small objects); compare `YOLO11n` vs `YOLO11s` (`EXP-D`) and controlled tuning (`EXP-E`); state limitations clearly | Member 1 |
| **R-06** | **Webcam request backlog & stale overlays** | `SG-SPEC-001` p.15, p.21 | Growing latency lag, browser freeze, and misaligned bounding boxes | Bound in-flight requests, drop stale frames instead of queuing, throttle capture rate to measured inference capacity, preserve aspect ratio, and synchronize rendered boxes with frames | Member 3 |
| **R-07** | **ONNX export / postprocessing mismatch vs. PyTorch checkpoint** | `SG-SPEC-001` p.13, p.21 | Silent accuracy drop or wrong boxes in production FastAPI service | Run automated ONNX graph inspection and PyTorch-vs-ONNX Runtime parity tests on representative images before deployment (`CA-04`) | Member 1, Member 3 |
| **R-08** | **Metric or artifact claims lack provenance** | `SG-SPEC-001` p.4, p.21 | Loss of reproducibility and credibility during defense | Bind every reported result to MLflow run ID, DVC data version, `configs/class_mapping.yaml` hash, and Git commit; never invent metrics | Member 1, Member 2 |
| **R-09** | **"Notebook-Only Syndrome" (delayed backend/frontend integration)** | Prof. Guide p.11; `SG-SPEC-001` p.19 | Broken integration or rushed UI/API at Jalon 4 | Scaffold FastAPI schemas and React UI during Jalon 2; replace early mocks with real ONNX Runtime inference well ahead of Jalon 3/4 | Member 3 |
| **R-10** | **Committing large datasets, weights, or secrets to Git** | Prof. Guide p.11; `SG-SPEC-001` p.17–18 | Bloated Git repo, leaked credentials, broken clean clone | Enforce `.gitignore`, DVC for `data/` and `models/`, environment variables + GitHub Actions secrets, and `.dvc` verification in CI | Member 2 |
| **R-11** | **"Empty Shell" UI or missing automated tests in CI** | Prof. Guide p.11 | Grade penalties on Full-Stack (5 pts) and MLOps (5 pts) rubrics | Implement interactive canvas bounding boxes, animated confidence bars, latency readouts, `pytest` API/unit suites, and Playwright checks in GitHub Actions | Member 2, Member 3 |
| **R-12** | **Misrepresenting operational proxy stats as verified drift or compliance certification** | `SG-SPEC-001` p.3, p.13, p.16 | Unsafe overclaiming or factual inaccuracy | Label all alerts as potential events requiring human review; label unlabelled distribution shift alerts strictly as proxy/statistical warnings | All Members |

---

## 2. Unresolved Risk-Control Decisions (Pending Approval)

1. **Duplicate Removal Sign-Off**: Establishing the review workflow so Member 1 presents candidate duplicate pairs for explicit user/team sign-off before pruning any training samples (`GEMINI.md` Rule 17).
2. **Minimum Per-Class Support Threshold**: Confirming what minimum image/instance count in the data audit would trigger dropping or deferring a class (e.g., `no_vest` or `gloves`) from the V1 canonical intersection.
