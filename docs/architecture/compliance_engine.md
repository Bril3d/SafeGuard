# SafeGuard Compliance Engine Architecture

**Module**: `src/compliance/rules.py`  
**Service Wrapper**: `backend/app/services/compliance_service.py`  
**Route**: `POST /predict/compliance`, `POST /compliance/reset`

---

## 1. End-to-End Decision-Support Pipeline

```text
Frame (UploadFile)
  -> InferenceService (singleton ONNXDetector, opset 17 YOLO11s)
  -> ByteTrackTracker (anonymous person tracks across frames)
  -> Geometric Person-to-PPE Association (head, torso, hands/arms zones)
  -> Compliance Rules & State Evaluation
  -> Temporal Confirmation (N consecutive frames) & Cooldown Deduplication
  -> Structured ComplianceEvent (potential_event_requires_human_review)
```

- Raw detections (`DetectionInput`), person-to-PPE geometric associations (`PersonPPEAssociation`), and derived alerts (`ComplianceEvent`) are strictly separated schemas (`backend/app/schemas.py`).
- Existing stateless inference endpoints (`/health`, `/model/metadata`, `/classes`, `/predict`, `/predict/visualize`, `/predict/frame`) remain completely unchanged.

---

## 2. Canonical Taxonomy & Strict Alert Rule

SafeGuard's canonical `EXP-E` detector predicts `5` classes (`configs/class_mapping.yaml`):
`0: person`, `1: helmet`, `2: vest`, `3: gloves`, `4: no_helmet`.

1. **Missing PPE != Automatic Violation**:
   - Because `no_vest` (`pending_approval`) and `no_gloves` are **not** trained classes in `EXP-E`, absence of a positive `vest` or `gloves` detection **never** emits a violation alert. When no `vest` is matched to a person, `vest_status` is set to `"unknown"` (`uncertainty_reasons: ["vest_not_detected_no_vest_class_untrained"]`) and `overall_status` drops to `"unknown"`.
   - When neither `helmet` nor `no_helmet` is observed inside a person's head zone, `helmet_status` is `"unknown"` (`"head_ppe_not_observed"`).
2. **Explicit Missing-Helmet Trigger Only**:
   - Only an explicit `no_helmet` detection matched geometrically to a tracked person's head zone sets `helmet_status = "violation"`.
3. **Conflict / Ambiguity Handling**:
   - If both `helmet` and `no_helmet` overlap the same person's head zone within `ambiguous_margin` (default `0.15` confidence delta), `helmet_status` is marked `"ambiguous"` and `overall_status` is set to `"unknown"` rather than asserting a false violation.
4. **Decision-Support Boundary**:
   - Every emitted alert uses `event_type = "potential_no_helmet"` and `requires_human_review = True`. SafeGuard does not issue autonomous sanctions or certify workplace compliance.

---

## 3. Spatial Association Geometry

PPE detections are assigned 1-to-1 to the highest-scoring tracked person (`_assign_ppe_to_best_person`) using normalized containment inside body zones derived from each person's `[px1, py1, px2, py2]` box (`pw = px2 - px1`, `ph = py2 - py1`):
- **Head zone** (`helmet`, `no_helmet`): `[px1 - 0.12*pw, py1 - 0.18*ph, px2 + 0.12*pw, py1 + 0.42*ph]`, requiring containment `>= min_head_containment` (`0.45`) and `(ppe_area / person_area) <= max_head_person_area_ratio` (`0.55`).
- **Torso zone** (`vest`): `[px1 - 0.06*pw, py1 + 0.10*ph, px2 + 0.06*pw, py1 + 0.80*ph]`, requiring containment `>= min_torso_containment` (`0.40`).
- **Arms/hands zone** (`gloves`): `[px1 - 0.18*pw, py1 + 0.10*ph, px2 + 0.18*pw, py2 + 0.06*ph]`, requiring containment `>= min_gloves_containment` (`0.35`).
- PPE boxes without any qualifying person host are retained in `unmatched_ppe`.

---

## 4. Temporal Confirmation, Cooldown & Configuration

Configurable via `ComplianceConfig` (`src/compliance/rules.py`) and `SAFEGUARD_COMPLIANCE_*` settings (`backend/app/config.py`):

| Parameter | Default | Purpose |
|---|---:|---|
| `confirm_frames` | `3` | Consecutive frames with `no_helmet` required per `track_id` before emitting an event (`pending_confirmation` prior to confirmation). |
| `cooldown_seconds` | `10.0` | Minimum interval between emitted `potential_no_helmet` events for the same `(track_id, event_type)`. |
| `ambiguous_margin` | `0.15` | Confidence difference threshold below which conflicting `helmet` + `no_helmet` is `"ambiguous"`. |
| `conf_person` / `conf_helmet` / `conf_vest` / `conf_gloves` / `conf_no_helmet` | `0.25` | Minimum class confidence filters. |

---

## 5. Known Limitations & Assumptions

- Assumes upright or semi-upright worker silhouettes (head inside upper ~40% + overhang, vest inside middle ~70%); extreme camera tilts or horizontal postures may place PPE in `unmatched_ppe` / `"unknown"`.
- Because `roboflow_v30` and `ultralytics_ppe` test sets revealed modest `no_helmet` recall (`0.56–0.57`) and high small-object miss rates, distant workers frequently remain `"unknown"` rather than emitting alerts.
- Synthetic unit tests verify rule transitions, geometry, confirmation, and cooldown determinism only and do not represent real-world site accuracy.
