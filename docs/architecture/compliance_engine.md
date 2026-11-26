# SafeGuard Compliance Engine, Alerts & Analytics Architecture

**Modules**: `src/compliance/rules.py`, `src/compliance/alerts.py`, `src/compliance/analytics.py`  
**Service Wrapper**: `backend/app/services/compliance_service.py`  
**Routes**: `POST /predict/compliance`, `GET /compliance/alerts`, `PATCH /compliance/alerts/{event_id}`, `GET /compliance/analytics`, `POST /compliance/reset`

---

## 1. Strictly Separated 4-Stage Pipeline

```text
Stage 1: Raw Model Detections (ONNXDetector -> DetectionInput + Taxonomy Gate)
   │
   ▼
Stage 2: Anonymous Worker Tracking (ByteTrackTracker -> active, lost, retired tracks)
   │
   ▼
Stage 3: Spatial Association & Compliance Interpretation (ComplianceInterpreter)
   │
   ▼
Stage 4: Alert Generation & Stream Analytics (AlertManager + StreamAnalyticsTracker)
```

1. **Raw Model Detections (`DetectionInput` / `load_approved_class_names`)**:
   - Validates finite bounding-box coordinates (`x2 > x1`, `y2 > y1`), confidence bounds `[0.0, 1.0]`, and dynamically enforces approved canonical classes from `configs/class_mapping.yaml` (`person`, `helmet`, `vest`, `gloves`, `no_helmet`). Unapproved or candidate classes (such as `no_vest` or `head`) are filtered and counted (`unapproved_detections_filtered`), never reaching compliance or alert rules.
2. **Worker Tracking (`ByteTrackTracker`)**:
   - Assigns stream-scoped anonymous IDs only (`track_id`), handling temporary occlusion recovery (`lost_tracks`) and clean expiration (`retired_track_ids`).
3. **PPE Compliance Interpretation (`ComplianceInterpreter`)**:
   - Runs optimal 1-to-1 bipartite matching (`_optimal_bipartite_assign`) for unique PPE items (`helmet`, `no_helmet`, `vest`) and multi-hand matching for `gloves`, guarded by `spatial_ambiguity_margin` and box-visibility checks.
4. **Alert Generation & Analytics (`AlertManager`, `StreamAnalyticsTracker`)**:
   - Confirms explicit missing-PPE violations across `confirm_frames` uninterrupted active frames, enforces per-`(track_id, violation_category)` cooldown & duplicate suppression, renders annotated JPEG evidence snapshots (`evidence_frame_b64`) when a violation is triggered, manages `event_status` lifecycle (`pending_review`, `acknowledged`, `verified_potential_event`, `dismissed_false_alarm`, `resolved`), and aggregates stream analytics (`workers_observed`, `ppe_detections`, `potential_violations`, `alerts_over_time`, `violation_categories`).

---

## 2. Compliance State Semantics & Decision-Support Boundaries

- **Missing Positive PPE Is Never Proof of Absence**:
  - Because `no_vest` (`pending_approval`) and `no_gloves` are **not** trained classes in `EXP-E`, an undetected `vest` sets `vest_status = "unknown"` (`"vest_not_detected_no_vest_class_untrained"`), never a violation.
  - An undetected `helmet` sets `helmet_status = "unknown"` (`"head_ppe_not_observed"`).
- **`insufficient_visibility`**:
  - Triggered when a person silhouette is too small (`height < min_person_height_px` or `width < min_person_width_px`), currently occluded (`track.state == "lost"`), or has its head/upper region clipped along the top image border (`y1 <= edge_clip_margin_px`). Resets violation confirmation streaks so distant or partial silhouettes do not generate false alerts.
- **`uncertain` / `ambiguous`**:
  - Triggered when conflicting `helmet` and `no_helmet` boxes overlap the same person within `ambiguous_margin` (`helmet_status = "ambiguous"`, `overall_status = "unknown"`), or when a single PPE detection sits equidistantly inside the overlapping zones of multiple adjacent workers within `spatial_ambiguity_margin` without disambiguating competing boxes (`overall_status = "uncertain"`, `"ambiguous_spatial_overlap"`).
- **Alert Trigger Rule**:
  - Only explicit `no_helmet` detections reliably associated with a sufficiently visible worker for `confirm_frames` consecutive active frames emit a `potential_no_helmet` event (`violation_category = "no_helmet"`, `requires_human_review = True`). Occluded or intermittent frame gaps reset the streak.

---

## 3. Spatial Association Geometry

PPE detections are matched inside anatomical zones derived from each tracked `[px1, py1, px2, py2]` box (`pw = px2 - px1`, `ph = py2 - py1`):

| Region | Classes | Relative Bounds | Min Containment | Extra Guard |
|---|---|---|---:|---|
| Head zone | `helmet`, `no_helmet` | `[px1-0.12*pw, py1-0.18*ph, px2+0.12*pw, py1+0.42*ph]` | `0.45` | `ppe_area / person_area <= 0.55`; 1-to-1 + overlap ambiguity check |
| Torso zone | `vest` | `[px1-0.06*pw, py1+0.10*ph, px2+0.06*pw, py1+0.80*ph]` | `0.40` | 1-to-1 + overlap ambiguity check |
| Hands/arms zone | `gloves` | `[px1-0.18*pw, py1+0.10*ph, px2+0.18*pw, py2+0.06*ph]` | `0.35` | Multi-item assignment (up to pair of gloves) |

---

## 4. Known Spatial Association & Compliance Limitations

1. **Upright-Posture & Camera-Angle Bias**:
   - The head/torso/gloves zones assume approximately upright workers viewed from eye-level or elevated oblique CCTV angles. Workers bending deeply forward, kneeling, crouching below scaffolding, or viewed from near-overhead cameras can place a worn helmet outside the upper 42% box fraction, causing valid PPE to land in `unmatched_ppe` and the worker to report `"unknown"`.
2. **2D Depth Overlap Confusion**:
   - Even with 1-to-1 Hungarian matching and `spatial_ambiguity_margin`, severe front-to-back 2D overlap (one worker standing directly in front of another without depth sensing) can leave one worker occluded or trigger `"uncertain"` states (`ambiguous_spatial_overlap`), intentionally trading recall for low false-accusation risk.
3. **Cross-Dataset `no_helmet` Box Geometry Variance**:
   - Roboflow `NO-Hardhat` annotations often frame only the upper hairline/crown while Ultralytics `no_helmet` frames the full head/face; small-object recall for `no_helmet` (`0.56–0.57` on test splits) limits detection distance in wide-angle site views.
4. **Temporal & Privacy Trade-offs**:
   - Fixed `confirm_frames` translates to different wall-clock durations depending on client FPS (e.g., 3 frames = 0.6s at 5 FPS vs 0.1s at 30 FPS).
   - Annotated evidence frames (`evidence_frame_b64`) are bounded in memory per stream and annotated only for review of triggered alerts; SafeGuard never runs facial recognition, worker identification, or continuous raw video recording.
