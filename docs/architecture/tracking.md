# SafeGuard Anonymous Multi-Object Tracking Architecture & Limitations

**Module**: `src/tracking/tracker.py`  
**Pattern**: Pure-Python two-stage ByteTrack IoU + damped constant-velocity tracker

---

## 1. Design Principles & Privacy Boundaries

1. **Strict Stage Separation**:
   - Stage 2 tracking consumes generic `TrackDetection(bbox_xyxy, confidence, class_id)` inputs filtered exclusively to `person` detections and outputs `TrackingStepResult` (`active_tracks`, `lost_tracks`, `retired_track_ids`).
   - It contains zero imports from `ultralytics`, `torch`, `onnxruntime`, or `backend/`, and zero knowledge of PPE rules or alert logic.
2. **Zero Biometrics & Zero Personal Identities**:
   - Tracks strictly temporary anonymous integer `track_id` values (`1, 2, 3, ...`) scoped in memory per `stream_id`.
   - `Track` and `_InternalTrackState` store **only** bounding-box geometry (`cx, cy, w, h`), damped velocity (`vx, vy, vw, vh`), confidence, frame `age`, `hits`, and `time_since_update`.
   - SafeGuard **never** extracts, retains, or compares facial landmarks, visual Re-ID descriptors, clothing embeddings, or persistent worker identities.
3. **Stream Isolation & Lifecycle Cleanup**:
   - `CompliancePipelineService` maintains LRU-bounded per-`stream_id` instances (max `32` concurrent streams) with `POST /compliance/reset` to purge active tracks immediately.

---

## 2. Two-Stage ByteTrack Lifecycle & Missing/Lost Track Handling

1. **Kinematic Motion Prediction**:
   - Each step advances `(cx, cy, w, h)` using exponentially smoothed velocity (`velocity_alpha=0.4`) with decay (`velocity_damping=0.75`), incrementing `age` and `time_since_update`.
2. **Stage 1 (High-Confidence Association)**:
   - Person detections with `confidence >= high_conf_threshold` (`0.45`) are matched to retained tracks using greedy bipartite IoU assignment gated by `match_iou_threshold` (`0.25`).
3. **Stage 2 (Low-Confidence Occlusion Recovery)**:
   - Unmatched tracks (including temporarily occluded workers whose detector score dips) are matched against detections in `[low_conf_threshold, high_conf_threshold)` (`[0.15, 0.45)`) using relaxed `second_stage_iou_threshold` (`0.15`), preserving the temporary `track_id`.
4. **Track States (`active`, `tentative`, `lost`, `retired`)**:
   - `active`: matched on the current frame (`time_since_update == 0`) with `hits >= min_hits`.
   - `tentative`: matched on current frame (`time_since_update == 0`) before reaching `min_hits`.
   - `lost`: retained in memory up to `max_age` frames (`15` default) while occluded/missing; exposed via `TrackingStepResult.lost_tracks` and `get_lost_tracks()`, without fabricating synthetic active worker detections.
   - `retired`: permanently expired once `time_since_update > max_age`; reported in `TrackingStepResult.retired_track_ids` and never reused.

---

## 3. Configuration Parameters (`TrackerConfig` / `SAFEGUARD_TRACKER_*`)

| Parameter | Default | Role |
|---|---:|---|
| `high_conf_threshold` | `0.45` | Minimum confidence for Stage-1 match and new-track initialization. |
| `low_conf_threshold` | `0.15` | Lower confidence bound for Stage-2 ByteTrack occlusion recovery. |
| `match_iou_threshold` | `0.25` | Minimum IoU in Stage-1 matching. |
| `second_stage_iou_threshold` | `0.15` | Relaxed IoU gate in Stage-2 low-confidence recovery. |
| `max_age` | `15` | Maximum missed frames before retiring an occluded track. |
| `min_hits` | `1` | Minimum hits before reporting a track as `active`. |

---

## 4. Known Tracking Limitations

1. **Identity Switches Under Heavy Crossover or Long Occlusion**:
   - Because SafeGuard deliberately prohibits visual/facial Re-ID embeddings to protect worker privacy, ID continuity relies purely on geometric bounding-box IoU and damped linear motion.
   - When two workers cross paths tightly and swap screen positions, or when a worker disappears behind machinery for longer than `max_age` frames, the tracker cannot visually confirm identity and will either switch IDs or mint a fresh temporary `track_id` upon reappearance.
2. **Sensitivity to Sampling Rate & Camera Motion**:
   - Low webcam/API polling rates (e.g., 2–5 FPS during congested CPU inference) increase frame-to-frame spatial displacement relative to bounding-box width, reducing IoU below `match_iou_threshold` for briskly walking workers.
   - Sudden pan/tilt camera movement shifts all scene boxes simultaneously and violates the stationary-camera constant-velocity assumption, causing transient track fragmentation.
3. **Detector Fragmentation & Crowding**:
   - NMS jitter or split partial-person boxes in dense scenes can spawn short-lived secondary tracks when a partial silhouette exceeds `high_conf_threshold`.
4. **Single-Stream Scope**:
   - Multi-camera handoff and cross-session identity linking are intentionally unsupported.
