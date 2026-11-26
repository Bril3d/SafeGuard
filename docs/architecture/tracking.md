# SafeGuard Anonymous Multi-Object Tracking Architecture

**Module**: `src/tracking/tracker.py`  
**Pattern**: Pure-Python two-stage ByteTrack-style IoU + constant-velocity tracker

---

## 1. Design Principles & Privacy Boundaries

1. **Zero Biometrics & Zero PII**:
   - Tracks strictly anonymous integer `track_id` values (`1, 2, 3, ...`) assigned per `stream_id` in memory.
   - `Track` and `_InternalTrackState` store **only** bounding-box geometry (`cx, cy, w, h`), damped velocity (`vx, vy, vw, vh`), confidence, frame age/hit counters, and `time_since_update`.
   - No cropped worker images, facial features, appearance embeddings, or cross-session persistent profiles are extracted or stored.
2. **Inference Independence**:
   - `ByteTrackTracker.update(list[TrackDetection])` accepts generic bounding-box + confidence inputs (`bbox_xyxy`, `confidence`, `class_id`) and contains zero imports from `ultralytics`, `torch`, `onnxruntime`, or `backend/`.
3. **Stream Isolation**:
   - `CompliancePipelineService` manages bounded per-`stream_id` tracker instances (LRU cap `32` streams) protected by per-stream locks, with `POST /compliance/reset` to clear temporary session IDs on demand.

---

## 2. Two-Stage ByteTrack Association Lifecycle

1. **Motion Prediction**: At each frame step, existing tracks advance `(cx, cy, w, h)` using exponentially smoothed velocity (`velocity_alpha=0.4`) with decay (`velocity_damping=0.75`), incrementing `age` and `time_since_update`.
2. **Stage 1 (High-Confidence Matching)**:
   - Person detections with `confidence >= high_conf_threshold` (`0.45`) are matched against active/lost tracks using greedy bipartite IoU assignment gated by `match_iou_threshold` (`0.25`).
3. **Stage 2 (Low-Confidence Occlusion Recovery)**:
   - Unmatched tracks (including temporarily occluded workers whose detector confidence dips) are matched against detections in `[low_conf_threshold, high_conf_threshold)` (`[0.15, 0.45)`) using `second_stage_iou_threshold` (`0.15`). Matched tracks retain their exact `track_id`.
4. **Track Birth & Retirement**:
   - Remaining unmatched high-confidence detections spawn new tracks (low-confidence detections never spawn standalone tracks, suppressing background noise).
   - Tracks unmatched for `time_since_update > max_age` (`15` frames default) are deleted; expired IDs are never reused.

---

## 3. Configuration Parameters (`TrackerConfig` / `SAFEGUARD_TRACKER_*`)

| Parameter | Default | Role |
|---|---:|---|
| `high_conf_threshold` | `0.45` | Minimum confidence for Stage-1 match and new-track initialization. |
| `low_conf_threshold` | `0.15` | Lower confidence bound for Stage-2 ByteTrack occlusion recovery. |
| `match_iou_threshold` | `0.25` | Minimum IoU in Stage-1 matching. |
| `second_stage_iou_threshold` | `0.15` | Relaxed IoU gate in Stage-2 low-confidence recovery. |
| `max_age` | `15` | Maximum missed frames before retiring an occluded track. |
| `min_hits` | `1` | Minimum hits required before reporting a track as `active`. |

---

## 4. Known Limitations

- Pure geometric IoU + constant-velocity tracking can switch anonymous IDs after long occlusions (`> max_age`), abrupt camera cuts, or severe crossing trajectories where workers swap spatial positions without visual Re-ID (deliberately omitted to avoid biometric/appearance retention).
- Multi-camera cross-view re-identification is out of scope by design.
