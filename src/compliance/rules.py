"""Stage 1 & 3: Taxonomy validation, spatial PPE association, and compliance rules."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import yaml

from src.compliance.alerts import AlertConfig, AlertManager, ComplianceEvent
from src.compliance.analytics import StreamAnalyticsSnapshot, StreamAnalyticsTracker
from src.tracking.tracker import ByteTrackTracker, Track, TrackDetection

_DEFAULT_APPROVED_CLASSES: frozenset[str] = frozenset(
    {"person", "helmet", "vest", "gloves", "no_helmet"}
)
_DEFAULT_CLASS_MAP_PATH: Path = (
    Path(__file__).resolve().parents[2] / "configs" / "class_mapping.yaml"
)


@lru_cache(maxsize=4)
def load_approved_class_names(mapping_path: str | None = None) -> frozenset[str]:
    """Load approved canonical class names strictly from configs/class_mapping.yaml."""
    path = Path(mapping_path) if mapping_path else _DEFAULT_CLASS_MAP_PATH
    if not path.exists():
        return _DEFAULT_APPROVED_CLASSES

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    approved = {
        str(entry["name"])
        for entry in data.get("canonical_classes", [])
        if isinstance(entry, dict)
        and entry.get("status") == "approved"
        and "name" in entry
    }
    return frozenset(approved) if approved else _DEFAULT_APPROVED_CLASSES


@dataclass(frozen=True)
class ComplianceConfig:
    """Configurable thresholds and temporal policy for SafeGuard compliance rules."""

    conf_person: float = 0.25
    conf_helmet: float = 0.25
    conf_vest: float = 0.25
    conf_gloves: float = 0.25
    conf_no_helmet: float = 0.25
    ambiguous_margin: float = 0.15
    spatial_ambiguity_margin: float = 0.08
    min_head_containment: float = 0.45
    min_torso_containment: float = 0.40
    min_gloves_containment: float = 0.35
    max_head_person_area_ratio: float = 0.55
    min_person_height_px: float = 48.0
    min_person_width_px: float = 20.0
    edge_clip_margin_px: float = 3.0
    confirm_frames: int = 3
    cooldown_seconds: float = 10.0
    max_event_history: int = 200
    capture_evidence: bool = True
    class_mapping_path: str | None = None


@dataclass(frozen=True)
class DetectionInput:
    """Stage 1 raw detector-agnostic canonical detection input."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class PersonPPEAssociation:
    """Stage 3 per-person spatial PPE association and compliance assessment."""

    track_id: int
    person_confidence: float
    bbox_xyxy: list[float]
    helmet_boxes: list[DetectionInput]
    no_helmet_boxes: list[DetectionInput]
    vest_boxes: list[DetectionInput]
    gloves_boxes: list[DetectionInput]
    helmet_status: str
    vest_status: str
    overall_status: str
    uncertainty_reasons: list[str]
    violation_streak: int
    visibility_status: str = "sufficient"


@dataclass(frozen=True)
class ComplianceInterpretationResult:
    """Pure Stage 3 compliance interpretation output decoupled from alerts."""

    associations: list[PersonPPEAssociation]
    unmatched_ppe: list[DetectionInput]
    valid_detections: list[DetectionInput]
    invalid_detections_filtered: int
    unapproved_detections_filtered: int


@dataclass(frozen=True)
class ComplianceFrameResult:
    """Full pipeline frame output separating detections, tracks, compliance, alerts."""

    frame_index: int
    timestamp: float
    tracks: list[Track]
    associations: list[PersonPPEAssociation]
    unmatched_ppe: list[DetectionInput]
    events: list[ComplianceEvent]
    suppressed_duplicates: int
    invalid_detections_filtered: int
    lost_tracks: list[Track] = field(default_factory=list)
    retired_track_ids: list[int] = field(default_factory=list)
    unapproved_detections_filtered: int = 0


def _is_valid_detection_input(det: DetectionInput) -> bool:
    if len(det.bbox_xyxy) != 4 or not math.isfinite(det.confidence):
        return False
    if not (0.0 <= det.confidence <= 1.0):
        return False
    x1, y1, x2, y2 = det.bbox_xyxy
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    return x2 > x1 and y2 > y1


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _containment_ratio(inner: list[float], region: list[float]) -> float:
    ix1 = max(inner[0], region[0])
    iy1 = max(inner[1], region[1])
    ix2 = min(inner[2], region[2])
    iy2 = min(inner[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_inner = _box_area(inner)
    return inter / area_inner if area_inner > 0.0 else 0.0


def _head_zone(person_box: list[float]) -> list[float]:
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1
    return [px1 - 0.12 * pw, py1 - 0.18 * ph, px2 + 0.12 * pw, py1 + 0.42 * ph]


def _torso_zone(person_box: list[float]) -> list[float]:
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1
    return [px1 - 0.06 * pw, py1 + 0.10 * ph, px2 + 0.06 * pw, py1 + 0.80 * ph]


def _gloves_zone(person_box: list[float]) -> list[float]:
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1
    return [px1 - 0.18 * pw, py1 + 0.10 * ph, px2 + 0.18 * pw, py2 + 0.06 * ph]


def _compute_spatial_score(
    ppe: DetectionInput,
    track: Track,
    zone_fn: Callable[[list[float]], list[float]],
    min_containment: float,
    max_area_ratio: float | None,
) -> float | None:
    p_area = _box_area(track.bbox_xyxy)
    ppe_area = _box_area(ppe.bbox_xyxy)
    if max_area_ratio is not None and p_area > 0.0:
        if (ppe_area / p_area) > max_area_ratio:
            return None

    zone = zone_fn(track.bbox_xyxy)
    containment = _containment_ratio(ppe.bbox_xyxy, zone)
    if containment < min_containment:
        return None

    zcx = (zone[0] + zone[2]) * 0.5
    zcy = (zone[1] + zone[3]) * 0.5
    pcx = (ppe.bbox_xyxy[0] + ppe.bbox_xyxy[2]) * 0.5
    pcy = (ppe.bbox_xyxy[1] + ppe.bbox_xyxy[3]) * 0.5
    zw = max(1.0, zone[2] - zone[0])
    zh = max(1.0, zone[3] - zone[1])
    norm_dist = math.hypot((pcx - zcx) / zw, (pcy - zcy) / zh)
    return containment - 0.15 * norm_dist


def _optimal_bipartite_assign(scores: np.ndarray) -> list[tuple[int, int]]:
    """Solve small 1-to-1 max-score assignment between tracks (rows) and PPE (cols)."""
    num_tracks, num_ppe = scores.shape
    if num_tracks == 0 or num_ppe == 0:
        return []

    n = max(num_tracks, num_ppe)
    pad = np.full((n, n), -1e6, dtype=np.float64)
    for r in range(num_tracks):
        for c in range(num_ppe):
            if np.isfinite(scores[r, c]):
                pad[r, c] = float(scores[r, c])

    max_val = max(0.0, float(np.max(pad)))
    cost = max_val - pad

    # Jonker-Volgenant / Hungarian O(n^3) shortest augmenting path solver.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)
    way = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, float("inf"), dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = float(minv[j])
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        r = int(p[j]) - 1
        c = j - 1
        if 0 <= r < num_tracks and 0 <= c < num_ppe and np.isfinite(scores[r, c]):
            pairs.append((r, c))
    return pairs


def _assign_ppe_with_overlap_guard(
    ppe_list: list[DetectionInput],
    tracks: list[Track],
    zone_fn: Callable[[list[float]], list[float]],
    min_containment: float,
    max_area_ratio: float | None = None,
    allow_multiple_per_track: bool = False,
    ambiguity_margin: float = 0.08,
) -> tuple[dict[int, list[DetectionInput]], list[DetectionInput], set[int]]:
    """Associate PPE boxes to tracked persons while guarding ambiguous overlaps."""
    assigned: dict[int, list[DetectionInput]] = {t.track_id: [] for t in tracks}
    unmatched: list[DetectionInput] = []
    ambiguous_track_ids: set[int] = set()

    if not ppe_list or not tracks:
        return assigned, list(ppe_list), ambiguous_track_ids

    num_t, num_p = len(tracks), len(ppe_list)
    score_mat = np.full((num_t, num_p), -np.inf, dtype=np.float64)
    for t_idx, trk in enumerate(tracks):
        for p_idx, ppe in enumerate(ppe_list):
            s = _compute_spatial_score(
                ppe, trk, zone_fn, min_containment, max_area_ratio
            )
            if s is not None:
                score_mat[t_idx, p_idx] = s

    if allow_multiple_per_track:
        for p_idx, ppe in enumerate(ppe_list):
            col = score_mat[:, p_idx]
            valid_rows = [r for r in range(num_t) if np.isfinite(col[r])]
            if not valid_rows:
                unmatched.append(ppe)
                continue
            best_row = max(valid_rows, key=lambda r: float(col[r]))
            assigned[tracks[best_row].track_id].append(ppe)
        return assigned, unmatched, ambiguous_track_ids

    matched_pairs = _optimal_bipartite_assign(score_mat)
    matched_t_set = {r for r, _ in matched_pairs}
    matched_p_set = {c for _, c in matched_pairs}

    for t_idx, p_idx in matched_pairs:
        win_score = float(score_mat[t_idx, p_idx])
        # Check if an unmatched overlapping worker competes within ambiguity_margin.
        competing = [
            other_t
            for other_t in range(num_t)
            if other_t != t_idx
            and other_t not in matched_t_set
            and np.isfinite(score_mat[other_t, p_idx])
            and (win_score - float(score_mat[other_t, p_idx])) <= ambiguity_margin
        ]
        if competing:
            ambiguous_track_ids.add(tracks[t_idx].track_id)
            for other_t in competing:
                ambiguous_track_ids.add(tracks[other_t].track_id)
            unmatched.append(ppe_list[p_idx])
        else:
            assigned[tracks[t_idx].track_id].append(ppe_list[p_idx])

    for p_idx in range(num_p):
        if p_idx not in matched_p_set:
            unmatched.append(ppe_list[p_idx])

    return assigned, unmatched, ambiguous_track_ids


def _assess_visibility(
    track: Track,
    config: ComplianceConfig,
    frame_size: tuple[int, int] | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    x1, y1, x2, y2 = track.bbox_xyxy
    w = x2 - x1
    h = y2 - y1

    if track.state == "lost":
        reasons.append("track_currently_occluded")
    if h < config.min_person_height_px or w < config.min_person_width_px:
        reasons.append("person_box_too_small_for_reliable_ppe")
    if frame_size is not None:
        fw, fh = frame_size
        margin = config.edge_clip_margin_px
        if y1 <= margin and h < 0.75 * fh:
            reasons.append("head_region_clipped_at_frame_edge")
        elif x1 <= margin or x2 >= (fw - margin):
            if w < config.min_person_width_px * 1.5:
                reasons.append("worker_partially_out_of_frame")

    if reasons:
        return "insufficient_visibility", reasons
    return "sufficient", []


class ComplianceInterpreter:
    """Stage 3 stateless spatial PPE association and compliance interpretation."""

    def __init__(self, config: ComplianceConfig | None = None) -> None:
        self.config = config or ComplianceConfig()
        self.approved_classes = load_approved_class_names(
            self.config.class_mapping_path
        )

    def evaluate(
        self,
        raw_detections: list[DetectionInput],
        active_tracks: list[Track],
        lost_tracks: list[Track] | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ComplianceInterpretationResult:
        """Filter to approved taxonomy, associate PPE cleanly, and classify states."""
        valid_dets: list[DetectionInput] = []
        invalid_count = 0
        unapproved_count = 0

        for det in raw_detections:
            if not _is_valid_detection_input(det):
                invalid_count += 1
                continue
            if det.class_name not in self.approved_classes:
                unapproved_count += 1
                continue
            valid_dets.append(det)

        eligible_tracks = [
            t for t in active_tracks if t.confidence >= self.config.conf_person
        ]

        visibility_by_tid: dict[int, tuple[str, list[str]]] = {}
        visible_tracks: list[Track] = []
        for trk in eligible_tracks:
            vis_status, vis_reasons = _assess_visibility(trk, self.config, frame_size)
            visibility_by_tid[trk.track_id] = (vis_status, vis_reasons)
            if vis_status == "sufficient":
                visible_tracks.append(trk)

        helmets = [
            d
            for d in valid_dets
            if d.class_name == "helmet" and d.confidence >= self.config.conf_helmet
        ]
        no_helmets = [
            d
            for d in valid_dets
            if d.class_name == "no_helmet"
            and d.confidence >= self.config.conf_no_helmet
        ]
        vests = [
            d
            for d in valid_dets
            if d.class_name == "vest" and d.confidence >= self.config.conf_vest
        ]
        gloves = [
            d
            for d in valid_dets
            if d.class_name == "gloves" and d.confidence >= self.config.conf_gloves
        ]

        h_map, unm_h, amb_h = _assign_ppe_with_overlap_guard(
            helmets,
            visible_tracks,
            _head_zone,
            self.config.min_head_containment,
            max_area_ratio=self.config.max_head_person_area_ratio,
            ambiguity_margin=self.config.spatial_ambiguity_margin,
        )
        nh_map, unm_nh, amb_nh = _assign_ppe_with_overlap_guard(
            no_helmets,
            visible_tracks,
            _head_zone,
            self.config.min_head_containment,
            max_area_ratio=self.config.max_head_person_area_ratio,
            ambiguity_margin=self.config.spatial_ambiguity_margin,
        )
        v_map, unm_v, amb_v = _assign_ppe_with_overlap_guard(
            vests,
            visible_tracks,
            _torso_zone,
            self.config.min_torso_containment,
            ambiguity_margin=self.config.spatial_ambiguity_margin,
        )
        g_map, unm_g, _ = _assign_ppe_with_overlap_guard(
            gloves,
            visible_tracks,
            _gloves_zone,
            self.config.min_gloves_containment,
            allow_multiple_per_track=True,
        )
        unmatched_ppe = unm_h + unm_nh + unm_v + unm_g

        associations: list[PersonPPEAssociation] = []
        for trk in eligible_tracks:
            tid = trk.track_id
            vis_status, vis_reasons = visibility_by_tid[tid]
            if vis_status != "sufficient":
                associations.append(
                    PersonPPEAssociation(
                        track_id=tid,
                        person_confidence=trk.confidence,
                        bbox_xyxy=list(trk.bbox_xyxy),
                        helmet_boxes=[],
                        no_helmet_boxes=[],
                        vest_boxes=[],
                        gloves_boxes=[],
                        helmet_status="insufficient_visibility",
                        vest_status="insufficient_visibility",
                        overall_status="insufficient_visibility",
                        uncertainty_reasons=vis_reasons,
                        violation_streak=0,
                        visibility_status=vis_status,
                    )
                )
                continue

            h_boxes = h_map.get(tid, [])
            nh_boxes = nh_map.get(tid, [])
            v_boxes = v_map.get(tid, [])
            g_boxes = g_map.get(tid, [])
            reasons: list[str] = []

            max_h = max((b.confidence for b in h_boxes), default=0.0)
            max_nh = max((b.confidence for b in nh_boxes), default=0.0)
            head_overlap_ambiguous = (tid in amb_h or tid in amb_nh) and not (
                h_boxes or nh_boxes
            )

            if head_overlap_ambiguous:
                helmet_status = "uncertain"
                reasons.append("ambiguous_spatial_overlap")
            elif h_boxes and nh_boxes:
                if abs(max_h - max_nh) <= self.config.ambiguous_margin:
                    helmet_status = "ambiguous"
                    reasons.append("conflicting_helmet_and_no_helmet")
                elif max_h > max_nh:
                    helmet_status = "compliant"
                else:
                    helmet_status = "violation"
            elif nh_boxes:
                helmet_status = "violation"
            elif h_boxes:
                helmet_status = "compliant"
            else:
                helmet_status = "unknown"
                reasons.append("head_ppe_not_observed")

            if v_boxes:
                vest_status = "compliant"
            elif tid in amb_v:
                vest_status = "uncertain"
                reasons.append("ambiguous_torso_overlap")
            else:
                # Strict rule: absent vest != violation without trained no_vest.
                vest_status = "unknown"
                reasons.append("vest_not_detected_no_vest_class_untrained")

            if helmet_status == "violation":
                overall_status = "violation_candidate"
            elif helmet_status == "uncertain" or vest_status == "uncertain":
                overall_status = "uncertain"
            elif helmet_status == "compliant" and vest_status == "compliant":
                overall_status = "compliant"
            else:
                overall_status = "unknown"

            associations.append(
                PersonPPEAssociation(
                    track_id=tid,
                    person_confidence=trk.confidence,
                    bbox_xyxy=list(trk.bbox_xyxy),
                    helmet_boxes=h_boxes,
                    no_helmet_boxes=nh_boxes,
                    vest_boxes=v_boxes,
                    gloves_boxes=g_boxes,
                    helmet_status=helmet_status,
                    vest_status=vest_status,
                    overall_status=overall_status,
                    uncertainty_reasons=reasons,
                    violation_streak=0,
                    visibility_status=vis_status,
                )
            )

        _ = lost_tracks
        return ComplianceInterpretationResult(
            associations=associations,
            unmatched_ppe=unmatched_ppe,
            valid_detections=valid_dets,
            invalid_detections_filtered=invalid_count,
            unapproved_detections_filtered=unapproved_count,
        )


@dataclass
class ComplianceEngine:
    """Orchestrates Raw -> Tracking -> Compliance -> Alerts + Stream Analytics."""

    config: ComplianceConfig = field(default_factory=ComplianceConfig)
    tracker: ByteTrackTracker = field(default_factory=ByteTrackTracker)
    interpreter: ComplianceInterpreter = field(init=False)
    alert_manager: AlertManager = field(init=False)
    analytics: StreamAnalyticsTracker = field(
        default_factory=StreamAnalyticsTracker, init=False
    )
    _frame_index: int = field(default=0, init=False)
    _violation_streaks: dict[int, int] = field(default_factory=dict, init=False)
    _last_seen_tracker_frame: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.interpreter = ComplianceInterpreter(config=self.config)
        self.alert_manager = AlertManager(
            config=AlertConfig(
                cooldown_seconds=self.config.cooldown_seconds,
                max_event_history=self.config.max_event_history,
                capture_evidence=self.config.capture_evidence,
            )
        )

    def reset(self) -> None:
        """Reset tracker, confirmation streaks, alert history, and analytics."""
        self.tracker.reset()
        self.alert_manager.reset()
        self.analytics.reset()
        self._frame_index = 0
        self._violation_streaks.clear()
        self._last_seen_tracker_frame.clear()

    @property
    def event_history(self) -> list[ComplianceEvent]:
        """Bounded rolling list of recent potential compliance events."""
        return self.alert_manager.event_history

    def get_alerts(
        self, status: str | None = None, limit: int = 100
    ) -> list[ComplianceEvent]:
        """Return recent alerts optionally filtered by event_status."""
        return self.alert_manager.list_events(status=status, limit=limit)

    def get_alert(self, event_id: str) -> ComplianceEvent | None:
        """Retrieve a single retained alert by event_id."""
        return self.alert_manager.get_event(event_id)

    def update_alert_status(
        self, event_id: str, new_status: str
    ) -> ComplianceEvent | None:
        """Update human review status on a recorded alert and sync analytics."""
        updated = self.alert_manager.update_status(event_id, new_status)
        if updated is not None:
            self.analytics.sync_alert_status(event_id, new_status)
        return updated

    def get_analytics(self) -> StreamAnalyticsSnapshot:
        """Return cumulative stream tracking, compliance, and alert analytics."""
        return self.analytics.snapshot()

    def process_frame(
        self,
        detections: list[DetectionInput],
        timestamp: float | None = None,
        frame_bgr: np.ndarray | None = None,
        frame_size: tuple[int, int] | None = None,
    ) -> ComplianceFrameResult:
        """Run Raw -> Tracking -> Compliance -> Alerts + Analytics pipeline."""
        self._frame_index += 1
        ts = time.time() if timestamp is None else float(timestamp)
        effective_size = frame_size
        if effective_size is None and frame_bgr is not None and frame_bgr.ndim >= 2:
            effective_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))

        # Stage 1: Raw taxonomy/geometry check before tracking persons.
        approved_classes = self.interpreter.approved_classes
        person_track_inputs = [
            TrackDetection(
                bbox_xyxy=d.bbox_xyxy,
                confidence=d.confidence,
                class_id=d.class_id,
            )
            for d in detections
            if _is_valid_detection_input(d)
            and d.class_name in approved_classes
            and d.class_name == "person"
            and d.confidence >= self.tracker.config.low_conf_threshold
        ]

        # Stage 2: Temporary anonymous ByteTrack person tracking.
        track_step = self.tracker.step(person_track_inputs)
        eligible_tracks = [
            t
            for t in track_step.active_tracks
            if t.confidence >= self.config.conf_person
        ]

        # Stage 3: Spatial PPE association & rule interpretation.
        interp = self.interpreter.evaluate(
            raw_detections=detections,
            active_tracks=track_step.active_tracks,
            lost_tracks=track_step.lost_tracks,
            frame_size=effective_size,
        )

        # Clear streak state for retired or missing tracks so gaps reset.
        for retired_id in track_step.retired_track_ids:
            self._violation_streaks.pop(retired_id, None)
            self._last_seen_tracker_frame.pop(retired_id, None)
        for lost in track_step.lost_tracks:
            self._violation_streaks[lost.track_id] = 0

        resolved_associations: list[PersonPPEAssociation] = []
        for assoc in interp.associations:
            tid = assoc.track_id
            prev_trk_frame = self._last_seen_tracker_frame.get(tid)
            had_missing_gap = (
                prev_trk_frame is not None
                and (track_step.frame_index - prev_trk_frame) > 1
            )
            self._last_seen_tracker_frame[tid] = track_step.frame_index
            if had_missing_gap:
                self._violation_streaks[tid] = 0

            if assoc.visibility_status != "sufficient":
                self._violation_streaks[tid] = 0
                resolved_associations.append(assoc)
                continue

            if assoc.helmet_status == "violation":
                streak = self._violation_streaks.get(tid, 0) + 1
                self._violation_streaks[tid] = streak
                if streak >= self.config.confirm_frames:
                    final_status = "potential_violation"
                    reasons = list(assoc.uncertainty_reasons)
                else:
                    final_status = "pending_confirmation"
                    req = self.config.confirm_frames
                    reasons = list(assoc.uncertainty_reasons) + [
                        f"awaiting_temporal_confirmation_{streak}_of_{req}"
                    ]
                resolved_associations.append(
                    replace(
                        assoc,
                        overall_status=final_status,
                        uncertainty_reasons=reasons,
                        violation_streak=streak,
                    )
                )
            else:
                self._violation_streaks[tid] = 0
                resolved_associations.append(assoc)

        # Stage 4: Deduplicated alert emission with annotated evidence & analytics.
        emitted_events, suppressed_duplicates = self.alert_manager.process_associations(
            associations=resolved_associations,
            frame_index=self._frame_index,
            timestamp=ts,
            frame_bgr=frame_bgr,
        )
        self.analytics.record_frame(
            valid_detections=interp.valid_detections,
            active_tracks=eligible_tracks,
            lost_tracks=track_step.lost_tracks,
            associations=resolved_associations,
            emitted_events=emitted_events,
            suppressed_duplicates=suppressed_duplicates,
        )

        return ComplianceFrameResult(
            frame_index=self._frame_index,
            timestamp=ts,
            tracks=eligible_tracks,
            associations=resolved_associations,
            unmatched_ppe=interp.unmatched_ppe,
            events=emitted_events,
            suppressed_duplicates=suppressed_duplicates,
            invalid_detections_filtered=interp.invalid_detections_filtered,
            lost_tracks=track_step.lost_tracks,
            retired_track_ids=track_step.retired_track_ids,
            unapproved_detections_filtered=interp.unapproved_detections_filtered,
        )
