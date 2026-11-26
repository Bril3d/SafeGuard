"""Person-to-PPE spatial association, safety rules, and temporal alert logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

from src.tracking.tracker import ByteTrackTracker, Track, TrackDetection


@dataclass(frozen=True)
class ComplianceConfig:
    """Configurable thresholds and temporal policy for SafeGuard compliance rules."""

    conf_person: float = 0.25
    conf_helmet: float = 0.25
    conf_vest: float = 0.25
    conf_gloves: float = 0.25
    conf_no_helmet: float = 0.25
    ambiguous_margin: float = 0.15
    min_head_containment: float = 0.45
    min_torso_containment: float = 0.40
    min_gloves_containment: float = 0.35
    max_head_person_area_ratio: float = 0.55
    confirm_frames: int = 3
    cooldown_seconds: float = 10.0
    max_event_history: int = 200


@dataclass(frozen=True)
class DetectionInput:
    """Detector-agnostic canonical detection input."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


@dataclass(frozen=True)
class PersonPPEAssociation:
    """Per-person spatial PPE association and compliance assessment."""

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


@dataclass(frozen=True)
class ComplianceEvent:
    """Derived safety decision-support event requiring human review."""

    event_id: str
    event_type: str
    severity: str
    track_id: int
    timestamp: float
    frame_index: int
    confirmed_frames: int
    confidence: float
    bbox_xyxy: list[float]
    requires_human_review: bool
    message: str


@dataclass(frozen=True)
class ComplianceFrameResult:
    """Frame compliance output separating tracks, associations, and alerts."""

    frame_index: int
    timestamp: float
    tracks: list[Track]
    associations: list[PersonPPEAssociation]
    unmatched_ppe: list[DetectionInput]
    events: list[ComplianceEvent]
    suppressed_duplicates: int
    invalid_detections_filtered: int


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


def _assign_ppe_to_best_person(
    ppe_list: list[DetectionInput],
    tracks: list[Track],
    zone_fn,
    min_containment: float,
    max_area_ratio: float | None = None,
) -> tuple[dict[int, list[DetectionInput]], list[DetectionInput]]:
    assigned: dict[int, list[DetectionInput]] = {t.track_id: [] for t in tracks}
    unmatched: list[DetectionInput] = []

    for ppe in ppe_list:
        ppe_area = _box_area(ppe.bbox_xyxy)
        best_track_id: int | None = None
        best_score = -1.0

        for trk in tracks:
            p_area = _box_area(trk.bbox_xyxy)
            if max_area_ratio is not None and p_area > 0.0:
                if (ppe_area / p_area) > max_area_ratio:
                    continue
            zone = zone_fn(trk.bbox_xyxy)
            containment = _containment_ratio(ppe.bbox_xyxy, zone)
            if containment < min_containment:
                continue
            # Prefer higher containment, breaking ties by center proximity.
            zcx = (zone[0] + zone[2]) * 0.5
            zcy = (zone[1] + zone[3]) * 0.5
            pcx = (ppe.bbox_xyxy[0] + ppe.bbox_xyxy[2]) * 0.5
            pcy = (ppe.bbox_xyxy[1] + ppe.bbox_xyxy[3]) * 0.5
            zw = max(1.0, zone[2] - zone[0])
            zh = max(1.0, zone[3] - zone[1])
            norm_dist = math.hypot((pcx - zcx) / zw, (pcy - zcy) / zh)
            score = containment - 0.15 * norm_dist
            if score > best_score:
                best_score = score
                best_track_id = trk.track_id

        if best_track_id is not None:
            assigned[best_track_id].append(ppe)
        else:
            unmatched.append(ppe)

    return assigned, unmatched


@dataclass
class ComplianceEngine:
    """Stateful compliance rules engine with temporal confirmation and cooldown."""

    config: ComplianceConfig = field(default_factory=ComplianceConfig)
    tracker: ByteTrackTracker = field(default_factory=ByteTrackTracker)
    _frame_index: int = field(default=0, init=False)
    _violation_streaks: dict[int, int] = field(default_factory=dict, init=False)
    _last_alert_ts: dict[tuple[int, str], float] = field(
        default_factory=dict, init=False
    )
    _event_counter: int = field(default=0, init=False)
    _event_history: list[ComplianceEvent] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Reset tracker, confirmation streaks, cooldown timers, and event history."""
        self.tracker.reset()
        self._frame_index = 0
        self._violation_streaks.clear()
        self._last_alert_ts.clear()
        self._event_counter = 0
        self._event_history.clear()

    @property
    def event_history(self) -> list[ComplianceEvent]:
        """Bounded rolling list of recent potential compliance events."""
        return list(self._event_history)

    def process_frame(
        self,
        detections: list[DetectionInput],
        timestamp: float | None = None,
    ) -> ComplianceFrameResult:
        """Run tracking, PPE association, compliance evaluation, and event emission."""
        self._frame_index += 1
        ts = time.time() if timestamp is None else float(timestamp)

        valid_dets: list[DetectionInput] = []
        invalid_count = 0
        for det in detections:
            if _is_valid_detection_input(det):
                valid_dets.append(det)
            else:
                invalid_count += 1

        person_track_inputs = [
            TrackDetection(
                bbox_xyxy=d.bbox_xyxy,
                confidence=d.confidence,
                class_id=d.class_id,
            )
            for d in valid_dets
            if d.class_name == "person"
            and d.confidence >= self.tracker.config.low_conf_threshold
        ]
        tracks = self.tracker.update(person_track_inputs)
        # Filter active tracks by minimum person compliance confidence if configured.
        eligible_tracks = [t for t in tracks if t.confidence >= self.config.conf_person]

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

        h_map, unm_h = _assign_ppe_to_best_person(
            helmets,
            eligible_tracks,
            _head_zone,
            self.config.min_head_containment,
            max_area_ratio=self.config.max_head_person_area_ratio,
        )
        nh_map, unm_nh = _assign_ppe_to_best_person(
            no_helmets,
            eligible_tracks,
            _head_zone,
            self.config.min_head_containment,
            max_area_ratio=self.config.max_head_person_area_ratio,
        )
        v_map, unm_v = _assign_ppe_to_best_person(
            vests, eligible_tracks, _torso_zone, self.config.min_torso_containment
        )
        g_map, unm_g = _assign_ppe_to_best_person(
            gloves, eligible_tracks, _gloves_zone, self.config.min_gloves_containment
        )
        unmatched_ppe = unm_h + unm_nh + unm_v + unm_g

        associations: list[PersonPPEAssociation] = []
        emitted_events: list[ComplianceEvent] = []
        suppressed_duplicates = 0

        active_ids = {t.track_id for t in eligible_tracks}
        for stale_id in [k for k in self._violation_streaks if k not in active_ids]:
            # Decay streak when a track disappears or is temporarily occluded.
            if self.tracker.active_track_count == 0 or stale_id not in {
                trk.track_id for trk in self.tracker._tracks
            }:
                self._violation_streaks.pop(stale_id, None)

        for trk in eligible_tracks:
            tid = trk.track_id
            h_boxes = h_map.get(tid, [])
            nh_boxes = nh_map.get(tid, [])
            v_boxes = v_map.get(tid, [])
            g_boxes = g_map.get(tid, [])
            reasons: list[str] = []

            max_h = max((b.confidence for b in h_boxes), default=0.0)
            max_nh = max((b.confidence for b in nh_boxes), default=0.0)

            if h_boxes and nh_boxes:
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
            else:
                # Strict rule: absent vest != violation without trained no_vest.
                vest_status = "unknown"
                reasons.append("vest_not_detected_no_vest_class_untrained")

            if helmet_status == "violation":
                streak = self._violation_streaks.get(tid, 0) + 1
                self._violation_streaks[tid] = streak
                if streak >= self.config.confirm_frames:
                    overall_status = "potential_violation"
                    alert_key = (tid, "potential_no_helmet")
                    last_ts = self._last_alert_ts.get(alert_key)
                    if (
                        last_ts is None
                        or (ts - last_ts) >= self.config.cooldown_seconds
                    ):
                        self._event_counter += 1
                        self._last_alert_ts[alert_key] = ts
                        evt = ComplianceEvent(
                            event_id=f"evt-{self._event_counter:06d}",
                            event_type="potential_no_helmet",
                            severity="high",
                            track_id=tid,
                            timestamp=ts,
                            frame_index=self._frame_index,
                            confirmed_frames=streak,
                            confidence=round(max_nh, 4),
                            bbox_xyxy=list(trk.bbox_xyxy),
                            requires_human_review=True,
                            message=(
                                f"Potential missing helmet on track #{tid} "
                                f"(confirmed across {streak} frames; review required)."
                            ),
                        )
                        emitted_events.append(evt)
                        self._event_history.append(evt)
                        if len(self._event_history) > self.config.max_event_history:
                            self._event_history = self._event_history[
                                -self.config.max_event_history :
                            ]
                    else:
                        suppressed_duplicates += 1
                else:
                    overall_status = "pending_confirmation"
                    n_req = self.config.confirm_frames
                    reasons.append(
                        f"awaiting_temporal_confirmation_{streak}_of_{n_req}"
                    )
            else:
                self._violation_streaks[tid] = 0
                if helmet_status == "compliant" and vest_status == "compliant":
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
                    violation_streak=self._violation_streaks.get(tid, 0),
                )
            )

        return ComplianceFrameResult(
            frame_index=self._frame_index,
            timestamp=ts,
            tracks=eligible_tracks,
            associations=associations,
            unmatched_ppe=unmatched_ppe,
            events=emitted_events,
            suppressed_duplicates=suppressed_duplicates,
            invalid_detections_filtered=invalid_count,
        )
