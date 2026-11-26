"""Stream-level compliance and tracking analytics aggregator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from src.tracking.tracker import Track

if TYPE_CHECKING:
    from src.compliance.alerts import ComplianceEvent
    from src.compliance.rules import DetectionInput, PersonPPEAssociation


CANONICAL_CLASS_ORDER: tuple[str, ...] = (
    "person",
    "helmet",
    "vest",
    "gloves",
    "no_helmet",
)

COMPLIANCE_STATE_KEYS: tuple[str, ...] = (
    "compliant",
    "potential_violation",
    "pending_confirmation",
    "unknown",
    "uncertain",
    "insufficient_visibility",
)


@dataclass(frozen=True)
class AlertTimelinePoint:
    """Chronological record of an emitted alert for trend analytics."""

    timestamp: float
    frame_index: int
    event_id: str
    track_id: int
    violation_category: str
    confidence: float
    event_status: str


@dataclass(frozen=True)
class StreamAnalyticsSnapshot:
    """Operational telemetry snapshot for one logical monitoring stream."""

    frames_processed: int
    workers_observed: int
    active_workers: int
    lost_workers: int
    ppe_detections: dict[str, int]
    compliance_states: dict[str, int]
    potential_violations: int
    alerts_emitted: int
    suppressed_duplicates: int
    violation_categories: dict[str, int]
    alerts_over_time: list[AlertTimelinePoint]


@dataclass
class StreamAnalyticsTracker:
    """Accumulates per-stream worker, PPE, compliance, and alert trend metrics."""

    max_timeline_points: int = 500
    _frames_processed: int = field(default=0, init=False)
    _observed_track_ids: set[int] = field(default_factory=set, init=False)
    _active_workers: int = field(default=0, init=False)
    _lost_workers: int = field(default=0, init=False)
    _ppe_detections: dict[str, int] = field(default_factory=dict, init=False)
    _compliance_states: dict[str, int] = field(default_factory=dict, init=False)
    _potential_violations: int = field(default=0, init=False)
    _alerts_emitted: int = field(default=0, init=False)
    _suppressed_duplicates: int = field(default=0, init=False)
    _violation_categories: dict[str, int] = field(default_factory=dict, init=False)
    _alerts_over_time: list[AlertTimelinePoint] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        self._init_counters()

    def _init_counters(self) -> None:
        self._ppe_detections = {cls_name: 0 for cls_name in CANONICAL_CLASS_ORDER}
        self._compliance_states = {state: 0 for state in COMPLIANCE_STATE_KEYS}

    def reset(self) -> None:
        """Reset all cumulative stream metrics."""
        self._frames_processed = 0
        self._observed_track_ids.clear()
        self._active_workers = 0
        self._lost_workers = 0
        self._potential_violations = 0
        self._alerts_emitted = 0
        self._suppressed_duplicates = 0
        self._violation_categories.clear()
        self._alerts_over_time.clear()
        self._init_counters()

    def record_frame(
        self,
        valid_detections: list[DetectionInput],
        active_tracks: list[Track],
        lost_tracks: list[Track],
        associations: list[PersonPPEAssociation],
        emitted_events: list[ComplianceEvent],
        suppressed_duplicates: int,
    ) -> None:
        """Update counters from a completed pipeline step."""
        self._frames_processed += 1
        self._active_workers = len(active_tracks)
        self._lost_workers = len(lost_tracks)

        for trk in active_tracks:
            self._observed_track_ids.add(trk.track_id)

        for det in valid_detections:
            if det.class_name in self._ppe_detections:
                self._ppe_detections[det.class_name] += 1

        for assoc in associations:
            state = assoc.overall_status
            self._compliance_states[state] = self._compliance_states.get(state, 0) + 1
            if state == "potential_violation":
                self._potential_violations += 1

        self._suppressed_duplicates += suppressed_duplicates
        for evt in emitted_events:
            self._alerts_emitted += 1
            cat = evt.violation_category
            self._violation_categories[cat] = self._violation_categories.get(cat, 0) + 1
            self._alerts_over_time.append(
                AlertTimelinePoint(
                    timestamp=evt.timestamp,
                    frame_index=evt.frame_index,
                    event_id=evt.event_id,
                    track_id=evt.track_id,
                    violation_category=cat,
                    confidence=evt.confidence,
                    event_status=evt.event_status,
                )
            )
            if len(self._alerts_over_time) > self.max_timeline_points:
                self._alerts_over_time = self._alerts_over_time[
                    -self.max_timeline_points :
                ]

    def sync_alert_status(self, event_id: str, new_status: str) -> None:
        """Reflect operator review status updates in timeline records."""
        for idx, pt in enumerate(self._alerts_over_time):
            if pt.event_id == event_id:
                self._alerts_over_time[idx] = replace(pt, event_status=new_status)

    def snapshot(self) -> StreamAnalyticsSnapshot:
        """Return an immutable snapshot of current stream analytics."""
        return StreamAnalyticsSnapshot(
            frames_processed=self._frames_processed,
            workers_observed=len(self._observed_track_ids),
            active_workers=self._active_workers,
            lost_workers=self._lost_workers,
            ppe_detections=dict(self._ppe_detections),
            compliance_states=dict(self._compliance_states),
            potential_violations=self._potential_violations,
            alerts_emitted=self._alerts_emitted,
            suppressed_duplicates=self._suppressed_duplicates,
            violation_categories=dict(self._violation_categories),
            alerts_over_time=list(self._alerts_over_time),
        )
