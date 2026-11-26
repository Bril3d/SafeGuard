"""Stage 4: Alert generation, cooldown deduplication, status lifecycle, and evidence."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.compliance.rules import PersonPPEAssociation


VALID_EVENT_STATUSES: frozenset[str] = frozenset(
    {
        "pending_review",
        "acknowledged",
        "verified_potential_event",
        "dismissed_false_alarm",
        "resolved",
    }
)


@dataclass(frozen=True)
class AlertConfig:
    """Configurable cooldown, deduplication, and annotated evidence policy."""

    cooldown_seconds: float = 10.0
    max_event_history: int = 200
    capture_evidence: bool = True
    max_evidence_width: int = 640
    jpeg_quality: int = 80


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
    violation_category: str = "no_helmet"
    event_status: str = "pending_review"
    evidence_bbox_xyxy: list[float] | None = None
    evidence_frame_b64: str | None = None


def render_evidence_frame(
    frame_bgr: np.ndarray,
    track_id: int,
    person_bbox: list[float],
    evidence_bbox: list[float] | None,
    category: str,
    confidence: float,
    timestamp: float,
    max_width: int = 640,
    jpeg_quality: int = 80,
) -> str | None:
    """Annotate an evidence snapshot for a triggered alert and encode as JPEG b64."""
    if frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return None

    img_h, img_w = frame_bgr.shape[:2]
    scale = min(1.0, float(max_width) / float(img_w)) if max_width > 0 else 1.0
    if scale < 1.0:
        canvas = cv2.resize(
            frame_bgr,
            (max(1, int(round(img_w * scale))), max(1, int(round(img_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        canvas = frame_bgr.copy()

    ch, cw = canvas.shape[:2]

    def _clip_box(box: list[float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in box)
        return (
            max(0, min(cw - 1, x1)),
            max(0, min(ch - 1, y1)),
            max(0, min(cw - 1, x2)),
            max(0, min(ch - 1, y2)),
        )

    px1, py1, px2, py2 = _clip_box(person_bbox)
    cv2.rectangle(canvas, (px1, py1), (px2, py2), (37, 99, 235), 2, cv2.LINE_AA)

    if evidence_bbox is not None:
        ex1, ey1, ex2, ey2 = _clip_box(evidence_bbox)
        cv2.rectangle(canvas, (ex1, ey1), (ex2, ey2), (38, 38, 220), 2, cv2.LINE_AA)

    banner = (
        f"POTENTIAL {category.upper()} | Track #{track_id} "
        f"| conf={confidence:.2f} | t={timestamp:.1f}s | REVIEW REQUIRED"
    )
    cv2.rectangle(canvas, (0, 0), (cw, 26), (15, 23, 42), -1)
    cv2.putText(
        canvas,
        banner,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    tag = f"#{track_id} {category}"
    cv2.putText(
        canvas,
        tag,
        (px1, max(40, py1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    ok, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(20, min(95, jpeg_quality))],
    )
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@dataclass
class AlertManager:
    """Manages alert generation, cooldown, duplicate suppression, and status."""

    config: AlertConfig = field(default_factory=AlertConfig)
    _last_alert_ts: dict[tuple[int, str], float] = field(
        default_factory=dict, init=False
    )
    _event_counter: int = field(default=0, init=False)
    _event_history: list[ComplianceEvent] = field(default_factory=list, init=False)

    def reset(self) -> None:
        """Clear cooldown timers and alert history for a stream."""
        self._last_alert_ts.clear()
        self._event_counter = 0
        self._event_history.clear()

    @property
    def event_history(self) -> list[ComplianceEvent]:
        """Bounded list of emitted compliance alert events."""
        return list(self._event_history)

    def list_events(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[ComplianceEvent]:
        """Return recent alerts optionally filtered by event_status."""
        events = list(reversed(self._event_history))
        if status is not None:
            events = [evt for evt in events if evt.event_status == status]
        return events[: max(1, limit)]

    def get_event(self, event_id: str) -> ComplianceEvent | None:
        """Fetch an alert by event_id if still retained in bounded history."""
        for evt in self._event_history:
            if evt.event_id == event_id:
                return evt
        return None

    def update_status(self, event_id: str, new_status: str) -> ComplianceEvent | None:
        """Update review/lifecycle status for an existing event."""
        if new_status not in VALID_EVENT_STATUSES:
            raise ValueError(
                f"Unsupported event_status '{new_status}'. "
                f"Expected one of {sorted(VALID_EVENT_STATUSES)}."
            )
        for idx, evt in enumerate(self._event_history):
            if evt.event_id == event_id:
                updated = replace(evt, event_status=new_status)
                self._event_history[idx] = updated
                return updated
        return None

    def process_associations(
        self,
        associations: list[PersonPPEAssociation],
        frame_index: int,
        timestamp: float,
        frame_bgr: np.ndarray | None = None,
    ) -> tuple[list[ComplianceEvent], int]:
        """Emit deduplicated alerts for confirmed potential violations."""
        emitted: list[ComplianceEvent] = []
        suppressed = 0

        for assoc in associations:
            if assoc.overall_status != "potential_violation":
                continue
            if not assoc.no_helmet_boxes:
                continue

            category = "no_helmet"
            alert_key = (assoc.track_id, category)
            last_ts = self._last_alert_ts.get(alert_key)
            if (
                last_ts is not None
                and (timestamp - last_ts) < self.config.cooldown_seconds
            ):
                suppressed += 1
                continue

            best_nh = max(assoc.no_helmet_boxes, key=lambda b: b.confidence)
            self._event_counter += 1
            self._last_alert_ts[alert_key] = timestamp

            evidence_b64: str | None = None
            if frame_bgr is not None and self.config.capture_evidence:
                evidence_b64 = render_evidence_frame(
                    frame_bgr=frame_bgr,
                    track_id=assoc.track_id,
                    person_bbox=assoc.bbox_xyxy,
                    evidence_bbox=best_nh.bbox_xyxy,
                    category=category,
                    confidence=best_nh.confidence,
                    timestamp=timestamp,
                    max_width=self.config.max_evidence_width,
                    jpeg_quality=self.config.jpeg_quality,
                )

            evt = ComplianceEvent(
                event_id=f"evt-{self._event_counter:06d}",
                event_type="potential_no_helmet",
                violation_category=category,
                severity="high",
                track_id=assoc.track_id,
                timestamp=timestamp,
                frame_index=frame_index,
                confirmed_frames=assoc.violation_streak,
                confidence=round(best_nh.confidence, 4),
                bbox_xyxy=list(assoc.bbox_xyxy),
                evidence_bbox_xyxy=list(best_nh.bbox_xyxy),
                requires_human_review=True,
                event_status="pending_review",
                evidence_frame_b64=evidence_b64,
                message=(
                    f"Potential missing helmet on track #{assoc.track_id} "
                    f"(confirmed across {assoc.violation_streak} frames; "
                    "review required)."
                ),
            )
            emitted.append(evt)
            self._event_history.append(evt)
            if len(self._event_history) > self.config.max_event_history:
                self._event_history = self._event_history[
                    -self.config.max_event_history :
                ]

        return emitted, suppressed
