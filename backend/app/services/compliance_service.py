"""Compliance pipeline orchestrator reusing singleton InferenceService."""

from __future__ import annotations

from collections import OrderedDict
import threading
import time

from backend.app.schemas import (
    AlertStatusUpdateResponse,
    AlertTimelinePointSchema,
    ComplianceAlertsResponse,
    ComplianceEventSchema,
    ComplianceMonitorResponse,
    ComplianceResetResponse,
    ComplianceSummarySchema,
    DetectionSchema,
    PersonAssociationSchema,
    StreamAnalyticsResponse,
    TrackSchema,
)
from backend.app.services.inference_service import InferenceService
from src.compliance.alerts import ComplianceEvent
from src.compliance.rules import ComplianceConfig, ComplianceEngine, DetectionInput
from src.tracking.tracker import ByteTrackTracker, Track, TrackerConfig


def _to_detection_schema(det: DetectionInput) -> DetectionSchema:
    return DetectionSchema(
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        bbox_xyxy=det.bbox_xyxy,
    )


def _to_track_schema(trk: Track) -> TrackSchema:
    return TrackSchema(
        track_id=trk.track_id,
        bbox_xyxy=trk.bbox_xyxy,
        confidence=trk.confidence,
        age=trk.age,
        hits=trk.hits,
        time_since_update=trk.time_since_update,
        state=trk.state,
    )


def _to_event_schema(evt: ComplianceEvent) -> ComplianceEventSchema:
    return ComplianceEventSchema(
        event_id=evt.event_id,
        event_type=evt.event_type,
        severity=evt.severity,
        track_id=evt.track_id,
        timestamp=evt.timestamp,
        frame_index=evt.frame_index,
        confirmed_frames=evt.confirmed_frames,
        confidence=evt.confidence,
        bbox_xyxy=evt.bbox_xyxy,
        requires_human_review=evt.requires_human_review,
        message=evt.message,
        violation_category=evt.violation_category,
        event_status=evt.event_status,
        evidence_bbox_xyxy=evt.evidence_bbox_xyxy,
        evidence_frame_b64=evt.evidence_frame_b64,
    )


class CompliancePipelineService:
    """Connects ONNX InferenceService -> ByteTrackTracker -> ComplianceEngine."""

    def __init__(
        self,
        inference_service: InferenceService,
        compliance_config: ComplianceConfig | None = None,
        tracker_config: TrackerConfig | None = None,
        max_streams: int = 32,
    ) -> None:
        self._inference_service = inference_service
        self._compliance_config = compliance_config or ComplianceConfig()
        self._tracker_config = tracker_config or TrackerConfig()
        self._max_streams = max(1, max_streams)
        self._lock = threading.Lock()
        self._streams: OrderedDict[str, tuple[ComplianceEngine, threading.Lock]] = (
            OrderedDict()
        )

    def _get_engine(self, stream_id: str) -> tuple[ComplianceEngine, threading.Lock]:
        with self._lock:
            if stream_id in self._streams:
                self._streams.move_to_end(stream_id)
                return self._streams[stream_id]
            engine = ComplianceEngine(
                config=self._compliance_config,
                tracker=ByteTrackTracker(config=self._tracker_config),
            )
            stream_lock = threading.Lock()
            self._streams[stream_id] = (engine, stream_lock)
            while len(self._streams) > self._max_streams:
                self._streams.popitem(last=False)
            return engine, stream_lock

    def reset_stream(self, stream_id: str) -> ComplianceResetResponse:
        """Reset tracking, temporal compliance history, and analytics for a stream."""
        with self._lock:
            entry = self._streams.get(stream_id)
        if entry is not None:
            engine, stream_lock = entry
            with stream_lock:
                engine.reset()
        return ComplianceResetResponse(
            stream_id=stream_id,
            reset=True,
            message=f"Compliance state reset for stream '{stream_id}'.",
        )

    def get_alerts(
        self,
        stream_id: str = "default",
        status: str | None = None,
        limit: int = 100,
    ) -> ComplianceAlertsResponse:
        """Return retained potential compliance events for a logical stream."""
        engine, stream_lock = self._get_engine(stream_id)
        with stream_lock:
            events = engine.get_alerts(status=status, limit=limit)
            total_retained = len(engine.event_history)
        return ComplianceAlertsResponse(
            stream_id=stream_id,
            total_retained=total_retained,
            events=[_to_event_schema(evt) for evt in events],
        )

    def update_alert_status(
        self,
        stream_id: str,
        event_id: str,
        new_status: str,
    ) -> AlertStatusUpdateResponse | None:
        """Update human review status on a retained stream alert."""
        engine, stream_lock = self._get_engine(stream_id)
        with stream_lock:
            updated = engine.update_alert_status(event_id, new_status)
        if updated is None:
            return None
        return AlertStatusUpdateResponse(
            stream_id=stream_id,
            event=_to_event_schema(updated),
        )

    def get_analytics(self, stream_id: str = "default") -> StreamAnalyticsResponse:
        """Return cumulative tracking, PPE, and alert trend analytics."""
        engine, stream_lock = self._get_engine(stream_id)
        with stream_lock:
            snap = engine.get_analytics()
        return StreamAnalyticsResponse(
            stream_id=stream_id,
            frames_processed=snap.frames_processed,
            workers_observed=snap.workers_observed,
            active_workers=snap.active_workers,
            lost_workers=snap.lost_workers,
            ppe_detections=snap.ppe_detections,
            compliance_states=snap.compliance_states,
            potential_violations=snap.potential_violations,
            alerts_emitted=snap.alerts_emitted,
            suppressed_duplicates=snap.suppressed_duplicates,
            violation_categories=snap.violation_categories,
            alerts_over_time=[
                AlertTimelinePointSchema(
                    timestamp=pt.timestamp,
                    frame_index=pt.frame_index,
                    event_id=pt.event_id,
                    track_id=pt.track_id,
                    violation_category=pt.violation_category,
                    confidence=pt.confidence,
                    event_status=pt.event_status,
                )
                for pt in snap.alerts_over_time
            ],
        )

    def process_frame(
        self,
        payload: bytes,
        content_type: str | None,
        request_id: str,
        stream_id: str = "default",
        frame_id: str | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        timestamp: float | None = None,
    ) -> ComplianceMonitorResponse:
        """Execute Raw -> Tracking -> Compliance -> Alerts + Analytics."""
        pred, frame_bgr = self._inference_service.predict_with_frame(
            payload=payload,
            content_type=content_type,
            request_id=request_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        t_comp0 = time.perf_counter()
        det_inputs = [
            DetectionInput(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox_xyxy=d.bbox_xyxy,
            )
            for d in pred.detections
        ]

        engine, stream_lock = self._get_engine(stream_id)
        with stream_lock:
            comp_res = engine.process_frame(
                det_inputs,
                timestamp=timestamp,
                frame_bgr=frame_bgr,
                frame_size=(pred.image_width, pred.image_height),
            )
        comp_ms = round((time.perf_counter() - t_comp0) * 1000.0, 3)

        compliant_cnt = sum(
            1 for a in comp_res.associations if a.overall_status == "compliant"
        )
        pot_viol_cnt = sum(
            1
            for a in comp_res.associations
            if a.overall_status == "potential_violation"
        )
        pending_cnt = sum(
            1
            for a in comp_res.associations
            if a.overall_status == "pending_confirmation"
        )
        unknown_cnt = sum(
            1 for a in comp_res.associations if a.overall_status == "unknown"
        )
        uncertain_cnt = sum(
            1 for a in comp_res.associations if a.overall_status == "uncertain"
        )
        insuff_cnt = sum(
            1
            for a in comp_res.associations
            if a.overall_status == "insufficient_visibility"
        )

        return ComplianceMonitorResponse(
            request_id=request_id,
            stream_id=stream_id,
            frame_id=frame_id,
            model_version=pred.model_version,
            image_width=pred.image_width,
            image_height=pred.image_height,
            frame_index=comp_res.frame_index,
            timestamp=comp_res.timestamp,
            detections=pred.detections,
            tracks=[_to_track_schema(t) for t in comp_res.tracks],
            lost_tracks=[_to_track_schema(t) for t in comp_res.lost_tracks],
            associations=[
                PersonAssociationSchema(
                    track_id=a.track_id,
                    person_confidence=a.person_confidence,
                    bbox_xyxy=a.bbox_xyxy,
                    helmet_boxes=[_to_detection_schema(b) for b in a.helmet_boxes],
                    no_helmet_boxes=[
                        _to_detection_schema(b) for b in a.no_helmet_boxes
                    ],
                    vest_boxes=[_to_detection_schema(b) for b in a.vest_boxes],
                    gloves_boxes=[_to_detection_schema(b) for b in a.gloves_boxes],
                    helmet_status=a.helmet_status,
                    vest_status=a.vest_status,
                    overall_status=a.overall_status,
                    uncertainty_reasons=a.uncertainty_reasons,
                    violation_streak=a.violation_streak,
                    visibility_status=a.visibility_status,
                )
                for a in comp_res.associations
            ],
            unmatched_ppe=[_to_detection_schema(b) for b in comp_res.unmatched_ppe],
            events=[_to_event_schema(e) for e in comp_res.events],
            summary=ComplianceSummarySchema(
                tracked_persons=len(comp_res.associations),
                compliant_count=compliant_cnt,
                potential_violation_count=pot_viol_cnt,
                pending_confirmation_count=pending_cnt,
                unknown_count=unknown_cnt,
                emitted_event_count=len(comp_res.events),
                suppressed_duplicates=comp_res.suppressed_duplicates,
                uncertain_count=uncertain_cnt,
                insufficient_visibility_count=insuff_cnt,
                lost_tracks_count=len(comp_res.lost_tracks),
            ),
            timings_ms=pred.timings_ms,
            compliance_ms=comp_ms,
        )
