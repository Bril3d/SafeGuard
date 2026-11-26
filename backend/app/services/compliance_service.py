"""Compliance pipeline orchestrator reusing singleton InferenceService."""

from __future__ import annotations

from collections import OrderedDict
import threading
import time

from backend.app.schemas import (
    ComplianceEventSchema,
    ComplianceMonitorResponse,
    ComplianceResetResponse,
    ComplianceSummarySchema,
    DetectionSchema,
    PersonAssociationSchema,
    TrackSchema,
)
from backend.app.services.inference_service import InferenceService
from src.compliance.rules import ComplianceConfig, ComplianceEngine, DetectionInput
from src.tracking.tracker import ByteTrackTracker, TrackerConfig


def _to_detection_schema(det: DetectionInput) -> DetectionSchema:
    return DetectionSchema(
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        bbox_xyxy=det.bbox_xyxy,
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
        """Reset tracking and temporal compliance history for a stream."""
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
        """Execute Detection -> Tracking -> Association -> Compliance Rules."""
        pred = self._inference_service.predict(
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
            comp_res = engine.process_frame(det_inputs, timestamp=timestamp)
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
            tracks=[
                TrackSchema(
                    track_id=t.track_id,
                    bbox_xyxy=t.bbox_xyxy,
                    confidence=t.confidence,
                    age=t.age,
                    hits=t.hits,
                    time_since_update=t.time_since_update,
                    state=t.state,
                )
                for t in comp_res.tracks
            ],
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
                )
                for a in comp_res.associations
            ],
            unmatched_ppe=[_to_detection_schema(b) for b in comp_res.unmatched_ppe],
            events=[
                ComplianceEventSchema(
                    event_id=e.event_id,
                    event_type=e.event_type,
                    severity=e.severity,
                    track_id=e.track_id,
                    timestamp=e.timestamp,
                    frame_index=e.frame_index,
                    confirmed_frames=e.confirmed_frames,
                    confidence=e.confidence,
                    bbox_xyxy=e.bbox_xyxy,
                    requires_human_review=e.requires_human_review,
                    message=e.message,
                )
                for e in comp_res.events
            ],
            summary=ComplianceSummarySchema(
                tracked_persons=len(comp_res.associations),
                compliant_count=compliant_cnt,
                potential_violation_count=pot_viol_cnt,
                pending_confirmation_count=pending_cnt,
                unknown_count=unknown_cnt,
                emitted_event_count=len(comp_res.events),
                suppressed_duplicates=comp_res.suppressed_duplicates,
            ),
            timings_ms=pred.timings_ms,
            compliance_ms=comp_ms,
        )
