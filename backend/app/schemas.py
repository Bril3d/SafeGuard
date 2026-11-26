"""Strict Pydantic v2 schemas for the SafeGuard FastAPI inference service."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Base schema rejecting unexpected extra fields."""

    model_config = ConfigDict(extra="forbid")


class ErrorDetail(StrictSchema):
    """Machine-readable error detail payload."""

    code: str = Field(description="Semantic error code identifier.")
    message: str = Field(description="Human-readable error description.")
    request_id: str = Field(description="Request correlation identifier.")
    details: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional structured field-level validation details.",
    )


class ErrorResponse(StrictSchema):
    """Uniform structured error envelope for 4xx/5xx responses."""

    error: ErrorDetail = Field(description="Structured error payload.")


class HealthResponse(StrictSchema):
    """Response payload for GET /health."""

    status: str = Field(description="Service liveness status.")
    service: str = Field(description="Configured service name.")
    environment: str = Field(description="Active runtime environment.")
    model_loaded: bool = Field(
        description="Whether an ONNX inference session is initialized."
    )
    model_version: str = Field(
        description="Loaded model version or 'unloaded' if unavailable."
    )
    gpu_available: bool = Field(
        description="Whether a GPU execution provider is active."
    )
    runtime_provider: str = Field(
        description="Active ONNX Runtime provider, 'uninitialized', or 'error'."
    )


class TensorSpecSchema(StrictSchema):
    """ONNX input/output tensor metadata description."""

    name: str = Field(description="Tensor binding name.")
    shape: list[int | str] = Field(description="Expected tensor shape.")
    type: str = Field(description="ONNX element data type.")


class ClassItemSchema(StrictSchema):
    """Canonical detection class ID and name."""

    id: int = Field(ge=0, description="Canonical class integer ID.")
    name: str = Field(description="Canonical class label.")


class ClassesResponse(StrictSchema):
    """Response payload for GET /classes."""

    model_version: str = Field(description="Active ONNX model version.")
    classes: list[ClassItemSchema] = Field(
        description="Ordered canonical class definitions."
    )


class ModelMetadataResponse(StrictSchema):
    """Response payload for GET /model/metadata."""

    model_name: str = Field(description="Deployed artifact name.")
    model_version: str = Field(description="Model version identifier.")
    architecture: str = Field(description="Backbone/detector architecture.")
    opset: int | None = Field(default=None, description="ONNX opset version.")
    imgsz: int = Field(gt=0, description="Configured square inference resolution.")
    default_conf: float = Field(
        ge=0.0, le=1.0, description="Default confidence cutoff."
    )
    default_iou: float = Field(ge=0.0, le=1.0, description="Default NMS IoU threshold.")
    onnx_sha256: str | None = Field(
        default=None, description="SHA-256 digest of the loaded ONNX file."
    )
    mlflow_run_id: str | None = Field(
        default=None, description="Originating MLflow training/optimization run ID."
    )
    artifact_size_mb: float | None = Field(
        default=None, description="ONNX file size in megabytes."
    )
    runtime_provider: str = Field(description="Active ONNX Runtime execution provider.")
    class_names: dict[int, str] = Field(description="Canonical ID-to-name mapping.")
    inputs: list[TensorSpecSchema] = Field(description="Model input tensor specs.")
    outputs: list[TensorSpecSchema] = Field(description="Model output tensor specs.")


class DetectionSchema(StrictSchema):
    """Single object detection in original pixel xyxy coordinates."""

    class_id: int = Field(ge=0, description="Canonical class ID.")
    class_name: str = Field(description="Canonical class label.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence score.")
    bbox_xyxy: list[float] = Field(
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in original image pixels.",
    )


class TimingBreakdownSchema(StrictSchema):
    """Server-side timing measurements in milliseconds."""

    decode_ms: float = Field(ge=0.0, description="Image validation and decoding time.")
    preprocess_ms: float = Field(ge=0.0, description="Letterbox and tensor build time.")
    inference_ms: float = Field(ge=0.0, description="ONNX Runtime forward-pass time.")
    postprocess_ms: float = Field(ge=0.0, description="NMS and box unscaling time.")
    total_ms: float = Field(ge=0.0, description="Total server processing time.")


class PredictResponse(StrictSchema):
    """Response payload for POST /predict."""

    request_id: str = Field(description="Request correlation identifier.")
    model_version: str = Field(description="Model version used for inference.")
    image_width: int = Field(gt=0, description="Decoded image width in pixels.")
    image_height: int = Field(gt=0, description="Decoded image height in pixels.")
    conf_threshold: float = Field(
        ge=0.0, le=1.0, description="Applied confidence threshold."
    )
    iou_threshold: float = Field(
        ge=0.0, le=1.0, description="Applied NMS IoU threshold."
    )
    detection_count: int = Field(ge=0, description="Number of returned detections.")
    detections: list[DetectionSchema] = Field(description="List of detected objects.")
    timings_ms: TimingBreakdownSchema = Field(
        description="Server-side stage latency breakdown."
    )


class FramePredictResponse(PredictResponse):
    """Response payload for POST /predict/frame."""

    frame_id: str | None = Field(
        default=None, description="Optional client-supplied frame identifier."
    )


class TrackSchema(StrictSchema):
    """Anonymous tracked person bounding box and lifecycle counters."""

    track_id: int = Field(ge=1, description="Temporary anonymous frame-to-frame ID.")
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    confidence: float = Field(ge=0.0, le=1.0)
    age: int = Field(ge=1)
    hits: int = Field(ge=1)
    time_since_update: int = Field(ge=0)
    state: str = Field(description="Track status ('active' or 'lost').")


class PersonAssociationSchema(StrictSchema):
    """Person-to-PPE spatial association and safety status."""

    track_id: int = Field(ge=1)
    person_confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    helmet_boxes: list[DetectionSchema]
    no_helmet_boxes: list[DetectionSchema]
    vest_boxes: list[DetectionSchema]
    gloves_boxes: list[DetectionSchema]
    helmet_status: str
    vest_status: str
    overall_status: str
    uncertainty_reasons: list[str]
    violation_streak: int = Field(ge=0)
    visibility_status: str = Field(default="sufficient")


class ComplianceEventSchema(StrictSchema):
    """Potential compliance event requiring human verification."""

    event_id: str
    event_type: str
    severity: str
    track_id: int = Field(ge=1)
    timestamp: float
    frame_index: int = Field(ge=1)
    confirmed_frames: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)
    requires_human_review: bool
    message: str
    violation_category: str = Field(default="no_helmet")
    event_status: str = Field(default="pending_review")
    evidence_bbox_xyxy: list[float] | None = None
    evidence_frame_b64: str | None = None


class ComplianceSummarySchema(StrictSchema):
    """Aggregate compliance counts for the processed frame."""

    tracked_persons: int = Field(ge=0)
    compliant_count: int = Field(ge=0)
    potential_violation_count: int = Field(ge=0)
    pending_confirmation_count: int = Field(ge=0)
    unknown_count: int = Field(ge=0)
    emitted_event_count: int = Field(ge=0)
    suppressed_duplicates: int = Field(ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    insufficient_visibility_count: int = Field(default=0, ge=0)
    lost_tracks_count: int = Field(default=0, ge=0)


class ComplianceMonitorResponse(StrictSchema):
    """Full pipeline response for POST /predict/compliance."""

    request_id: str
    stream_id: str
    frame_id: str | None = None
    model_version: str
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    frame_index: int = Field(ge=1)
    timestamp: float
    detections: list[DetectionSchema]
    tracks: list[TrackSchema]
    lost_tracks: list[TrackSchema] = Field(default_factory=list)
    associations: list[PersonAssociationSchema]
    unmatched_ppe: list[DetectionSchema]
    events: list[ComplianceEventSchema]
    summary: ComplianceSummarySchema
    timings_ms: TimingBreakdownSchema
    compliance_ms: float = Field(ge=0.0)


class ComplianceResetResponse(StrictSchema):
    """Response payload for POST /compliance/reset."""

    stream_id: str
    reset: bool
    message: str


class AlertTimelinePointSchema(StrictSchema):
    """Chronological point representing a generated alert event."""

    timestamp: float
    frame_index: int = Field(ge=1)
    event_id: str
    track_id: int = Field(ge=1)
    violation_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    event_status: str


class StreamAnalyticsResponse(StrictSchema):
    """Cumulative analytics snapshot for a monitoring stream."""

    stream_id: str
    frames_processed: int = Field(ge=0)
    workers_observed: int = Field(ge=0)
    active_workers: int = Field(ge=0)
    lost_workers: int = Field(ge=0)
    ppe_detections: dict[str, int]
    compliance_states: dict[str, int]
    potential_violations: int = Field(ge=0)
    alerts_emitted: int = Field(ge=0)
    suppressed_duplicates: int = Field(ge=0)
    violation_categories: dict[str, int]
    alerts_over_time: list[AlertTimelinePointSchema]


class ComplianceAlertsResponse(StrictSchema):
    """Retained alert list for a monitoring stream."""

    stream_id: str
    total_retained: int = Field(ge=0)
    events: list[ComplianceEventSchema]


class AlertStatusUpdateRequest(StrictSchema):
    """Human review / status transition payload."""

    event_status: str = Field(
        description=(
            "Updated status: pending_review, acknowledged, "
            "verified_potential_event, dismissed_false_alarm, or resolved."
        )
    )


class AlertStatusUpdateResponse(StrictSchema):
    """Updated compliance alert payload."""

    stream_id: str
    event: ComplianceEventSchema
