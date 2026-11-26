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
