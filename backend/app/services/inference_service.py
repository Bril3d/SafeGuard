"""Image validation, ONNX inference orchestration, and OpenCV rendering service."""

from __future__ import annotations

import io
import time
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.inference import DetectionResult, InferenceOutput, ONNXDetector
from backend.app.monitoring import OperationalMonitor
from backend.app.schemas import (
    ClassesResponse,
    ClassItemSchema,
    DetectionSchema,
    ModelMetadataResponse,
    PredictResponse,
    TensorSpecSchema,
    TimingBreakdownSchema,
)

ALLOWED_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }
)

# OpenCV BGR palette aligned with canonical SafeGuard classes.
CLASS_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (235, 99, 37),  # person: blue
    1: (94, 197, 34),  # helmet: green
    2: (182, 168, 20),  # vest: teal
    3: (11, 158, 245),  # gloves: amber
    4: (68, 68, 239),  # no_helmet: red
}


class ImageValidationError(Exception):
    """Raised when an uploaded image fails MIME, size, or decode validation."""


class ModelUnavailableError(Exception):
    """Raised when ONNX inference session is uninitialized or unavailable."""


def validate_and_decode_image(
    payload: bytes,
    content_type: str | None,
    max_upload_bytes: int,
    max_image_dimension: int,
) -> Image.Image:
    """Validate uploaded bytes and decode into an oriented RGB PIL image."""
    if not payload:
        raise ImageValidationError("Uploaded image file is empty.")

    if len(payload) > max_upload_bytes:
        raise ImageValidationError(
            f"Uploaded image size exceeds maximum limit of {max_upload_bytes} bytes."
        )

    normalized_mime = (content_type or "").split(";")[0].strip().lower()
    if normalized_mime and normalized_mime not in (
        ALLOWED_IMAGE_MIME_TYPES | {"application/octet-stream"}
    ):
        raise ImageValidationError(
            f"Unsupported media type '{normalized_mime}'. "
            "Allowed types: JPEG, PNG, WEBP, BMP."
        )

    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(payload)) as img:
            oriented = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Invalid or corrupted image file.") from exc

    width, height = oriented.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("Decoded image has invalid zero dimensions.")
    if width > max_image_dimension or height > max_image_dimension:
        raise ImageValidationError(
            f"Decoded image dimensions {width}x{height} exceed limit "
            f"of {max_image_dimension}px."
        )
    return oriented


def annotate_image_bytes(
    image: Image.Image,
    detections: list[DetectionResult],
    output_format: Literal["jpeg", "png"] = "jpeg",
) -> bytes:
    """Render bounding boxes, class labels, and confidences using OpenCV."""
    rgb = np.asarray(image, dtype=np.uint8)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_h, img_w = canvas.shape[:2]
    thickness = max(2, int(round(min(img_h, img_w) / 320.0)))
    font_scale = max(0.45, min(img_h, img_w) / 960.0)

    for det in detections:
        color = CLASS_COLORS_BGR.get(det.class_id, (200, 200, 200))
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox_xyxy)
        x1 = max(0, min(img_w - 1, x1))
        y1 = max(0, min(img_h - 1, y1))
        x2 = max(0, min(img_w - 1, x2))
        y2 = max(0, min(img_h - 1, y2))

        cv2.rectangle(
            canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA
        )

        label = f"{det.class_name} {det.confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        pill_top = max(0, y1 - text_h - baseline - 6)
        pill_bottom = max(text_h + baseline + 6, y1)
        cv2.rectangle(
            canvas,
            (x1, pill_top),
            (min(img_w - 1, x1 + text_w + 8), pill_bottom),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (x1 + 4, max(text_h + 2, pill_bottom - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness - 1),
            lineType=cv2.LINE_AA,
        )

    ext = ".png" if output_format == "png" else ".jpg"
    encode_params = (
        [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if output_format == "png"
        else [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    ok, buffer = cv2.imencode(ext, canvas, encode_params)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode annotated {output_format} image.")
    return buffer.tobytes()


class InferenceService:
    """Thread-safe inference facade wrapping a singleton ONNXDetector."""

    def __init__(
        self,
        detector: ONNXDetector | None,
        model_version: str,
        max_upload_bytes: int,
        max_image_dimension: int,
        monitor: OperationalMonitor | None = None,
    ) -> None:
        self._detector = detector
        self.model_version = model_version
        self.max_upload_bytes = max_upload_bytes
        self.max_image_dimension = max_image_dimension
        self.monitor = monitor

    def _require_detector(self) -> ONNXDetector:
        if self._detector is None:
            raise ModelUnavailableError(
                "Inference model is not initialized or unavailable."
            )
        return self._detector

    def get_metadata(self) -> ModelMetadataResponse:
        """Return metadata for the currently loaded ONNX runtime artifact."""
        det = self._require_detector()
        meta = det.metadata
        inputs = [
            TensorSpecSchema(
                name=inp.name,
                shape=list(inp.shape),
                type=str(inp.type),
            )
            for inp in det.session.get_inputs()
        ]
        outputs = [
            TensorSpecSchema(
                name=out.name,
                shape=list(out.shape),
                type=str(out.type),
            )
            for out in det.session.get_outputs()
        ]
        return ModelMetadataResponse(
            model_name=det.onnx_path.name,
            model_version=self.model_version,
            architecture=str(meta.get("architecture", "YOLO11s")),
            opset=int(meta["opset"]) if meta.get("opset") is not None else None,
            imgsz=det.imgsz,
            default_conf=det.default_conf,
            default_iou=det.default_iou,
            onnx_sha256=meta.get("onnx_sha256"),
            mlflow_run_id=meta.get("mlflow_run_id"),
            artifact_size_mb=meta.get("onnx_size_mb"),
            runtime_provider=det.active_provider,
            class_names=dict(sorted(det.class_names.items())),
            inputs=inputs,
            outputs=outputs,
        )

    def get_classes(self) -> ClassesResponse:
        """Return ordered canonical classes supported by the loaded model."""
        det = self._require_detector()
        items = [
            ClassItemSchema(id=cid, name=cname)
            for cid, cname in sorted(det.class_names.items())
        ]
        return ClassesResponse(model_version=self.model_version, classes=items)

    def _run_validated_inference(
        self,
        payload: bytes,
        content_type: str | None,
        conf_threshold: float | None,
        iou_threshold: float | None,
    ) -> tuple[Image.Image, InferenceOutput, float, float, float]:
        det = self._require_detector()
        t_decode0 = time.perf_counter()
        image = validate_and_decode_image(
            payload=payload,
            content_type=content_type,
            max_upload_bytes=self.max_upload_bytes,
            max_image_dimension=self.max_image_dimension,
        )
        decode_ms = (time.perf_counter() - t_decode0) * 1000.0
        applied_conf = det.default_conf if conf_threshold is None else conf_threshold
        applied_iou = det.default_iou if iou_threshold is None else iou_threshold
        out = det.predict_image(
            image=image,
            conf_threshold=applied_conf,
            iou_threshold=applied_iou,
        )
        return image, out, decode_ms, applied_conf, applied_iou

    def _build_predict_response(
        self,
        out: InferenceOutput,
        decode_ms: float,
        applied_conf: float,
        applied_iou: float,
        request_id: str,
    ) -> PredictResponse:
        total_ms = decode_ms + out.timings_ms["end_to_end_ms"]
        return PredictResponse(
            request_id=request_id,
            model_version=self.model_version,
            image_width=out.image_width,
            image_height=out.image_height,
            conf_threshold=applied_conf,
            iou_threshold=applied_iou,
            detection_count=len(out.detections),
            detections=[
                DetectionSchema(
                    class_id=d.class_id,
                    class_name=d.class_name,
                    confidence=d.confidence,
                    bbox_xyxy=d.bbox_xyxy,
                )
                for d in out.detections
            ],
            timings_ms=TimingBreakdownSchema(
                decode_ms=round(decode_ms, 3),
                preprocess_ms=out.timings_ms["preprocess_ms"],
                inference_ms=out.timings_ms["inference_ms"],
                postprocess_ms=out.timings_ms["postprocess_ms"],
                total_ms=round(total_ms, 3),
            ),
        )

    def predict(
        self,
        payload: bytes,
        content_type: str | None,
        request_id: str,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        route_label: str = "/predict",
    ) -> PredictResponse:
        """Validate image bytes and return structured detections and timings."""
        _, out, decode_ms, applied_conf, applied_iou = self._run_validated_inference(
            payload=payload,
            content_type=content_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        if self.monitor is not None:
            self.monitor.record_prediction(
                route=route_label,
                status_code=200,
                model_version=self.model_version,
                inference_ms=out.timings_ms["inference_ms"],
                image_width=out.image_width,
                image_height=out.image_height,
                detections=out.detections,
                request_id=request_id,
            )
        return self._build_predict_response(
            out, decode_ms, applied_conf, applied_iou, request_id
        )

    def predict_with_frame(
        self,
        payload: bytes,
        content_type: str | None,
        request_id: str,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        route_label: str = "/predict/compliance",
    ) -> tuple[PredictResponse, np.ndarray]:
        """Run inference and also return decoded BGR frame for evidence snapshots."""
        image, out, decode_ms, applied_conf, applied_iou = (
            self._run_validated_inference(
                payload=payload,
                content_type=content_type,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        )
        if self.monitor is not None:
            self.monitor.record_prediction(
                route=route_label,
                status_code=200,
                model_version=self.model_version,
                inference_ms=out.timings_ms["inference_ms"],
                image_width=out.image_width,
                image_height=out.image_height,
                detections=out.detections,
                request_id=request_id,
            )
        rgb = np.asarray(image, dtype=np.uint8)
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        resp = self._build_predict_response(
            out, decode_ms, applied_conf, applied_iou, request_id
        )
        return resp, frame_bgr

    def predict_and_visualize(
        self,
        payload: bytes,
        content_type: str | None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        output_format: Literal["jpeg", "png"] = "jpeg",
        request_id: str = "unknown",
    ) -> tuple[bytes, int, float, float]:
        """Run inference and return encoded annotated image bytes + headers metadata."""
        image, out, decode_ms, _, _ = self._run_validated_inference(
            payload=payload,
            content_type=content_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        if self.monitor is not None:
            self.monitor.record_prediction(
                route="/predict/visualize",
                status_code=200,
                model_version=self.model_version,
                inference_ms=out.timings_ms["inference_ms"],
                image_width=out.image_width,
                image_height=out.image_height,
                detections=out.detections,
                request_id=request_id,
            )
        encoded = annotate_image_bytes(
            image=image,
            detections=out.detections,
            output_format=output_format,
        )
        total_ms = round(decode_ms + out.timings_ms["end_to_end_ms"], 3)
        return (
            encoded,
            len(out.detections),
            out.timings_ms["inference_ms"],
            total_ms,
        )
