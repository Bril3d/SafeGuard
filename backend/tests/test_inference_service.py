"""Unit and integration tests for backend InferenceService and ONNX pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from backend.app.inference import DetectionResult, ONNXDetector
from backend.app.services.inference_service import (
    ImageValidationError,
    InferenceService,
    ModelUnavailableError,
    annotate_image_bytes,
    validate_and_decode_image,
)


def _encode_png_bytes(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(64, 128, 192))
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_validate_and_decode_image_accepts_valid_png_and_rejects_bad_inputs() -> None:
    raw = _encode_png_bytes(320, 240)
    pil_img = validate_and_decode_image(
        payload=raw,
        content_type="image/png",
        max_upload_bytes=1024 * 1024,
        max_image_dimension=2048,
    )
    assert pil_img.size == (320, 240)
    assert pil_img.mode == "RGB"

    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_decode_image(b"", "image/png", 1024 * 1024, 2048)

    with pytest.raises(ImageValidationError, match="Unsupported media type"):
        validate_and_decode_image(raw, "text/plain", 1024 * 1024, 2048)

    with pytest.raises(ImageValidationError, match="size exceeds"):
        validate_and_decode_image(raw, "image/png", 10, 2048)

    with pytest.raises(ImageValidationError, match="Invalid or corrupted"):
        validate_and_decode_image(b"not-an-image", "image/jpeg", 1024 * 1024, 2048)

    with pytest.raises(ImageValidationError, match="exceed limit"):
        validate_and_decode_image(raw, "image/png", 1024 * 1024, 200)


def test_annotate_image_bytes_renders_valid_jpeg_and_png() -> None:
    img = Image.new("RGB", (640, 480), color=(30, 40, 50))
    dets = [
        DetectionResult(
            class_id=1,
            class_name="helmet",
            confidence=0.91,
            bbox_xyxy=[100.0, 50.0, 180.0, 130.0],
        ),
        DetectionResult(
            class_id=4,
            class_name="no_helmet",
            confidence=0.77,
            bbox_xyxy=[300.0, 80.0, 360.0, 150.0],
        ),
    ]
    jpg_bytes = annotate_image_bytes(img, dets, output_format="jpeg")
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (480, 640, 3)

    png_bytes = annotate_image_bytes(img, dets, output_format="png")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_inference_service_raises_when_model_unloaded() -> None:
    svc = InferenceService(
        detector=None,
        model_version="unloaded",
        max_upload_bytes=1024 * 1024,
        max_image_dimension=2048,
    )
    with pytest.raises(ModelUnavailableError):
        svc.predict(
            payload=_encode_png_bytes(), content_type="image/png", request_id="r1"
        )


def test_inference_service_with_real_production_onnx_when_present() -> None:
    onnx_path = Path("models/production/safeguard_yolo11s.onnx")
    sample_dir = Path("data/processed/val_ultralytics/images")
    if not onnx_path.exists() or not sample_dir.exists():
        pytest.skip("Production ONNX artifact or validation image not present.")

    sample_img = next(sample_dir.glob("*.jpg"))
    detector = ONNXDetector(onnx_path=onnx_path)
    svc = InferenceService(
        detector=detector,
        model_version=detector.model_version,
        max_upload_bytes=10 * 1024 * 1024,
        max_image_dimension=4096,
    )
    result = svc.predict(
        payload=sample_img.read_bytes(),
        content_type="image/jpeg",
        request_id="real-check",
    )
    assert result.model_version == "EXP-E-yolo11s-onnx"
    assert result.timings_ms.inference_ms > 0.0
    assert result.timings_ms.total_ms >= result.timings_ms.inference_ms
    for det in result.detections:
        assert det.class_id in {0, 1, 2, 3, 4}
        x1, y1, x2, y2 = det.bbox_xyxy
        assert 0.0 <= x1 < x2 <= result.image_width
        assert 0.0 <= y1 < y2 <= result.image_height
