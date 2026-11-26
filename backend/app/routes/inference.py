"""Inference endpoints for structured detections, visualization, and video frames."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, Query, Response, UploadFile

from backend.app.routes.dependencies import InferenceServiceDep, RequestIdDep
from backend.app.schemas import ErrorResponse, FramePredictResponse, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unsupported image."},
        422: {"model": ErrorResponse, "description": "Request validation failure."},
        503: {"model": ErrorResponse, "description": "ONNX model unavailable."},
        500: {"model": ErrorResponse, "description": "Internal inference failure."},
    },
)


@router.post(
    "/predict",
    summary="Run PPE detection on an uploaded image",
)
def predict(
    service: InferenceServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WEBP, BMP).")],
    conf: Annotated[
        float | None,
        Query(ge=0.01, le=1.0, description="Optional confidence threshold override."),
    ] = None,
    iou: Annotated[
        float | None,
        Query(ge=0.05, le=0.95, description="Optional NMS IoU threshold override."),
    ] = None,
) -> PredictResponse:
    """Validate uploaded image and return structured ONNX detections and timings."""
    payload = file.file.read()
    result = service.predict(
        payload=payload,
        content_type=file.content_type,
        request_id=request_id,
        conf_threshold=conf,
        iou_threshold=iou,
    )
    logger.info(
        "request_id=%s route=/predict size=%dx%d dets=%d inf_ms=%.2f total_ms=%.2f",
        request_id,
        result.image_width,
        result.image_height,
        result.detection_count,
        result.timings_ms.inference_ms,
        result.timings_ms.total_ms,
    )
    return result


@router.post(
    "/predict/visualize",
    summary="Return an annotated image with bounding boxes and labels",
    responses={
        200: {
            "content": {"image/jpeg": {}, "image/png": {}},
            "description": "Annotated image binary.",
        }
    },
)
def predict_visualize(
    service: InferenceServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WEBP, BMP).")],
    conf: Annotated[
        float | None,
        Query(ge=0.01, le=1.0, description="Optional confidence threshold override."),
    ] = None,
    iou: Annotated[
        float | None,
        Query(ge=0.05, le=0.95, description="Optional NMS IoU threshold override."),
    ] = None,
    output_format: Annotated[
        Literal["jpeg", "png"],
        Query(description="Encoded output image format."),
    ] = "jpeg",
) -> Response:
    """Run ONNX detection and return OpenCV-annotated image bytes."""
    payload = file.file.read()
    encoded, det_count, inf_ms, total_ms = service.predict_and_visualize(
        payload=payload,
        content_type=file.content_type,
        conf_threshold=conf,
        iou_threshold=iou,
        output_format=output_format,
    )
    media_type = "image/png" if output_format == "png" else "image/jpeg"
    logger.info(
        "request_id=%s route=/predict/visualize format=%s dets=%d "
        "inf_ms=%.2f total_ms=%.2f",
        request_id,
        output_format,
        det_count,
        inf_ms,
        total_ms,
    )
    return Response(
        content=encoded,
        media_type=media_type,
        headers={
            "X-Request-Id": request_id,
            "X-Detection-Count": str(det_count),
            "X-Inference-Ms": str(inf_ms),
            "X-Total-Ms": str(total_ms),
        },
    )


@router.post(
    "/predict/frame",
    summary="Run PPE detection on a live webcam or video frame",
)
def predict_frame(
    service: InferenceServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Captured frame image.")],
    frame_id: Annotated[
        str | None,
        Query(max_length=64, description="Optional client frame identifier."),
    ] = None,
    conf: Annotated[
        float | None,
        Query(ge=0.01, le=1.0, description="Optional confidence threshold override."),
    ] = None,
    iou: Annotated[
        float | None,
        Query(ge=0.05, le=0.95, description="Optional NMS IoU threshold override."),
    ] = None,
) -> FramePredictResponse:
    """Share the /predict ONNX service layer for low-overhead webcam frame inference."""
    payload = file.file.read()
    pred = service.predict(
        payload=payload,
        content_type=file.content_type,
        request_id=request_id,
        conf_threshold=conf,
        iou_threshold=iou,
    )
    logger.info(
        "request_id=%s route=/predict/frame frame_id=%s dets=%d "
        "inf_ms=%.2f total_ms=%.2f",
        request_id,
        frame_id,
        pred.detection_count,
        pred.timings_ms.inference_ms,
        pred.timings_ms.total_ms,
    )
    return FramePredictResponse(frame_id=frame_id, **pred.model_dump())
