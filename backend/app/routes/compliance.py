"""Compliance monitoring and anonymous tracking API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile

from backend.app.routes.dependencies import ComplianceServiceDep, RequestIdDep
from backend.app.schemas import (
    ComplianceMonitorResponse,
    ComplianceResetResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Compliance & Tracking"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unsupported image."},
        422: {"model": ErrorResponse, "description": "Request validation failure."},
        503: {"model": ErrorResponse, "description": "ONNX model unavailable."},
        500: {"model": ErrorResponse, "description": "Internal pipeline failure."},
    },
)


@router.post(
    "/predict/compliance",
    summary="Run full ONNX detection, anonymous tracking, and PPE compliance rules",
)
def predict_compliance(
    service: ComplianceServiceDep,
    request_id: RequestIdDep,
    file: Annotated[
        UploadFile, File(description="Frame image (JPEG, PNG, WEBP, BMP).")
    ],
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Logical stream identifier."),
    ] = "default",
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
    timestamp: Annotated[
        float | None,
        Query(ge=0.0, description="Optional monotonic frame timestamp in seconds."),
    ] = None,
) -> ComplianceMonitorResponse:
    """Execute Frame -> ONNX -> Tracking -> PPE Association -> Compliance -> Events."""
    payload = file.file.read()
    res = service.process_frame(
        payload=payload,
        content_type=file.content_type,
        request_id=request_id,
        stream_id=stream_id,
        frame_id=frame_id,
        conf_threshold=conf,
        iou_threshold=iou,
        timestamp=timestamp,
    )
    logger.info(
        "request_id=%s route=/predict/compliance stream=%s tracks=%d events=%d "
        "inf_ms=%.2f comp_ms=%.2f",
        request_id,
        stream_id,
        res.summary.tracked_persons,
        res.summary.emitted_event_count,
        res.timings_ms.inference_ms,
        res.compliance_ms,
    )
    return res


@router.post(
    "/compliance/reset",
    summary="Reset anonymous tracking and cooldown state for a stream",
)
def reset_compliance_stream(
    service: ComplianceServiceDep,
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Stream identifier to clear."),
    ] = "default",
) -> ComplianceResetResponse:
    """Clear active tracks, violation streaks, and cooldown timers for a stream."""
    return service.reset_stream(stream_id=stream_id)
