"""Compliance monitoring, alerts, and anonymous tracking API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Body,
    File,
    HTTPException,
    Path,
    Query,
    UploadFile,
    status,
)

from backend.app.routes.dependencies import ComplianceServiceDep, RequestIdDep
from backend.app.schemas import (
    AlertStatusUpdateRequest,
    AlertStatusUpdateResponse,
    ComplianceAlertsResponse,
    ComplianceMonitorResponse,
    ComplianceResetResponse,
    ErrorResponse,
    StreamAnalyticsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Compliance & Tracking"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request or image."},
        404: {"model": ErrorResponse, "description": "Alert event not found."},
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
    """Execute Raw -> Tracking -> Compliance -> Alerts + Analytics."""
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


@router.get(
    "/compliance/alerts",
    summary="List retained potential compliance alerts with evidence snapshots",
)
def get_compliance_alerts(
    service: ComplianceServiceDep,
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Logical stream identifier."),
    ] = "default",
    event_status: Annotated[
        str | None,
        Query(description="Optional status filter (e.g. pending_review)."),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Maximum number of recent alerts."),
    ] = 50,
) -> ComplianceAlertsResponse:
    """Return recent alert events (newest first) for human review."""
    return service.get_alerts(stream_id=stream_id, status=event_status, limit=limit)


@router.patch(
    "/compliance/alerts/{event_id}",
    summary="Update human review status of a potential compliance alert",
)
def update_compliance_alert_status(
    service: ComplianceServiceDep,
    event_id: Annotated[str, Path(min_length=1, max_length=64)],
    body: Annotated[AlertStatusUpdateRequest, Body()],
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Logical stream identifier."),
    ] = "default",
) -> AlertStatusUpdateResponse:
    """Transition an alert event_status (e.g. pending_review -> verified/dismissed)."""
    try:
        updated = service.update_alert_status(
            stream_id=stream_id,
            event_id=event_id,
            new_status=body.event_status,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert event '{event_id}' was not found on stream '{stream_id}'.",
        )
    return updated


@router.get(
    "/compliance/analytics",
    summary="Fetch cumulative worker tracking, PPE, and alert trend analytics",
)
def get_compliance_analytics(
    service: ComplianceServiceDep,
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Logical stream identifier."),
    ] = "default",
) -> StreamAnalyticsResponse:
    """Return observed workers, PPE counts, violation categories, and timeline."""
    return service.get_analytics(stream_id=stream_id)


@router.post(
    "/compliance/reset",
    summary="Reset anonymous tracking, alerts, and analytics state for a stream",
)
def reset_compliance_stream(
    service: ComplianceServiceDep,
    stream_id: Annotated[
        str,
        Query(min_length=1, max_length=64, description="Stream identifier to clear."),
    ] = "default",
) -> ComplianceResetResponse:
    """Clear active tracks, violation streaks, cooldown timers, and analytics."""
    return service.reset_stream(stream_id=stream_id)
