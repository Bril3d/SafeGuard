"""Shared FastAPI dependencies for configuration, request IDs, and services."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.app.config import Settings, get_settings
from backend.app.services.compliance_service import CompliancePipelineService
from backend.app.services.inference_service import InferenceService
from src.compliance.rules import ComplianceConfig
from src.tracking.tracker import TrackerConfig


def get_inference_service(request: Request) -> InferenceService:
    """Build or retrieve request-scoped InferenceService wrapping app state."""
    existing = getattr(request.app.state, "inference_service", None)
    if existing is not None:
        return existing
    settings = get_settings()
    return InferenceService(
        detector=getattr(request.app.state, "detector", None),
        model_version=str(
            getattr(request.app.state, "loaded_model_version", settings.model_version)
        ),
        max_upload_bytes=settings.max_upload_bytes,
        max_image_dimension=settings.max_image_dimension,
    )


def get_compliance_service(request: Request) -> CompliancePipelineService:
    """Retrieve singleton CompliancePipelineService initialized in lifespan."""
    existing = getattr(request.app.state, "compliance_service", None)
    if existing is not None:
        return existing
    settings = get_settings()
    return CompliancePipelineService(
        inference_service=get_inference_service(request),
        compliance_config=ComplianceConfig(
            confirm_frames=settings.compliance_confirm_frames,
            cooldown_seconds=settings.compliance_cooldown_seconds,
            ambiguous_margin=settings.compliance_ambiguous_margin,
        ),
        tracker_config=TrackerConfig(
            high_conf_threshold=settings.tracker_high_conf,
            low_conf_threshold=settings.tracker_low_conf,
            match_iou_threshold=settings.tracker_iou_threshold,
            max_age=settings.tracker_max_age,
        ),
    )


def get_request_id(request: Request) -> str:
    """Return correlation request ID assigned by middleware."""
    return str(getattr(request.state, "request_id", "unknown"))


SettingsDep = Annotated[Settings, Depends(get_settings)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
ComplianceServiceDep = Annotated[
    CompliancePipelineService, Depends(get_compliance_service)
]
RequestIdDep = Annotated[str, Depends(get_request_id)]
