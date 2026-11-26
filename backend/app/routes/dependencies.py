"""Shared FastAPI dependencies for configuration, request IDs, and inference."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.app.config import Settings, get_settings
from backend.app.services.inference_service import InferenceService


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


def get_request_id(request: Request) -> str:
    """Return correlation request ID assigned by middleware."""
    return str(getattr(request.state, "request_id", "unknown"))


SettingsDep = Annotated[Settings, Depends(get_settings)]
InferenceServiceDep = Annotated[InferenceService, Depends(get_inference_service)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
