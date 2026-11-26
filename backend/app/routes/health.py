"""System health, model metadata, and canonical classes endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.app.routes.dependencies import InferenceServiceDep, SettingsDep
from backend.app.schemas import ClassesResponse, HealthResponse, ModelMetadataResponse

router = APIRouter(tags=["System & Metadata"])


@router.get(
    "/health",
    summary="Service and model readiness check",
)
def health(
    request: Request,
    current_settings: SettingsDep,
) -> HealthResponse:
    """Return server status, environment, and model readiness state."""
    loaded_version = getattr(
        request.app.state, "loaded_model_version", current_settings.model_version
    )
    return HealthResponse(
        status="ok",
        service=current_settings.api_title,
        environment=current_settings.env,
        model_loaded=bool(getattr(request.app.state, "model_loaded", False)),
        model_version=str(loaded_version),
        gpu_available=bool(getattr(request.app.state, "gpu_available", False)),
        runtime_provider=str(
            getattr(request.app.state, "runtime_provider", "uninitialized")
        ),
    )


@router.get(
    "/model/metadata",
    summary="Inspect loaded ONNX model metadata",
)
def model_metadata(
    service: InferenceServiceDep,
) -> ModelMetadataResponse:
    """Return architecture, opset, input/output shapes, and class mapping."""
    return service.get_metadata()


@router.get(
    "/classes",
    summary="List canonical PPE detection classes",
)
def classes(
    service: InferenceServiceDep,
) -> ClassesResponse:
    """Return ordered canonical class IDs and names."""
    return service.get_classes()
