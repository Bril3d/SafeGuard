"""SafeGuard FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.inference import ONNXDetector
from backend.app.schemas import HealthResponse

logger = logging.getLogger(__name__)
SettingsDep = Annotated[Settings, Depends(get_settings)]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the verified ONNX artifact once at startup and track readiness."""
    settings = get_settings()
    application.state.detector = None
    application.state.model_loaded = False
    application.state.gpu_available = False
    application.state.runtime_provider = "uninitialized"
    application.state.loaded_model_version = settings.model_version

    model_path_str = (settings.model_path or "").strip()
    if model_path_str:
        model_file = Path(model_path_str)
        if model_file.exists():
            detector = ONNXDetector(
                onnx_path=model_file,
                metadata_path=settings.model_metadata_path,
                default_conf=settings.conf_threshold,
                default_iou=settings.iou_threshold,
            )
            application.state.detector = detector
            application.state.model_loaded = True
            application.state.gpu_available = detector.gpu_available
            application.state.runtime_provider = detector.active_provider
            application.state.loaded_model_version = (
                settings.model_version
                if settings.model_version != "unloaded"
                else detector.model_version
            )
            logger.info(
                "Loaded ONNX model %s via %s in %.2f ms",
                model_file,
                detector.active_provider,
                detector.load_time_ms,
            )
        else:
            logger.warning(
                "Configured ONNX artifact %s does not exist; starting unloaded.",
                model_file,
            )

    yield
    application.state.detector = None
    application.state.model_loaded = False


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "SafeGuard Real-Time Construction PPE Monitoring API " "(ONNX Runtime)"
        ),
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @application.get(
        "/health",
        tags=["System"],
        summary="Service and model readiness check",
    )
    async def health(
        current_settings: SettingsDep,
    ) -> HealthResponse:
        """Return server status, environment, and model readiness state."""
        loaded_version = getattr(
            application.state, "loaded_model_version", current_settings.model_version
        )
        return HealthResponse(
            status="ok",
            service=current_settings.api_title,
            environment=current_settings.env,
            model_loaded=bool(getattr(application.state, "model_loaded", False)),
            model_version=str(loaded_version),
            gpu_available=bool(getattr(application.state, "gpu_available", False)),
            runtime_provider=str(
                getattr(application.state, "runtime_provider", "uninitialized")
            ),
        )

    return application


app = create_app()
