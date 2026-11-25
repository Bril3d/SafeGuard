"""SafeGuard FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings, get_settings
from backend.app.schemas import HealthResponse


SettingsDep = Annotated[Settings, Depends(get_settings)]


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown state."""
    application.state.model_loaded = False
    application.state.gpu_available = False
    application.state.runtime_provider = "uninitialized"
    yield


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "SafeGuard Real-Time Construction PPE Monitoring API " "(Foundation Phase)"
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
        return HealthResponse(
            status="ok",
            service=current_settings.api_title,
            environment=current_settings.env,
            model_loaded=bool(getattr(application.state, "model_loaded", False)),
            model_version=current_settings.model_version,
            gpu_available=bool(getattr(application.state, "gpu_available", False)),
            runtime_provider=str(
                getattr(application.state, "runtime_provider", "uninitialized")
            ),
        )

    return application


app = create_app()
