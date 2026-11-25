"""Strict Pydantic v2 schemas for the SafeGuard FastAPI service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = Field(description="Service liveness status.")
    service: str = Field(description="Configured service name.")
    environment: str = Field(description="Active runtime environment.")
    model_loaded: bool = Field(
        description="Whether an ONNX inference session is initialized."
    )
    model_version: str = Field(
        description="Loaded model version or 'unloaded' during foundation phase."
    )
    gpu_available: bool = Field(
        description="Whether a GPU execution provider is active."
    )
    runtime_provider: str = Field(
        description="Active ONNX Runtime provider or 'uninitialized'."
    )
