"""Runtime configuration for the SafeGuard FastAPI service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the SafeGuard inference API."""

    env: str = Field(default="development", description="Runtime environment name.")
    api_title: str = Field(default="SafeGuard API", description="OpenAPI title.")
    api_version: str = Field(default="0.1.0", description="API semantic version.")
    log_level: str = Field(default="INFO", description="Application log level.")
    model_version: str = Field(
        default="unloaded",
        description="Version identifier of the loaded ONNX model.",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Allowed CORS origins for browser clients.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAFEGUARD_",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings instance."""
    return Settings()
