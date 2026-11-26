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
    model_path: str | None = Field(
        default="models/production/safeguard_yolo11s.onnx",
        description="Filesystem path to the exported ONNX inference model.",
    )
    model_metadata_path: str | None = Field(
        default=None,
        description="Optional override path to model metadata JSON.",
    )
    conf_threshold: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Default confidence threshold for ONNX predictions.",
    )
    iou_threshold: float = Field(
        default=0.50,
        ge=0.05,
        le=0.95,
        description="Default NMS IoU threshold for ONNX predictions.",
    )
    max_upload_bytes: int = Field(
        default=10 * 1024 * 1024,
        gt=0,
        description="Maximum accepted image upload payload in bytes (default 10 MB).",
    )
    max_image_dimension: int = Field(
        default=4096,
        gt=0,
        description="Maximum accepted image width or height in pixels.",
    )
    compliance_confirm_frames: int = Field(
        default=3,
        ge=1,
        description="Consecutive frames required before emitting a potential alert.",
    )
    compliance_cooldown_seconds: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum seconds between repeated alerts for the same track.",
    )
    compliance_ambiguous_margin: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence margin within which conflicting helmet/no_helmet is ambiguous."
        ),
    )
    compliance_spatial_ambiguity_margin: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum spatial score margin required between overlapping workers."
        ),
    )
    compliance_capture_evidence: bool = Field(
        default=True,
        description=(
            "Whether emitted alert events include an annotated JPEG evidence frame."
        ),
    )
    tracker_max_age: int = Field(
        default=15,
        ge=1,
        description="Max missed frames to retain an occluded anonymous track ID.",
    )
    tracker_high_conf: float = Field(
        default=0.45,
        ge=0.05,
        le=1.0,
        description="ByteTrack first-stage high-confidence person threshold.",
    )
    tracker_low_conf: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        description=(
            "ByteTrack second-stage low-confidence occlusion recovery threshold."
        ),
    )
    tracker_iou_threshold: float = Field(
        default=0.25,
        ge=0.05,
        le=0.95,
        description="Minimum IoU for primary track-to-detection matching.",
    )
    prediction_log_path: str | None = Field(
        default=None,
        description=(
            "Optional filesystem path for anonymized JSONL prediction telemetry."
        ),
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
