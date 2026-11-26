"""SafeGuard FastAPI application entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.inference import ONNXDetector
from backend.app.routes.compliance import router as compliance_router
from backend.app.routes.health import router as health_router
from backend.app.routes.inference import router as inference_router
from backend.app.schemas import ErrorDetail, ErrorResponse
from backend.app.services.compliance_service import CompliancePipelineService
from backend.app.services.inference_service import (
    ImageValidationError,
    InferenceService,
    ModelUnavailableError,
)
from src.compliance.rules import ComplianceConfig
from src.tracking.tracker import TrackerConfig

logger = logging.getLogger(__name__)


def _error_json(
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    details: list[dict[str, str]] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the verified ONNX artifact once at startup and handle load failures."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    detector: ONNXDetector | None = None
    application.state.detector = None
    application.state.model_loaded = False
    application.state.gpu_available = False
    application.state.runtime_provider = "uninitialized"
    application.state.loaded_model_version = settings.model_version
    application.state.model_error = None

    model_path_str = (settings.model_path or "").strip()
    if model_path_str:
        model_file = Path(model_path_str)
        if model_file.exists():
            try:
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
            except Exception as exc:
                application.state.runtime_provider = "error"
                application.state.model_error = str(exc)
                logger.exception("Failed to initialize ONNX model %s", model_file)
        else:
            logger.warning(
                "Configured ONNX artifact %s does not exist; starting unloaded.",
                model_file,
            )

    inf_service = InferenceService(
        detector=detector,
        model_version=str(application.state.loaded_model_version),
        max_upload_bytes=settings.max_upload_bytes,
        max_image_dimension=settings.max_image_dimension,
    )
    application.state.inference_service = inf_service
    application.state.compliance_service = CompliancePipelineService(
        inference_service=inf_service,
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

    yield
    application.state.detector = None
    application.state.inference_service = None
    application.state.compliance_service = None
    application.state.model_loaded = False


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    settings = get_settings()
    application = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=(
            "SafeGuard Real-Time Construction PPE Monitoring API (ONNX Runtime)"
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

    @application.middleware("http")
    async def attach_request_id(request: Request, call_next):
        req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers.setdefault("X-Request-Id", req_id)
        return response

    @application.exception_handler(ImageValidationError)
    async def handle_invalid_image(request: Request, exc: ImageValidationError):
        req_id = str(getattr(request.state, "request_id", "unknown"))
        return _error_json(
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_image",
            message=str(exc),
            request_id=req_id,
        )

    @application.exception_handler(ModelUnavailableError)
    async def handle_model_unavailable(request: Request, exc: ModelUnavailableError):
        req_id = str(getattr(request.state, "request_id", "unknown"))
        return _error_json(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="model_unavailable",
            message=str(exc),
            request_id=req_id,
        )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        req_id = str(getattr(request.state, "request_id", "unknown"))
        details = [
            {
                "loc": ".".join(str(part) for part in err.get("loc", ())),
                "msg": str(err.get("msg", "")),
                "type": str(err.get("type", "")),
            }
            for err in exc.errors()
        ]
        return _error_json(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="validation_error",
            message="Request validation failed.",
            request_id=req_id,
            details=details,
        )

    @application.exception_handler(HTTPException)
    async def handle_http_exception(request: Request, exc: HTTPException):
        req_id = str(getattr(request.state, "request_id", "unknown"))
        return _error_json(
            status_code=exc.status_code,
            code=f"http_{exc.status_code}",
            message=str(exc.detail),
            request_id=req_id,
        )

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, _exc: Exception):
        req_id = str(getattr(request.state, "request_id", "unknown"))
        logger.exception("Unhandled exception on request_id=%s", req_id)
        return _error_json(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message="An internal server error occurred.",
            request_id=req_id,
        )

    application.include_router(health_router)
    application.include_router(inference_router)
    application.include_router(compliance_router)
    return application


app = create_app()
