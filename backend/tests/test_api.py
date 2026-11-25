"""Foundation tests for the SafeGuard FastAPI service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.main import app


def test_health_endpoint_returns_expected_contract() -> None:
    """GET /health returns 200 and matches the foundation health schema."""
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "SafeGuard API"
    assert payload["environment"] == "development"
    assert payload["model_loaded"] is False
    assert payload["model_version"] == "unloaded"
    assert payload["gpu_available"] is False
    assert payload["runtime_provider"] == "uninitialized"


def test_settings_can_be_overridden_via_environment(monkeypatch) -> None:
    """SAFEGUARD_* environment variables override default backend settings."""
    monkeypatch.setenv("SAFEGUARD_ENV", "testing")
    monkeypatch.setenv("SAFEGUARD_API_TITLE", "SafeGuard Test API")
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "0.0.1-test")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["environment"] == "testing"
        assert payload["service"] == "SafeGuard Test API"
        assert payload["model_version"] == "0.0.1-test"
    finally:
        get_settings.cache_clear()


def test_openapi_docs_accessible() -> None:
    """Swagger UI and OpenAPI JSON schema are exposed at /docs and /openapi.json."""
    with TestClient(app) as client:
        docs_response = client.get("/docs")
        schema_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert schema_response.status_code == 200
    assert "/health" in schema_response.json()["paths"]
