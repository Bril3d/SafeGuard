"""Foundation and ONNX model-readiness tests for the SafeGuard FastAPI service."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app.config import get_settings
from backend.app.main import app


def test_health_endpoint_when_model_disabled_returns_unloaded(monkeypatch) -> None:
    """GET /health returns 200 and reports unloaded state when no model is active."""
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", "")
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "unloaded")
    get_settings.cache_clear()

    try:
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
    finally:
        get_settings.cache_clear()


def test_settings_can_be_overridden_via_environment(monkeypatch) -> None:
    """SAFEGUARD_* environment variables override default backend settings."""
    monkeypatch.setenv("SAFEGUARD_ENV", "testing")
    monkeypatch.setenv("SAFEGUARD_API_TITLE", "SafeGuard Test API")
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "0.0.1-test")
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", "")
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


def test_health_endpoint_loads_onnx_session_via_lifespan(
    monkeypatch, tmp_path: Path
) -> None:
    """Lifespan initializes ONNX Runtime once when configured artifact exists."""
    fake_onnx = tmp_path / "model.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")

    class _FakeNodeArg:
        def __init__(self, name: str, shape: list[int]) -> None:
            self.name = name
            self.shape = shape

    class _FakeSession:
        def __init__(self, model_path: str, providers: list[str]) -> None:
            self.model_path = model_path
            self._providers = providers

        def get_inputs(self) -> list[_FakeNodeArg]:
            return [_FakeNodeArg("images", [1, 3, 640, 640])]

        def get_outputs(self) -> list[_FakeNodeArg]:
            return [_FakeNodeArg("output0", [1, 9, 8400])]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, *_args, **_kwargs):
            return [np.zeros((1, 9, 8400), dtype=np.float32)]

    monkeypatch.setattr("backend.app.inference.ort.InferenceSession", _FakeSession)
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "EXP-E-yolo11s-onnx")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_loaded"] is True
        assert payload["model_version"] == "EXP-E-yolo11s-onnx"
        assert payload["runtime_provider"] == "CPUExecutionProvider"
        assert payload["gpu_available"] is False
    finally:
        get_settings.cache_clear()


def test_health_endpoint_loads_real_onnx_artifact_when_onnx_available(
    monkeypatch, tmp_path: Path
) -> None:
    """Real ONNX artifact created with onnx loads cleanly through ORT in lifespan."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    onnx_path = tmp_path / "test_detector.onnx"
    bias = np.zeros((1, 9, 8400), dtype=np.float32)
    c_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=helper.make_tensor(
            "b", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()
        ),
    )
    graph = helper.make_graph(
        nodes=[c_node],
        name="dummy_health_check_onnx",
        inputs=[
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
        ],
        outputs=[
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 9, 8400])
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.save(model, str(onnx_path))

    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(onnx_path))
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "EXP-E-yolo11s-onnx")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["model_loaded"] is True
        assert payload["model_version"] == "EXP-E-yolo11s-onnx"
        assert payload["runtime_provider"] == "CPUExecutionProvider"
    finally:
        get_settings.cache_clear()


def test_openapi_docs_accessible(monkeypatch) -> None:
    """Swagger UI and OpenAPI JSON schema are exposed at /docs and /openapi.json."""
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", "")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            docs_response = client.get("/docs")
            schema_response = client.get("/openapi.json")
        assert docs_response.status_code == 200
        assert schema_response.status_code == 200
        assert "/health" in schema_response.json()["paths"]
    finally:
        get_settings.cache_clear()
