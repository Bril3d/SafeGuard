"""API unit and integration tests for the SafeGuard FastAPI backend."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from backend.app.config import get_settings
from backend.app.inference import ONNXDetector
from backend.app.main import app


def _make_fake_session_class(raw_output: np.ndarray | None = None):
    tensor_out = (
        raw_output
        if raw_output is not None
        else np.zeros((1, 9, 8400), dtype=np.float32)
    )

    class _FakeNodeArg:
        def __init__(self, name: str, shape: list[int], dtype: str) -> None:
            self.name = name
            self.shape = shape
            self.type = dtype

    class _FakeSession:
        def __init__(self, model_path: str, providers: list[str]) -> None:
            self.model_path = model_path
            self._providers = providers

        def get_inputs(self) -> list[_FakeNodeArg]:
            return [_FakeNodeArg("images", [1, 3, 640, 640], "tensor(float)")]

        def get_outputs(self) -> list[_FakeNodeArg]:
            return [_FakeNodeArg("output0", [1, 9, 8400], "tensor(float)")]

        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def run(self, *_args, **_kwargs):
            return [tensor_out]

    return _FakeSession


def _make_image_bytes(fmt: str = "JPEG", width: int = 640, height: int = 480) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(120, 160, 200))
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_health_and_unloaded_endpoints_return_503(monkeypatch) -> None:
    """Unloaded model reports healthy liveness but returns 503 on inference routes."""
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", "")
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "unloaded")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            h_res = client.get("/health")
            assert h_res.status_code == 200
            assert h_res.json()["model_loaded"] is False
            assert h_res.json()["runtime_provider"] == "uninitialized"

            m_res = client.get("/model/metadata")
            assert m_res.status_code == 503
            assert m_res.json()["error"]["code"] == "model_unavailable"

            c_res = client.get("/classes")
            assert c_res.status_code == 503

            p_res = client.post(
                "/predict",
                files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
            )
            assert p_res.status_code == 503
            assert p_res.json()["error"]["code"] == "model_unavailable"

            v_res = client.post(
                "/predict/visualize",
                files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
            )
            assert v_res.status_code == 503

            f_res = client.post(
                "/predict/frame",
                files={"file": ("frame.jpg", _make_image_bytes(), "image/jpeg")},
            )
            assert f_res.status_code == 503
    finally:
        get_settings.cache_clear()


def test_lifespan_handles_model_initialization_failure_gracefully(
    monkeypatch, tmp_path: Path
) -> None:
    """Corrupt ONNX load failure keeps /health up and surfaces 503 on /predict."""
    bad_onnx = tmp_path / "corrupt.onnx"
    bad_onnx.write_bytes(b"corrupt")

    def _boom(*_args, **_kwargs):
        raise RuntimeError("corrupt onnx protobuf")

    monkeypatch.setattr("backend.app.inference.ort.InferenceSession", _boom)
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(bad_onnx))
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "unloaded")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            h_res = client.get("/health")
            assert h_res.status_code == 200
            h_json = h_res.json()
            assert h_json["model_loaded"] is False
            assert h_json["runtime_provider"] == "error"

            p_res = client.post(
                "/predict",
                files={"file": ("test.jpg", _make_image_bytes(), "image/jpeg")},
            )
            assert p_res.status_code == 503
    finally:
        get_settings.cache_clear()


def test_predict_visualize_frame_and_metadata_with_loaded_session(
    monkeypatch, tmp_path: Path
) -> None:
    """Loaded session serves /model/metadata, /classes, /predict, /visualize, /frame."""
    fake_onnx = tmp_path / "detector.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")

    raw_out = np.zeros((1, 9, 8400), dtype=np.float32)
    raw_out[0, :4, 0] = [320.0, 280.0, 80.0, 90.0]
    raw_out[0, 4 + 1, 0] = 0.92  # class 1 = helmet
    raw_out[0, :4, 1] = [140.0, 220.0, 60.0, 70.0]
    raw_out[0, 4 + 4, 1] = 0.78  # class 4 = no_helmet

    monkeypatch.setattr(
        "backend.app.inference.ort.InferenceSession",
        _make_fake_session_class(raw_out),
    )
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "EXP-E-yolo11s-onnx")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            meta_res = client.get("/model/metadata")
            assert meta_res.status_code == 200
            meta = meta_res.json()
            assert meta["model_version"] == "EXP-E-yolo11s-onnx"
            assert meta["imgsz"] == 640
            assert meta["runtime_provider"] == "CPUExecutionProvider"
            assert meta["class_names"]["1"] == "helmet"

            cls_res = client.get("/classes")
            assert cls_res.status_code == 200
            assert len(cls_res.json()["classes"]) == 5

            jpg = _make_image_bytes("JPEG", 640, 480)
            pred_res = client.post(
                "/predict",
                files={"file": ("site.jpg", jpg, "image/jpeg")},
            )
            assert pred_res.status_code == 200
            pred = pred_res.json()
            assert pred["detection_count"] == 2
            assert pred["detections"][0]["class_name"] == "helmet"
            assert pred["detections"][1]["class_name"] == "no_helmet"
            assert pred["image_width"] == 640
            assert pred["image_height"] == 480
            assert pred["timings_ms"]["total_ms"] >= pred["timings_ms"]["inference_ms"]

            vis_res = client.post(
                "/predict/visualize?output_format=jpeg",
                files={"file": ("site.jpg", jpg, "image/jpeg")},
            )
            assert vis_res.status_code == 200
            assert vis_res.headers["content-type"] == "image/jpeg"
            assert vis_res.headers["x-detection-count"] == "2"
            decoded = cv2.imdecode(
                np.frombuffer(vis_res.content, np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded is not None
            assert decoded.shape == (480, 640, 3)

            frame_res = client.post(
                "/predict/frame?frame_id=cam-01&conf=0.85",
                files={"file": ("frame.jpg", jpg, "image/jpeg")},
            )
            assert frame_res.status_code == 200
            frame_json = frame_res.json()
            assert frame_json["frame_id"] == "cam-01"
            assert frame_json["detection_count"] == 1
            assert frame_json["detections"][0]["class_name"] == "helmet"
    finally:
        get_settings.cache_clear()


def test_predict_validation_errors_return_400_and_422(
    monkeypatch, tmp_path: Path
) -> None:
    """Invalid images return HTTP 400 and invalid query params return HTTP 422."""
    fake_onnx = tmp_path / "detector.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")
    monkeypatch.setattr(
        "backend.app.inference.ort.InferenceSession",
        _make_fake_session_class(),
    )
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            bad_bytes = client.post(
                "/predict",
                files={"file": ("broken.jpg", b"not-an-image", "image/jpeg")},
            )
            assert bad_bytes.status_code == 400
            assert bad_bytes.json()["error"]["code"] == "invalid_image"

            bad_mime = client.post(
                "/predict",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )
            assert bad_mime.status_code == 400

            empty_file = client.post(
                "/predict",
                files={"file": ("empty.jpg", b"", "image/jpeg")},
            )
            assert empty_file.status_code == 400

            bad_query = client.post(
                "/predict?conf=1.8",
                files={"file": ("ok.jpg", _make_image_bytes(), "image/jpeg")},
            )
            assert bad_query.status_code == 422
            assert bad_query.json()["error"]["code"] == "validation_error"
    finally:
        get_settings.cache_clear()


def test_singleton_detector_reused_across_requests(monkeypatch, tmp_path: Path) -> None:
    """ONNXDetector is constructed once in lifespan and never per request."""
    fake_onnx = tmp_path / "detector.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")
    init_calls = 0
    orig_init = ONNXDetector.__init__

    def _counting_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(
        "backend.app.inference.ort.InferenceSession",
        _make_fake_session_class(),
    )
    monkeypatch.setattr(ONNXDetector, "__init__", _counting_init)
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            jpg = _make_image_bytes()
            assert (
                client.post(
                    "/predict", files={"file": ("1.jpg", jpg, "image/jpeg")}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/predict", files={"file": ("2.jpg", jpg, "image/jpeg")}
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/predict/frame", files={"file": ("3.jpg", jpg, "image/jpeg")}
                ).status_code
                == 200
            )
        assert init_calls == 1
    finally:
        get_settings.cache_clear()


def test_openapi_docs_and_paths_accessible(monkeypatch) -> None:
    """Swagger UI and OpenAPI JSON expose all documented endpoints."""
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", "")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            docs_res = client.get("/docs")
            schema_res = client.get("/openapi.json")
        assert docs_res.status_code == 200
        assert schema_res.status_code == 200
        paths = schema_res.json()["paths"]
        for expected in (
            "/health",
            "/model/metadata",
            "/classes",
            "/predict",
            "/predict/visualize",
            "/predict/frame",
            "/predict/compliance",
            "/compliance/reset",
        ):
            assert expected in paths
    finally:
        get_settings.cache_clear()


def test_predict_compliance_pipeline_temporal_confirmation_and_cooldown(
    monkeypatch, tmp_path: Path
) -> None:
    """End-to-end compliance endpoint tracks IDs, confirms over N frames, and resets."""
    fake_onnx = tmp_path / "detector.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")

    # 640x480 letterboxed inside 640x640 has pad_top=80 -> y=130..430 maps to y=50..350.
    raw_out = np.zeros((1, 9, 8400), dtype=np.float32)
    raw_out[0, :4, 0] = [320.0, 280.0, 120.0, 300.0]  # person: x=260..380, y=50..350
    raw_out[0, 4 + 0, 0] = 0.92
    raw_out[0, :4, 1] = [320.0, 160.0, 50.0, 50.0]  # no_helmet in head zone
    raw_out[0, 4 + 4, 1] = 0.86

    init_calls = 0
    orig_init = ONNXDetector.__init__

    def _counting_init(self, *args, **kwargs):
        nonlocal init_calls
        init_calls += 1
        return orig_init(self, *args, **kwargs)

    monkeypatch.setattr(
        "backend.app.inference.ort.InferenceSession",
        _make_fake_session_class(raw_out),
    )
    monkeypatch.setattr(ONNXDetector, "__init__", _counting_init)
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("SAFEGUARD_COMPLIANCE_CONFIRM_FRAMES", "3")
    monkeypatch.setenv("SAFEGUARD_COMPLIANCE_COOLDOWN_SECONDS", "10.0")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            jpg = _make_image_bytes("JPEG", 640, 480)
            r1 = client.post(
                "/predict/compliance?stream_id=gate-1&timestamp=1.0",
                files={"file": ("f1.jpg", jpg, "image/jpeg")},
            )
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["summary"]["tracked_persons"] == 1
            assert d1["summary"]["pending_confirmation_count"] == 1
            assert d1["summary"]["emitted_event_count"] == 0
            tid = d1["tracks"][0]["track_id"]

            r2 = client.post(
                "/predict/compliance?stream_id=gate-1&timestamp=2.0",
                files={"file": ("f2.jpg", jpg, "image/jpeg")},
            )
            assert r2.status_code == 200
            assert r2.json()["tracks"][0]["track_id"] == tid
            assert r2.json()["summary"]["emitted_event_count"] == 0

            r3 = client.post(
                "/predict/compliance?stream_id=gate-1&timestamp=3.0",
                files={"file": ("f3.jpg", jpg, "image/jpeg")},
            )
            assert r3.status_code == 200
            d3 = r3.json()
            assert d3["summary"]["potential_violation_count"] == 1
            assert d3["summary"]["emitted_event_count"] == 1
            assert d3["events"][0]["event_type"] == "potential_no_helmet"
            assert d3["events"][0]["requires_human_review"] is True

            # Frame 4 inside 10s cooldown -> suppressed duplicate.
            r4 = client.post(
                "/predict/compliance?stream_id=gate-1&timestamp=5.0",
                files={"file": ("f4.jpg", jpg, "image/jpeg")},
            )
            assert r4.status_code == 200
            assert r4.json()["summary"]["emitted_event_count"] == 0
            assert r4.json()["summary"]["suppressed_duplicates"] == 1

            # Invalid image still returns 400.
            bad = client.post(
                "/predict/compliance?stream_id=gate-1",
                files={"file": ("bad.jpg", b"not-img", "image/jpeg")},
            )
            assert bad.status_code == 400

            reset_res = client.post("/compliance/reset?stream_id=gate-1")
            assert reset_res.status_code == 200
            assert reset_res.json()["reset"] is True
        assert init_calls == 1
    finally:
        get_settings.cache_clear()


def test_live_production_onnx_end_to_end_routes(monkeypatch) -> None:
    """Exercise real ONNX artifact across /health, /metadata, /predict, /visualize."""
    onnx_path = Path("models/production/safeguard_yolo11s.onnx")
    sample_dir = Path("data/processed/val_ultralytics/images")
    if not onnx_path.exists() or not sample_dir.exists():
        pytest.skip("Production ONNX artifact or validation image not present.")

    sample_img = next(sample_dir.glob("*.jpg"))
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(onnx_path))
    monkeypatch.setenv("SAFEGUARD_MODEL_VERSION", "unloaded")
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            h_res = client.get("/health")
            assert h_res.status_code == 200
            assert h_res.json()["model_loaded"] is True
            assert h_res.json()["model_version"] == "EXP-E-yolo11s-onnx"

            m_res = client.get("/model/metadata")
            assert m_res.status_code == 200
            assert m_res.json()["opset"] == 17
            assert m_res.json()["architecture"] == "YOLO11s"

            raw = sample_img.read_bytes()
            p_res = client.post(
                "/predict",
                files={"file": (sample_img.name, raw, "image/jpeg")},
            )
            assert p_res.status_code == 200
            p_data = p_res.json()
            assert p_data["timings_ms"]["inference_ms"] > 0.0

            v_res = client.post(
                "/predict/visualize",
                files={"file": (sample_img.name, raw, "image/jpeg")},
            )
            assert v_res.status_code == 200
            assert v_res.headers["content-type"] == "image/jpeg"

            f_res = client.post(
                "/predict/frame?frame_id=live-01",
                files={"file": (sample_img.name, raw, "image/jpeg")},
            )
            assert f_res.status_code == 200
            assert f_res.json()["frame_id"] == "live-01"
    finally:
        get_settings.cache_clear()
