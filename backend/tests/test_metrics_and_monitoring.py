"""TDD tests for Prometheus /metrics exposition and anonymized prediction logging."""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from backend.app.config import get_settings
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


def _make_image_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(90, 120, 150)).save(buf, format="JPEG")
    return buf.getvalue()


def test_prometheus_metrics_and_prediction_log_capture(
    monkeypatch, tmp_path: Path
) -> None:
    """Predictions increment Prometheus metrics and append privacy-safe JSONL logs."""
    fake_onnx = tmp_path / "detector.onnx"
    fake_onnx.write_bytes(b"onnx-placeholder")
    pred_log = tmp_path / "predictions.jsonl"

    raw_out = np.zeros((1, 9, 8400), dtype=np.float32)
    raw_out[0, :4, 0] = [320.0, 280.0, 80.0, 90.0]
    raw_out[0, 4 + 1, 0] = 0.91  # helmet

    monkeypatch.setattr(
        "backend.app.inference.ort.InferenceSession",
        _make_fake_session_class(raw_out),
    )
    monkeypatch.setenv("SAFEGUARD_MODEL_PATH", str(fake_onnx))
    monkeypatch.setenv("SAFEGUARD_PREDICTION_LOG_PATH", str(pred_log))
    get_settings.cache_clear()

    try:
        with TestClient(app) as client:
            jpg = _make_image_bytes()
            ok_res = client.post(
                "/predict", files={"file": ("site.jpg", jpg, "image/jpeg")}
            )
            assert ok_res.status_code == 200

            bad_res = client.post(
                "/predict", files={"file": ("bad.jpg", b"bad", "image/jpeg")}
            )
            assert bad_res.status_code == 400

            metrics_res = client.get("/metrics")
            assert metrics_res.status_code == 200
            body = metrics_res.text
            assert "safeguard_inference_requests_total" in body
            assert "safeguard_prediction_errors_total" in body
            assert "safeguard_inference_latency_seconds_bucket" in body
            assert (
                'safeguard_detections_by_class_total{class_name="helmet"} 1.0' in body
            )

        lines = pred_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["class_name"] == "helmet"
        assert rec["route"] == "/predict"
        assert 0.0 <= rec["confidence"] <= 1.0
        assert "image" not in rec
    finally:
        get_settings.cache_clear()
