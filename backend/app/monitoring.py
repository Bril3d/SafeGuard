"""Prometheus operational metrics and privacy-safe prediction logging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from backend.app.inference import DetectionResult


@dataclass
class OperationalMonitor:
    """Thread-safe Prometheus + bounded JSONL operational telemetry recorder."""

    log_path: Path | None = None
    registry: CollectorRegistry | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._reg = self.registry or CollectorRegistry(auto_describe=True)
        self.requests_total = Counter(
            "safeguard_inference_requests_total",
            "Total HTTP requests handled by SafeGuard inference routes.",
            labelnames=("route", "status_code"),
            registry=self._reg,
        )
        self.errors_total = Counter(
            "safeguard_prediction_errors_total",
            "Total inference and validation errors by route and error code.",
            labelnames=("route", "error_code"),
            registry=self._reg,
        )
        self.inference_latency_seconds = Histogram(
            "safeguard_inference_latency_seconds",
            "ONNX model inference latency in seconds.",
            labelnames=("route",),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5),
            registry=self._reg,
        )
        self.detections_by_class_total = Counter(
            "safeguard_detections_by_class_total",
            "Cumulative detections emitted per canonical class.",
            labelnames=("class_name",),
            registry=self._reg,
        )
        self.prediction_confidence = Histogram(
            "safeguard_prediction_confidence",
            "Proxy confidence distribution of returned detections.",
            buckets=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0),
            registry=self._reg,
        )

    def record_prediction(
        self,
        route: str,
        status_code: int,
        model_version: str,
        inference_ms: float,
        image_width: int,
        image_height: int,
        detections: Iterable[DetectionResult],
        request_id: str,
    ) -> None:
        """Observe Prometheus metrics and append anonymized stats-only JSONL entry."""
        det_list = list(detections)
        self.requests_total.labels(route=route, status_code=str(status_code)).inc()
        self.inference_latency_seconds.labels(route=route).observe(
            max(0.0, inference_ms / 1000.0)
        )
        for det in det_list:
            self.detections_by_class_total.labels(class_name=det.class_name).inc()
            self.prediction_confidence.observe(float(det.confidence))

        if self.log_path is None:
            return

        img_area = max(1.0, float(image_width * image_height))
        records = [
            {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "route": route,
                "model_version": model_version,
                "inference_ms": round(float(inference_ms), 3),
                "class_name": d.class_name,
                "confidence": round(float(d.confidence), 4),
                "bbox_area_norm": round(
                    max(0.0, d.bbox_xyxy[2] - d.bbox_xyxy[0])
                    * max(0.0, d.bbox_xyxy[3] - d.bbox_xyxy[1])
                    / img_area,
                    6,
                ),
            }
            for d in det_list
        ]
        if not records:
            records = [
                {
                    "ts_utc": datetime.now(timezone.utc).isoformat(),
                    "request_id": request_id,
                    "route": route,
                    "model_version": model_version,
                    "inference_ms": round(float(inference_ms), 3),
                    "class_name": "none",
                    "confidence": 0.0,
                    "bbox_area_norm": 0.0,
                }
            ]

        self._write_lines(records)

    def record_error(self, route: str, status_code: int, error_code: str) -> None:
        """Increment Prometheus error and failed-request counters."""
        self.requests_total.labels(route=route, status_code=str(status_code)).inc()
        self.errors_total.labels(route=route, error_code=error_code).inc()

    def _write_lines(self, items: list[dict[str, object]]) -> None:
        assert self.log_path is not None
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item) + "\n")

    def render_prometheus(self) -> tuple[bytes, str]:
        """Serialize Prometheus text exposition payload and media type."""
        return generate_latest(self._reg), CONTENT_TYPE_LATEST
