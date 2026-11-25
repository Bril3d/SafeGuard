"""TDD unit tests for SafeGuard YOLO training, evaluation, and error analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from src.models.benchmark import summarize_latencies
from src.models.error_analysis import analyze_image_errors, compute_iou_xyxy
from src.models.train import load_experiment_config, resolve_run_spec


def test_load_experiment_config_covers_smoke_and_exp_a_through_f() -> None:
    cfg = load_experiment_config(Path("configs/training.yaml"))
    expected = {"SMOKE", "EXP-A", "EXP-B", "EXP-C", "EXP-D", "EXP-E", "EXP-F"}
    assert expected.issubset(set(cfg["experiments"].keys()))


def test_resolve_run_spec_applies_overrides_and_canonical_taxonomy(
    tmp_path: Path,
) -> None:
    cfg_path = Path("configs/training.yaml")
    spec = resolve_run_spec(
        config_path=cfg_path,
        experiment_id="EXP-C",
        epochs=5,
        batch_size=8,
        imgsz=512,
        seed=123,
    )
    assert spec.experiment_id == "EXP-C"
    assert spec.architecture == "YOLO11n"
    assert spec.epochs == 5
    assert spec.batch_size == 8
    assert spec.imgsz == 512
    assert spec.seed == 123
    assert len(spec.class_map_sha256) == 64
    assert spec.canonical_names == ["person", "helmet", "vest", "gloves", "no_helmet"]


def test_error_analysis_identifies_tp_fp_fn_and_small_object_misses() -> None:
    gt_boxes = [
        {"class_id": 0, "bbox_xyxy": [10.0, 10.0, 110.0, 210.0], "area_norm": 0.05},
        {"class_id": 3, "bbox_xyxy": [200.0, 200.0, 215.0, 215.0], "area_norm": 0.0005},
    ]
    pred_boxes = [
        {"class_id": 0, "bbox_xyxy": [12.0, 12.0, 108.0, 208.0], "confidence": 0.9},
        {"class_id": 1, "bbox_xyxy": [300.0, 300.0, 340.0, 340.0], "confidence": 0.7},
    ]

    iou = compute_iou_xyxy([10, 10, 110, 210], [12, 12, 108, 208])
    assert iou > 0.85

    summary = analyze_image_errors(gt_boxes, pred_boxes, iou_threshold=0.5)
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["small_object_fn"] == 1


def test_benchmark_latency_summary_metrics() -> None:
    latencies = [10.0, 12.0, 14.0, 16.0, 28.0]
    stats = summarize_latencies(latencies)
    assert stats["mean_ms"] == 16.0
    assert stats["p95_ms"] >= 20.0
    assert stats["fps"] > 0.0
    assert isinstance(np.isfinite(stats["mean_ms"]), (bool, np.bool_))
    assert yaml is not None
