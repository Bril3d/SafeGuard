"""TDD unit tests for SafeGuard ONNX inference, parity checks, and benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from backend.app.inference import (
    ONNXDetector,
    letterbox_image,
    nms_boxes,
    postprocess_yolo_output,
    preprocess_image,
)
from src.models.export_onnx import (
    compare_detection_sets,
    select_best_experiment_from_records,
    verify_threshold_monotonicity,
)


def _create_synthetic_yolo_onnx(onnx_path: Path, num_classes: int = 5) -> None:
    """Create a tiny valid ONNX graph with YOLO-compatible (1, 4+nc, 8400) output."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    # Output tensor filled deterministically: take input mean and add fixed bias tensor
    # so inference exercises real ONNX Runtime execution without needing torch.
    bias = np.zeros((1, 4 + num_classes, 8400), dtype=np.float32)
    # Candidate 0: class 1 (helmet) near center (320, 320, 80, 90)
    bias[0, 0, 0] = 320.0
    bias[0, 1, 0] = 320.0
    bias[0, 2, 0] = 80.0
    bias[0, 3, 0] = 90.0
    bias[0, 4 + 1, 0] = 0.85
    # Candidate 1: overlapping class 1 box for NMS test (score 0.60)
    bias[0, 0, 1] = 322.0
    bias[0, 1, 1] = 321.0
    bias[0, 2, 1] = 78.0
    bias[0, 3, 1] = 88.0
    bias[0, 4 + 1, 1] = 0.60
    # Candidate 2: class 4 (no_helmet) separate box (score 0.40)
    bias[0, 0, 2] = 120.0
    bias[0, 1, 2] = 140.0
    bias[0, 2, 2] = 40.0
    bias[0, 3, 2] = 50.0
    bias[0, 4 + 4, 2] = 0.40

    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output0"],
        value=helper.make_tensor(
            "bias_tensor", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()
        ),
    )
    # Wire `images` into graph with zero contribution so ORT requires `images` input.
    zero_mul = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["zero_scalar"],
        value=helper.make_tensor("zero_val", TensorProto.FLOAT, [1], [0.0]),
    )
    reduce_node = helper.make_node(
        "ReduceMean",
        inputs=["images"],
        outputs=["img_mean"],
        axes=[0, 1, 2, 3],
        keepdims=0,
    )
    mul_node = helper.make_node(
        "Mul", inputs=["img_mean", "zero_scalar"], outputs=["zeroed"]
    )
    add_node = helper.make_node(
        "Add", inputs=["bias_tensor_out", "zeroed"], outputs=["output0"]
    )
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["bias_tensor_out"],
        value=helper.make_tensor(
            "bias_tensor", TensorProto.FLOAT, bias.shape, bias.flatten().tolist()
        ),
    )

    graph = helper.make_graph(
        nodes=[const_node, zero_mul, reduce_node, mul_node, add_node],
        name="synthetic_yolo_detector",
        inputs=[
            helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
        ],
        outputs=[
            helper.make_tensor_value_info(
                "output0", TensorProto.FLOAT, [1, 4 + num_classes, 8400]
            )
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(onnx_path))


def test_letterbox_preserves_aspect_ratio_and_unscales_accurately() -> None:
    landscape = Image.new("RGB", (800, 400), color=(10, 20, 30))
    lb_img, meta = letterbox_image(landscape, target_size=640)
    assert lb_img.size == (640, 640)
    assert meta.orig_width == 800
    assert meta.orig_height == 400
    assert meta.new_width == 640
    assert meta.new_height == 320
    assert meta.pad_top == 160
    assert meta.pad_left == 0

    tensor, tensor_meta = preprocess_image(landscape, target_size=640)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert tensor_meta == meta


def test_postprocess_yolo_output_applies_conf_nms_and_clipping() -> None:
    landscape = Image.new("RGB", (800, 400))
    _, meta = letterbox_image(landscape, target_size=640)
    raw = np.zeros((1, 9, 8400), dtype=np.float32)
    # High-confidence helmet + overlapping lower-confidence helmet
    raw[0, :4, 0] = [320.0, 320.0, 80.0, 90.0]
    raw[0, 5, 0] = 0.85
    raw[0, :4, 1] = [322.0, 321.0, 78.0, 88.0]
    raw[0, 5, 1] = 0.65
    # Distinct class 4 (no_helmet)
    raw[0, :4, 2] = [100.0, 240.0, 40.0, 40.0]
    raw[0, 8, 2] = 0.30

    names = {0: "person", 1: "helmet", 2: "vest", 3: "gloves", 4: "no_helmet"}
    dets = postprocess_yolo_output(
        raw, meta, names, conf_threshold=0.25, iou_threshold=0.50
    )
    assert len(dets) == 2
    assert dets[0].class_name == "helmet"
    assert dets[0].confidence == pytest.approx(0.85, abs=1e-5)
    assert dets[1].class_name == "no_helmet"
    for d in dets:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0.0 <= x1 < x2 <= 800.0
        assert 0.0 <= y1 < y2 <= 400.0


def test_nms_boxes_respects_iou_threshold() -> None:
    boxes = np.array([[0, 0, 100, 100], [10, 10, 105, 105]], dtype=np.float32)
    scores = np.array([0.9, 0.7], dtype=np.float32)
    assert nms_boxes(boxes, scores, iou_threshold=0.50).tolist() == [0]
    assert nms_boxes(boxes, scores, iou_threshold=0.95).tolist() == [0, 1]


def test_onnx_detector_runs_synthetic_artifact_and_reads_metadata(
    tmp_path: Path,
) -> None:
    onnx_path = tmp_path / "model.onnx"
    _create_synthetic_yolo_onnx(onnx_path, num_classes=5)
    meta_path = onnx_path.with_suffix(".metadata.json")
    meta_path.write_text(
        json.dumps(
            {
                "model_version": "EXP-E-test",
                "class_names": {
                    "0": "person",
                    "1": "helmet",
                    "2": "vest",
                    "3": "gloves",
                    "4": "no_helmet",
                },
                "imgsz": 640,
            }
        ),
        encoding="utf-8",
    )

    detector = ONNXDetector(onnx_path=onnx_path, metadata_path=meta_path)
    assert detector.model_version == "EXP-E-test"
    assert detector.input_shape == [1, 3, 640, 640]
    assert detector.output_shape == [1, 9, 8400]

    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    out_default = detector.predict_image(img, conf_threshold=0.25, iou_threshold=0.50)
    assert len(out_default.detections) == 2
    out_strict = detector.predict_image(img, conf_threshold=0.70, iou_threshold=0.50)
    assert len(out_strict.detections) == 1
    assert out_strict.detections[0].class_name == "helmet"
    assert (
        out_default.timings_ms["end_to_end_ms"]
        >= out_default.timings_ms["inference_ms"]
    )


def test_select_best_experiment_excludes_smoke_and_picks_highest_score() -> None:
    records = [
        {
            "experiment_id": "SMOKE",
            "run_id": "r0",
            "val_mAP50": 0.99,
            "ultralytics_test_mAP50": 0.99,
        },
        {
            "experiment_id": "EXP-D",
            "run_id": "r1",
            "val_mAP50": 0.814,
            "ultralytics_test_mAP50": 0.821,
            "roboflow_test_mAP50": 0.534,
        },
        {
            "experiment_id": "EXP-E",
            "run_id": "r2",
            "val_mAP50": 0.819,
            "ultralytics_test_mAP50": 0.8335,
            "roboflow_test_mAP50": 0.561,
        },
    ]
    best = select_best_experiment_from_records(records)
    assert best["experiment_id"] == "EXP-E"
    assert best["run_id"] == "r2"


def test_compare_detection_sets_and_threshold_monotonicity() -> None:
    pt = [
        {
            "class_id": 1,
            "class_name": "helmet",
            "confidence": 0.88,
            "bbox_xyxy": [10.0, 20.0, 50.0, 60.0],
        }
    ]
    ox = [
        {
            "class_id": 1,
            "class_name": "helmet",
            "confidence": 0.879,
            "bbox_xyxy": [10.4, 20.1, 49.8, 60.2],
        }
    ]
    parity = compare_detection_sets(pt, ox, iou_match_threshold=0.5)
    assert parity["matched_count"] == 1
    assert parity["class_match_rate"] == 1.0
    assert parity["mean_iou"] > 0.95
    assert parity["max_conf_diff"] == pytest.approx(0.001, abs=1e-5)

    assert verify_threshold_monotonicity(
        conf_counts={"0.15": 10, "0.25": 8, "0.50": 5, "0.75": 2},
        iou_counts={"0.30": 6, "0.50": 8, "0.70": 9},
    )
