"""ONNX export, parity verification, threshold checks, and MLflow registration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import yaml

from backend.app.inference import (
    DEFAULT_CANONICAL_NAMES,
    ONNXDetector,
    preprocess_image,
)
from src.data.inspect_dataset import IMAGE_EXTENSIONS, compute_sha256
from src.models.error_analysis import compute_iou_xyxy


def load_canonical_classes(
    class_mapping_path: Path = Path("configs/class_mapping.yaml"),
) -> dict[int, str]:
    """Read approved canonical class IDs and names from class_mapping.yaml."""
    if not class_mapping_path.exists():
        return dict(DEFAULT_CANONICAL_NAMES)
    raw = yaml.safe_load(class_mapping_path.read_text(encoding="utf-8")) or {}
    approved = [
        c for c in raw.get("canonical_classes", []) if c.get("status") == "approved"
    ]
    approved.sort(key=lambda item: int(item["id"]))
    return {int(item["id"]): str(item["name"]) for item in approved}


def select_best_experiment_from_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select best non-SMOKE validated experiment by test then validation mAP@50."""
    eligible = [r for r in records if str(r.get("experiment_id", "")) != "SMOKE"]
    if not eligible:
        raise ValueError("No non-SMOKE experimental runs available to select from.")

    def _score_key(rec: dict[str, Any]) -> tuple[float, float, float]:
        test_scores = [
            float(rec[k])
            for k in ("ultralytics_test_mAP50", "roboflow_test_mAP50")
            if rec.get(k) is not None
        ]
        mean_test = sum(test_scores) / len(test_scores) if test_scores else -1.0
        has_both = 1.0 if len(test_scores) == 2 else 0.0
        val_map = float(rec.get("val_mAP50") or 0.0)
        return (has_both, mean_test, val_map)

    return max(eligible, key=_score_key)


def resolve_best_mlflow_checkpoint(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "SafeGuard-PPE-Detection",
    reports_root: Path = Path("reports/experiments"),
) -> dict[str, Any]:
    """Query MLflow for the best validated experiment and return checkpoint info."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs([experiment.experiment_id], max_results=200)
    records: list[dict[str, Any]] = []
    for run in runs:
        exp_id = str(run.data.params.get("experiment_id", ""))
        records.append(
            {
                "experiment_id": exp_id,
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "architecture": run.data.params.get("architecture"),
                "imgsz": int(run.data.params.get("imgsz", 640)),
                "val_mAP50": run.data.metrics.get("val_mAP50"),
                "ultralytics_test_mAP50": run.data.metrics.get(
                    "ultralytics_test_mAP50"
                ),
                "roboflow_test_mAP50": run.data.metrics.get("roboflow_test_mAP50"),
            }
        )

    best = select_best_experiment_from_records(records)
    exp_id = best["experiment_id"]
    train_summary_path = reports_root / exp_id / "train_summary.json"
    if not train_summary_path.exists():
        raise FileNotFoundError(
            f"Expected train summary missing for {exp_id}: {train_summary_path}"
        )
    summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    weights_path = Path(summary["checkpoint_path"])
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")

    return {
        "experiment_id": exp_id,
        "run_name": summary["run_name"],
        "mlflow_run_id": best["run_id"],
        "architecture": summary.get("architecture", best.get("architecture")),
        "imgsz": int(summary.get("imgsz", best.get("imgsz", 640))),
        "checkpoint_path": str(weights_path),
        "checkpoint_sha256": compute_sha256(weights_path),
        "val_mAP50": best.get("val_mAP50"),
        "ultralytics_test_mAP50": best.get("ultralytics_test_mAP50"),
        "roboflow_test_mAP50": best.get("roboflow_test_mAP50"),
    }


def export_to_onnx(
    weights_path: Path,
    output_onnx_path: Path,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
    class_mapping_path: Path = Path("configs/class_mapping.yaml"),
    model_version: str | None = None,
    mlflow_run_id: str | None = None,
    architecture: str | None = None,
    default_conf: float = 0.25,
    default_iou: float = 0.50,
) -> dict[str, Any]:
    """Export trained YOLO checkpoint to ONNX, verify graph, and save metadata."""
    import onnx
    from ultralytics import YOLO

    canonical_names = load_canonical_classes(class_mapping_path)
    model = YOLO(str(weights_path))
    pt_names = {int(k): str(v) for k, v in model.names.items()}
    if pt_names != canonical_names:
        raise ValueError(
            f"Class mapping mismatch: PyTorch {pt_names} != Canonical {canonical_names}"
        )

    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=False,
    )
    exported_path = Path(exported_path_str)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_onnx_path.resolve():
        output_onnx_path.write_bytes(exported_path.read_bytes())
        exported_path.unlink(missing_ok=True)

    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)
    graph_opsets = [
        int(imp.version) for imp in onnx_model.opset_import if imp.domain == ""
    ]

    detector = ONNXDetector(
        onnx_path=output_onnx_path,
        providers=["CPUExecutionProvider"],
        default_conf=default_conf,
        default_iou=default_iou,
    )
    expected_in = [1, 3, imgsz, imgsz]
    expected_out = [1, 4 + len(canonical_names), detector.output_shape[-1]]
    if detector.input_shape != expected_in:
        raise ValueError(
            f"Unexpected ONNX input shape {detector.input_shape}, expected {expected_in}"
        )
    if len(detector.output_shape) != 3 or detector.output_shape[1] != expected_out[1]:
        raise ValueError(
            f"Unexpected ONNX output shape {detector.output_shape}, expected [1, {expected_out[1]}, N]"
        )

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    ort_out = detector.run_raw_tensor(dummy)

    inputs_meta = [
        {"name": inp.name, "shape": list(inp.shape), "type": inp.type}
        for inp in detector.session.get_inputs()
    ]
    outputs_meta = [
        {"name": out.name, "shape": list(out.shape), "type": out.type}
        for out in detector.session.get_outputs()
    ]

    metadata = {
        "model_version": model_version or f"{weights_path.stem}-opset{opset}",
        "architecture": architecture or "YOLO11s",
        "mlflow_run_id": mlflow_run_id,
        "source_checkpoint": str(weights_path),
        "checkpoint_sha256": compute_sha256(weights_path),
        "checkpoint_size_mb": round(weights_path.stat().st_size / (1024 * 1024), 3),
        "onnx_path": str(output_onnx_path),
        "onnx_sha256": compute_sha256(output_onnx_path),
        "onnx_size_mb": round(output_onnx_path.stat().st_size / (1024 * 1024), 3),
        "opset": opset,
        "graph_opsets": graph_opsets,
        "imgsz": imgsz,
        "dynamic": dynamic,
        "default_conf": default_conf,
        "default_iou": default_iou,
        "class_names": {str(k): v for k, v in canonical_names.items()},
        "inputs": inputs_meta,
        "outputs": outputs_meta,
        "smoke_output_shape": list(ort_out.shape),
        "cold_load_time_ms": detector.load_time_ms,
    }
    meta_path = output_onnx_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def compare_detection_sets(
    pt_dets: list[dict[str, Any]],
    onnx_dets: list[dict[str, Any]],
    iou_match_threshold: float = 0.5,
) -> dict[str, Any]:
    """Match PyTorch and ONNX detections greedily by IoU and compute parity stats."""
    used_onnx: set[int] = set()
    ious: list[float] = []
    class_matches = 0
    conf_diffs: list[float] = []
    box_coord_diffs: list[float] = []

    for pt in pt_dets:
        best_idx = -1
        best_iou = 0.0
        for idx, ox in enumerate(onnx_dets):
            if idx in used_onnx:
                continue
            iou = compute_iou_xyxy(pt["bbox_xyxy"], ox["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_match_threshold:
            used_onnx.add(best_idx)
            ox = onnx_dets[best_idx]
            ious.append(best_iou)
            if int(pt["class_id"]) == int(ox["class_id"]):
                class_matches += 1
            conf_diffs.append(abs(float(pt["confidence"]) - float(ox["confidence"])))
            coord_diff = max(
                abs(float(a) - float(b))
                for a, b in zip(pt["bbox_xyxy"], ox["bbox_xyxy"])
            )
            box_coord_diffs.append(coord_diff)

    matched = len(ious)
    return {
        "pt_count": len(pt_dets),
        "onnx_count": len(onnx_dets),
        "matched_count": matched,
        "class_match_rate": round(class_matches / matched if matched else 1.0, 6),
        "mean_iou": round(float(np.mean(ious)) if ious else 1.0, 6),
        "min_iou": round(float(np.min(ious)) if ious else 1.0, 6),
        "mean_conf_diff": round(float(np.mean(conf_diffs)) if conf_diffs else 0.0, 6),
        "max_conf_diff": round(float(np.max(conf_diffs)) if conf_diffs else 0.0, 6),
        "max_box_coord_diff_px": round(
            float(np.max(box_coord_diffs)) if box_coord_diffs else 0.0, 3
        ),
    }


def verify_threshold_monotonicity(
    conf_counts: dict[str, int],
    iou_counts: dict[str, int],
) -> bool:
    """Verify higher conf shrinks predictions and higher NMS IoU retains >= boxes."""
    ordered_conf = [conf_counts[k] for k in sorted(conf_counts.keys(), key=float)]
    conf_ok = all(
        ordered_conf[i] >= ordered_conf[i + 1] for i in range(len(ordered_conf) - 1)
    )

    ordered_iou = [iou_counts[k] for k in sorted(iou_counts.keys(), key=float)]
    iou_ok = all(
        ordered_iou[i] <= ordered_iou[i + 1] for i in range(len(ordered_iou) - 1)
    )
    return conf_ok and iou_ok


def collect_validation_images(max_images: int = 30) -> list[Path]:
    """Collect balanced deterministic validation images (never locked test data)."""
    rf_val = Path("data/processed/val_roboflow/images")
    ul_val = Path("data/processed/val_ultralytics/images")
    half = max_images // 2
    rf_imgs = sorted(
        p
        for p in rf_val.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[:half]
    ul_imgs = sorted(
        p
        for p in ul_val.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )[: max_images - len(rf_imgs)]
    return rf_imgs + ul_imgs


def _run_pytorch_predict(
    pt_model: Any,
    img_path: Path,
    imgsz: int,
    conf: float,
    iou: float,
) -> list[dict[str, Any]]:
    res = pt_model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )[0]
    dets: list[dict[str, Any]] = []
    names = {int(k): str(v) for k, v in res.names.items()}
    if res.boxes is not None and len(res.boxes) > 0:
        for box in res.boxes:
            cid = int(box.cls.item())
            dets.append(
                {
                    "class_id": cid,
                    "class_name": names.get(cid, str(cid)),
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )
    return dets


def verify_pytorch_onnx_parity(
    weights_path: Path,
    onnx_path: Path,
    image_paths: list[Path],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.50,
    class_mapping_path: Path = Path("configs/class_mapping.yaml"),
    output_path: Path = Path("reports/benchmark/verification_report.json"),
) -> dict[str, Any]:
    """Verify tensor, class-mapping, bbox, and threshold parity between PyTorch and ONNX."""
    import torch
    from ultralytics import YOLO

    canonical_names = load_canonical_classes(class_mapping_path)
    pt_model = YOLO(str(weights_path))
    pt_model.to("cpu")
    detector = ONNXDetector(
        onnx_path=onnx_path,
        providers=["CPUExecutionProvider"],
        default_conf=conf,
        default_iou=iou,
    )

    pt_names = {int(k): str(v) for k, v in pt_model.names.items()}
    class_mapping_consistent = (
        pt_names == canonical_names and detector.class_names == canonical_names
    )

    # 1. Raw forward-pass & same-tensor NMS parity on identical preprocessed inputs.
    from ultralytics.utils.nms import non_max_suppression

    raw_max_abs = 0.0
    raw_mean_abs_vals: list[float] = []
    raw_cos_vals: list[float] = []
    same_tensor_pt_cnt = 0
    same_tensor_ox_cnt = 0
    same_tensor_matched = 0
    same_tensor_ious: list[float] = []
    same_tensor_conf_diffs: list[float] = []
    same_tensor_coord_diffs: list[float] = []
    pt_inner = pt_model.model.eval().to("cpu")

    for img_path in image_paths[: min(10, len(image_paths))]:
        with Image.open(img_path) as im:
            tensor, meta = preprocess_image(im, target_size=imgsz)
        with torch.inference_mode():
            pt_out = pt_inner(torch.from_numpy(tensor))
            pt_raw_t = pt_out[0] if isinstance(pt_out, (tuple, list)) else pt_out
            pt_raw = pt_raw_t.cpu().numpy()
            pt_nms = non_max_suppression(pt_raw_t, conf_thres=conf, iou_thres=iou)
        onnx_raw = detector.run_raw_tensor(tensor)

        from backend.app.inference import postprocess_yolo_output

        ox_same = postprocess_yolo_output(
            onnx_raw, meta, detector.class_names, conf_threshold=conf, iou_threshold=iou
        )
        pt_same_list: list[dict[str, Any]] = []
        if pt_nms and pt_nms[0] is not None and len(pt_nms[0]) > 0:
            for row in pt_nms[0].cpu().numpy():
                x1, y1, x2, y2, score, cls_idx = row[:6]
                ux1 = float(
                    np.clip((x1 - meta.pad_left) / meta.scale, 0.0, meta.orig_width)
                )
                uy1 = float(
                    np.clip((y1 - meta.pad_top) / meta.scale, 0.0, meta.orig_height)
                )
                ux2 = float(
                    np.clip((x2 - meta.pad_left) / meta.scale, 0.0, meta.orig_width)
                )
                uy2 = float(
                    np.clip((y2 - meta.pad_top) / meta.scale, 0.0, meta.orig_height)
                )
                pt_same_list.append(
                    {
                        "class_id": int(cls_idx),
                        "class_name": detector.class_names.get(
                            int(cls_idx), str(int(cls_idx))
                        ),
                        "confidence": float(score),
                        "bbox_xyxy": [ux1, uy1, ux2, uy2],
                    }
                )
        ox_same_list = [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
            }
            for d in ox_same
        ]
        same_tensor_pt_cnt += len(pt_same_list)
        same_tensor_ox_cnt += len(ox_same_list)
        st_pair = compare_detection_sets(
            pt_same_list, ox_same_list, iou_match_threshold=0.5
        )
        same_tensor_matched += st_pair["matched_count"]
        if st_pair["matched_count"] > 0:
            same_tensor_ious.append(st_pair["mean_iou"])
            same_tensor_conf_diffs.append(st_pair["max_conf_diff"])
            same_tensor_coord_diffs.append(st_pair["max_box_coord_diff_px"])

        diff = np.abs(pt_raw - onnx_raw)
        raw_max_abs = max(raw_max_abs, float(np.max(diff)))
        raw_mean_abs_vals.append(float(np.mean(diff)))
        pt_flat = pt_raw.reshape(-1).astype(np.float64)
        ox_flat = onnx_raw.reshape(-1).astype(np.float64)
        denom = float(np.linalg.norm(pt_flat) * np.linalg.norm(ox_flat))
        cos = float(np.dot(pt_flat, ox_flat) / denom) if denom > 0 else 1.0
        raw_cos_vals.append(cos)

    # 2. End-to-end prediction & coordinate bounds parity.
    total_pt = 0
    total_onnx = 0
    total_matched = 0
    exact_count_images = 0
    all_ious: list[float] = []
    all_conf_diffs: list[float] = []
    all_coord_diffs: list[float] = []
    bounds_violations = 0
    class_mismatches = 0

    for img_path in image_paths:
        pt_dets = _run_pytorch_predict(
            pt_model, img_path, imgsz=imgsz, conf=conf, iou=iou
        )
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size
            onnx_out = detector.predict_image(
                im, conf_threshold=conf, iou_threshold=iou
            )

        ox_dets = [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "bbox_xyxy": d.bbox_xyxy,
            }
            for d in onnx_out.detections
        ]
        for od in ox_dets:
            x1, y1, x2, y2 = od["bbox_xyxy"]
            if not (
                0.0 <= x1 < x2 <= orig_w + 1e-3 and 0.0 <= y1 < y2 <= orig_h + 1e-3
            ):
                bounds_violations += 1
            if detector.class_names.get(od["class_id"]) != od["class_name"]:
                class_mismatches += 1

        total_pt += len(pt_dets)
        total_onnx += len(ox_dets)
        if len(pt_dets) == len(ox_dets):
            exact_count_images += 1

        pair_stats = compare_detection_sets(pt_dets, ox_dets, iou_match_threshold=0.5)
        total_matched += pair_stats["matched_count"]
        if pair_stats["matched_count"] > 0:
            all_ious.append(pair_stats["mean_iou"])
            all_conf_diffs.append(pair_stats["max_conf_diff"])
            all_coord_diffs.append(pair_stats["max_box_coord_diff_px"])
            if pair_stats["class_match_rate"] < 1.0:
                class_mismatches += 1

    # 3. Confidence and IoU threshold sweeps.
    sweep_imgs = image_paths[: min(10, len(image_paths))]
    pt_conf_counts: dict[str, int] = {}
    onnx_conf_counts: dict[str, int] = {}
    for c_val in (0.15, 0.25, 0.50, 0.75):
        key = f"{c_val:.2f}"
        pt_conf_counts[key] = sum(
            len(_run_pytorch_predict(pt_model, p, imgsz=imgsz, conf=c_val, iou=iou))
            for p in sweep_imgs
        )
        ox_cnt = 0
        for p in sweep_imgs:
            with Image.open(p) as im:
                out = detector.predict_image(
                    im, conf_threshold=c_val, iou_threshold=iou
                )
                assert all(d.confidence >= c_val - 1e-6 for d in out.detections)
                ox_cnt += len(out.detections)
        onnx_conf_counts[key] = ox_cnt

    pt_iou_counts: dict[str, int] = {}
    onnx_iou_counts: dict[str, int] = {}
    for i_val in (0.30, 0.50, 0.70):
        key = f"{i_val:.2f}"
        pt_iou_counts[key] = sum(
            len(_run_pytorch_predict(pt_model, p, imgsz=imgsz, conf=conf, iou=i_val))
            for p in sweep_imgs
        )
        ox_cnt = 0
        for p in sweep_imgs:
            with Image.open(p) as im:
                out = detector.predict_image(
                    im, conf_threshold=conf, iou_threshold=i_val
                )
                ox_cnt += len(out.detections)
        onnx_iou_counts[key] = ox_cnt

    report = {
        "weights_path": str(weights_path),
        "onnx_path": str(onnx_path),
        "images_verified": len(image_paths),
        "imgsz": imgsz,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "class_mapping_verification": {
            "consistent": class_mapping_consistent,
            "canonical_classes": {str(k): v for k, v in canonical_names.items()},
            "pytorch_classes": {str(k): v for k, v in pt_names.items()},
            "onnx_classes": {str(k): v for k, v in detector.class_names.items()},
            "class_mismatches": class_mismatches,
        },
        "tensor_shapes": {
            "input_shape": detector.input_shape,
            "output_shape": detector.output_shape,
        },
        "raw_tensor_parity": {
            "images_checked": min(10, len(image_paths)),
            "max_abs_diff": round(raw_max_abs, 7),
            "mean_abs_diff": round(float(np.mean(raw_mean_abs_vals)), 7),
            "min_cosine_similarity": round(float(min(raw_cos_vals)), 8),
        },
        "same_tensor_prediction_parity": {
            "images_checked": min(10, len(image_paths)),
            "total_pytorch_detections": same_tensor_pt_cnt,
            "total_onnx_detections": same_tensor_ox_cnt,
            "matched_detections": same_tensor_matched,
            "match_rate_vs_pytorch": round(
                same_tensor_matched / same_tensor_pt_cnt if same_tensor_pt_cnt else 1.0,
                4,
            ),
            "mean_matched_iou": round(
                float(np.mean(same_tensor_ious)) if same_tensor_ious else 1.0, 5
            ),
            "max_conf_diff": round(
                float(max(same_tensor_conf_diffs)) if same_tensor_conf_diffs else 0.0, 6
            ),
            "max_box_coord_diff_px": round(
                float(max(same_tensor_coord_diffs)) if same_tensor_coord_diffs else 0.0,
                3,
            ),
        },
        "prediction_parity": {
            "total_pytorch_detections": total_pt,
            "total_onnx_detections": total_onnx,
            "matched_detections": total_matched,
            "match_rate_vs_pytorch": round(
                total_matched / total_pt if total_pt else 1.0, 4
            ),
            "exact_count_images": exact_count_images,
            "mean_matched_iou": round(float(np.mean(all_ious)) if all_ious else 1.0, 4),
            "max_conf_diff": round(
                float(max(all_conf_diffs)) if all_conf_diffs else 0.0, 5
            ),
            "max_box_coord_diff_px": round(
                float(max(all_coord_diffs)) if all_coord_diffs else 0.0, 3
            ),
            "bounds_violations": bounds_violations,
        },
        "threshold_behavior": {
            "pytorch_conf_sweep": pt_conf_counts,
            "onnx_conf_sweep": onnx_conf_counts,
            "pytorch_iou_sweep": pt_iou_counts,
            "onnx_iou_sweep": onnx_iou_counts,
            "pytorch_monotonic": verify_threshold_monotonicity(
                pt_conf_counts, pt_iou_counts
            ),
            "onnx_monotonic": verify_threshold_monotonicity(
                onnx_conf_counts, onnx_iou_counts
            ),
        },
    }

    assert math.isfinite(report["raw_tensor_parity"]["max_abs_diff"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def log_optimization_to_mlflow(
    tracking_uri: str,
    mlflow_run_id: str | None,
    export_meta: dict[str, Any],
    verification_report: dict[str, Any],
    verification_report_path: Path,
    register_model_name: str = "SafeGuard-PPE-Detector",
) -> str | None:
    """Log ONNX artifact, verification metrics, and register verified model in MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    raw_parity = verification_report["raw_tensor_parity"]
    pred_parity = verification_report["prediction_parity"]

    ctx = (
        mlflow.start_run(run_id=mlflow_run_id)
        if mlflow_run_id
        else mlflow.start_run(run_name="ONNX-Export-Verify")
    )
    with ctx as active_run:
        run_id = active_run.info.run_id
        existing_params = dict(active_run.data.params)
        candidate_params = {
            "onnx_opset": str(export_meta["opset"]),
            "onnx_imgsz": str(export_meta["imgsz"]),
            "onnx_input_shape": json.dumps(export_meta["inputs"][0]["shape"]),
            "onnx_output_shape": json.dumps(export_meta["outputs"][0]["shape"]),
        }
        new_params = {
            k: v for k, v in candidate_params.items() if k not in existing_params
        }
        if new_params:
            mlflow.log_params(new_params)
        mlflow.set_tag("onnx_sha256", export_meta["onnx_sha256"])
        same_tensor = verification_report.get("same_tensor_prediction_parity", {})
        mlflow.log_metrics(
            {
                "pytorch_checkpoint_size_mb": export_meta["checkpoint_size_mb"],
                "onnx_size_mb": export_meta["onnx_size_mb"],
                "onnx_cold_load_ms": export_meta["cold_load_time_ms"],
                "onnx_raw_max_abs_diff": raw_parity["max_abs_diff"],
                "onnx_raw_min_cosine": raw_parity["min_cosine_similarity"],
                "onnx_same_tensor_match_rate": same_tensor.get(
                    "match_rate_vs_pytorch", 1.0
                ),
                "onnx_same_tensor_mean_iou": same_tensor.get("mean_matched_iou", 1.0),
                "onnx_pred_match_rate": pred_parity["match_rate_vs_pytorch"],
                "onnx_pred_mean_iou": pred_parity["mean_matched_iou"],
                "onnx_pred_max_conf_diff": pred_parity["max_conf_diff"],
                "onnx_pred_max_box_diff_px": pred_parity["max_box_coord_diff_px"],
            }
        )
        mlflow.log_artifact(export_meta["onnx_path"], artifact_path="onnx_model")
        meta_file = Path(export_meta["onnx_path"]).with_suffix(".metadata.json")
        if meta_file.exists():
            mlflow.log_artifact(str(meta_file), artifact_path="onnx_model")
        if verification_report_path.exists():
            mlflow.log_artifact(
                str(verification_report_path), artifact_path="benchmark"
            )

    if register_model_name:
        client = MlflowClient()
        try:
            client.create_registered_model(register_model_name)
        except Exception:
            pass
        onnx_name = Path(export_meta["onnx_path"]).name
        mv = client.create_model_version(
            name=register_model_name,
            source=f"runs:/{run_id}/onnx_model/{onnx_name}",
            run_id=run_id,
        )
        client.set_model_version_tag(
            register_model_name, mv.version, "onnx_verified", "true"
        )
        client.set_model_version_tag(
            register_model_name,
            mv.version,
            "experiment_id",
            export_meta.get("model_version", ""),
        )
        client.set_registered_model_alias(register_model_name, "production", mv.version)
        return str(mv.version)
    return None


def main() -> None:
    """CLI entrypoint: resolve best MLflow checkpoint, export ONNX, verify parity."""
    parser = argparse.ArgumentParser(
        description="Export and verify SafeGuard ONNX model."
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/production/safeguard_yolo11s.onnx"),
    )
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", type=str, default="SafeGuard-PPE-Detection")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-val-images", type=int, default=30)
    parser.add_argument(
        "--verification-output",
        type=Path,
        default=Path("reports/benchmark/verification_report.json"),
    )
    parser.add_argument("--register-name", type=str, default="SafeGuard-PPE-Detector")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    mlflow_run_id: str | None = None
    architecture = "YOLO11s"
    model_version: str | None = None
    if args.weights is None:
        resolved = resolve_best_mlflow_checkpoint(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
        weights_path = Path(resolved["checkpoint_path"])
        mlflow_run_id = resolved["mlflow_run_id"]
        architecture = str(resolved.get("architecture") or architecture)
        model_version = f"{resolved['experiment_id']}-{architecture.lower()}-onnx"
        imgsz = int(resolved.get("imgsz", args.imgsz))
    else:
        weights_path = args.weights
        imgsz = args.imgsz

    export_meta = export_to_onnx(
        weights_path=weights_path,
        output_onnx_path=args.output,
        imgsz=imgsz,
        opset=args.opset,
        model_version=model_version,
        mlflow_run_id=mlflow_run_id,
        architecture=architecture,
        default_conf=args.conf,
        default_iou=args.iou,
    )
    val_images = collect_validation_images(max_images=args.max_val_images)
    verification = verify_pytorch_onnx_parity(
        weights_path=weights_path,
        onnx_path=args.output,
        image_paths=val_images,
        imgsz=imgsz,
        conf=args.conf,
        iou=args.iou,
        output_path=args.verification_output,
    )

    registered_version = None
    if not args.skip_mlflow:
        registered_version = log_optimization_to_mlflow(
            tracking_uri=args.tracking_uri,
            mlflow_run_id=mlflow_run_id,
            export_meta=export_meta,
            verification_report=verification,
            verification_report_path=args.verification_output,
            register_model_name=args.register_name,
        )

    print(
        json.dumps(
            {
                "export_metadata": export_meta,
                "verification": verification,
                "registered_model_version": registered_version,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
