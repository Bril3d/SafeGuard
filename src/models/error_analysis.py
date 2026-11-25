"""Error analysis utilities: TP/FP/FN, confusion pairs, and small-object misses."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from PIL import Image

from src.data.inspect_dataset import parse_yolo_label_file


def compute_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0.0 else 0.0


def analyze_image_errors(
    gt_boxes: list[dict[str, Any]],
    pred_boxes: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    small_area_threshold: float = 0.01,
) -> dict[str, Any]:
    """Match predictions against ground truth and classify TP, FP, FN, and misses."""
    sorted_preds = sorted(
        pred_boxes, key=lambda p: float(p.get("confidence", 0.0)), reverse=True
    )
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    per_class_fp: dict[int, int] = defaultdict(int)
    per_class_fn: dict[int, int] = defaultdict(int)

    for pred in sorted_preds:
        pred_cls = int(pred["class_id"])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt or int(gt["class_id"]) != pred_cls:
                continue
            iou = compute_iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            matched_gt.add(best_idx)
            tp += 1
        else:
            fp += 1
            per_class_fp[pred_cls] += 1

    fn = 0
    small_object_fn = 0
    for idx, gt in enumerate(gt_boxes):
        if idx not in matched_gt:
            fn += 1
            per_class_fn[int(gt["class_id"])] += 1
            if float(gt.get("area_norm", 1.0)) < small_area_threshold:
                small_object_fn += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "small_object_fn": small_object_fn,
        "per_class_fp": dict(per_class_fp),
        "per_class_fn": dict(per_class_fn),
    }


def run_error_analysis(
    weights_path: Path,
    images_dir: Path,
    labels_dir: Path,
    output_path: Path,
    conf: float = 0.25,
    iou: float = 0.50,
    imgsz: int = 640,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run YOLO inference on a split and summarize TP/FP/FN and small-object misses."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        image_paths = image_paths[:limit]

    totals = {"tp": 0, "fp": 0, "fn": 0, "small_object_fn": 0}
    fp_by_cls: dict[int, int] = defaultdict(int)
    fn_by_cls: dict[int, int] = defaultdict(int)
    worst_images: list[dict[str, Any]] = []

    for img_path in image_paths:
        with Image.open(img_path) as img:
            img_w, img_h = img.size

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_raw = parse_yolo_label_file(lbl_path if lbl_path.exists() else None)
        gt_boxes = [
            {
                "class_id": cls_id,
                "bbox_xyxy": [
                    (cx - w / 2.0) * img_w,
                    (cy - h / 2.0) * img_h,
                    (cx + w / 2.0) * img_w,
                    (cy + h / 2.0) * img_h,
                ],
                "area_norm": w * h,
            }
            for cls_id, cx, cy, w, h in gt_raw
        ]

        res = model.predict(
            source=str(img_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False
        )[0]
        pred_boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().tolist()
            confs = res.boxes.conf.cpu().tolist()
            clss = res.boxes.cls.cpu().tolist()
            for box, score, cid in zip(xyxy, confs, clss):
                pred_boxes.append(
                    {
                        "class_id": int(cid),
                        "confidence": float(score),
                        "bbox_xyxy": [float(v) for v in box],
                    }
                )

        img_err = analyze_image_errors(gt_boxes, pred_boxes, iou_threshold=iou)
        for k in totals:
            totals[k] += img_err[k]
        for cid, cnt in img_err["per_class_fp"].items():
            fp_by_cls[cid] += cnt
        for cid, cnt in img_err["per_class_fn"].items():
            fn_by_cls[cid] += cnt
        err_score = img_err["fp"] + img_err["fn"]
        if err_score > 0:
            worst_images.append(
                {
                    "image": img_path.name,
                    "tp": img_err["tp"],
                    "fp": img_err["fp"],
                    "fn": img_err["fn"],
                    "small_object_fn": img_err["small_object_fn"],
                }
            )

    worst_images.sort(key=lambda r: r["fp"] + r["fn"], reverse=True)
    report = {
        "weights_path": str(weights_path),
        "images_analyzed": len(image_paths),
        "conf_threshold": conf,
        "iou_threshold": iou,
        "totals": totals,
        "per_class_fp": {str(k): v for k, v in sorted(fp_by_cls.items())},
        "per_class_fn": {str(k): v for k, v in sorted(fn_by_cls.items())},
        "worst_images_top10": worst_images[:10],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for error analysis."""
    parser = argparse.ArgumentParser(description="SafeGuard YOLO error analysis.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiments/error_analysis.json")
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    run_error_analysis(
        args.weights,
        args.images_dir,
        args.labels_dir,
        args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
