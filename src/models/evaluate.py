"""Evaluate trained YOLO checkpoints on locked source-specific test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import yaml


def _compute_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0.0 else 0.0


def evaluate_checkpoint(
    weights_path: Path,
    test_sets: dict[str, str],
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.50,
    device: str = "0",
    tracking_uri: str | None = None,
    mlflow_run_id: str | None = None,
    output_path: Path = Path("reports/experiments/active_eval_summary.json"),
) -> dict[str, Any]:
    """Evaluate a checkpoint on each locked source-specific test YAML."""
    import torch
    from ultralytics import YOLO, settings as ultra_settings

    ultra_settings.update({"mlflow": False})
    use_cuda = torch.cuda.is_available() and device != "cpu"
    device_arg = device if use_cuda else "cpu"

    model = YOLO(str(weights_path))
    test_results: dict[str, Any] = {}

    for test_name, test_yaml in test_sets.items():
        val_out = model.val(
            data=test_yaml,
            split="test",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device_arg,
            project="runs/eval",
            name=f"{weights_path.stem}_{test_name}",
            exist_ok=True,
            verbose=False,
        )
        mp = float(val_out.box.mp)
        mr = float(val_out.box.mr)
        map50 = float(val_out.box.map50)
        map50_95 = float(val_out.box.map)
        f1 = _compute_f1(mp, mr)

        per_class: dict[str, dict[str, float]] = {}
        names = val_out.names
        ap_class_index = list(getattr(val_out.box, "ap_class_index", []))
        for idx_pos, cls_id in enumerate(ap_class_index):
            cls_name = str(names[int(cls_id)])
            c_p = float(val_out.box.p[idx_pos])
            c_r = float(val_out.box.r[idx_pos])
            c_ap50 = float(val_out.box.ap50[idx_pos])
            c_ap = float(val_out.box.ap[idx_pos])
            per_class[cls_name] = {
                "precision": round(c_p, 4),
                "recall": round(c_r, 4),
                "f1": round(_compute_f1(c_p, c_r), 4),
                "mAP50": round(c_ap50, 4),
                "mAP50_95": round(c_ap, 4),
            }

        test_results[test_name] = {
            "test_yaml": test_yaml,
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "per_class": per_class,
            "save_dir": str(val_out.save_dir),
        }

    if tracking_uri and mlflow_run_id:
        mlflow.set_tracking_uri(tracking_uri)
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_params(
                {"eval_conf_threshold": conf, "eval_iou_threshold": iou}
            )
            for test_name, metrics in test_results.items():
                mlflow.log_metrics(
                    {
                        f"{test_name}_precision": metrics["precision"],
                        f"{test_name}_recall": metrics["recall"],
                        f"{test_name}_f1": metrics["f1"],
                        f"{test_name}_mAP50": metrics["mAP50"],
                        f"{test_name}_mAP50_95": metrics["mAP50_95"],
                    }
                )
                for cls_name, cls_m in metrics["per_class"].items():
                    mlflow.log_metric(
                        f"{test_name}_{cls_name}_AP50", cls_m["mAP50"]
                    )
                cm_path = Path(metrics["save_dir"]) / "confusion_matrix.png"
                if cm_path.exists():
                    mlflow.log_artifact(
                        str(cm_path), artifact_path=f"eval_{test_name}"
                    )

    payload = {
        "weights_path": str(weights_path),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "test_results": test_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    """CLI entrypoint for evaluating a trained checkpoint."""
    parser = argparse.ArgumentParser(description="SafeGuard locked test evaluation.")
    parser.add_argument(
        "--train-summary",
        type=Path,
        default=Path("reports/experiments/active_train_summary.json"),
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/active_eval_summary.json"),
    )
    args = parser.parse_args()

    summary = json.loads(args.train_summary.read_text(encoding="utf-8"))
    with args.params.open("r", encoding="utf-8") as handle:
        train_params = yaml.safe_load(handle).get("training", {})

    applicable = summary.get("applicable_tests", [])
    locked_sets = summary.get("locked_test_sets", {})
    selected_tests = {k: locked_sets[k] for k in applicable if k in locked_sets}

    eval_out = evaluate_checkpoint(
        weights_path=Path(summary["checkpoint_path"]),
        test_sets=selected_tests,
        imgsz=int(summary["imgsz"]),
        conf=float(train_params.get("conf_threshold", 0.25)),
        iou=float(train_params.get("iou_threshold", 0.50)),
        tracking_uri=summary.get("mlflow_tracking_uri"),
        mlflow_run_id=summary.get("mlflow_run_id"),
        output_path=args.output,
    )
    print(json.dumps(eval_out, indent=2))


if __name__ == "__main__":
    main()
