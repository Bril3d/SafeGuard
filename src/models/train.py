"""Configurable YOLO training runner with MLflow tracking and registry support."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import yaml

from src.data.inspect_dataset import compute_sha256


@dataclass(frozen=True)
class RunSpec:
    """Fully resolved, reproducible training experiment specification."""

    experiment_id: str
    run_name: str
    description: str
    architecture: str
    model_weights: str
    data_yaml: str
    dataset_scope: str
    epochs: int
    batch_size: int
    imgsz: int
    optimizer: str
    lr0: float
    lrf: float
    weight_decay: float
    seed: int
    device: str
    workers: int
    amp: bool
    patience: int
    conf: float
    iou: float
    augmentation_preset: str
    augment_params: dict[str, Any]
    class_map_path: str
    class_map_sha256: str
    canonical_names: list[str]
    mlflow_experiment: str
    applicable_tests: list[str]
    locked_test_sets: dict[str, str]


def load_experiment_config(config_path: Path) -> dict[str, Any]:
    """Load experiment matrix YAML configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_run_spec(
    config_path: Path,
    experiment_id: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    imgsz: int | None = None,
    seed: int | None = None,
    run_name: str | None = None,
) -> RunSpec:
    """Resolve an experiment entry merged with defaults and CLI/DVC overrides."""
    cfg = load_experiment_config(config_path)
    defaults = cfg["defaults"]
    experiments = cfg["experiments"]
    if experiment_id not in experiments:
        raise KeyError(
            f"Unknown experiment '{experiment_id}'. "
            f"Available: {sorted(experiments.keys())}"
        )

    exp = experiments[experiment_id]
    aug_preset = str(exp.get("augmentation_preset", "baseline"))
    aug_params = dict(cfg["augmentations"][aug_preset])

    class_map_path = Path(defaults["class_mapping_path"])
    with class_map_path.open("r", encoding="utf-8") as handle:
        mapping_raw = yaml.safe_load(handle)
    canonical_names = [
        str(c["name"])
        for c in sorted(
            mapping_raw.get("canonical_classes", []), key=lambda x: int(x["id"])
        )
        if c.get("status") == "approved"
    ]

    resolved_seed = int(seed if seed is not None else exp.get("seed", defaults["seed"]))
    return RunSpec(
        experiment_id=experiment_id,
        run_name=run_name or f"{experiment_id}_seed{resolved_seed}",
        description=str(exp["description"]),
        architecture=str(exp["architecture"]),
        model_weights=str(exp["model_weights"]),
        data_yaml=str(exp["data_yaml"]),
        dataset_scope=str(exp["dataset_scope"]),
        epochs=int(epochs if epochs is not None else exp.get("epochs", defaults["epochs"])),
        batch_size=int(
            batch_size
            if batch_size is not None
            else exp.get("batch_size", defaults["batch_size"])
        ),
        imgsz=int(imgsz if imgsz is not None else exp.get("imgsz", defaults["imgsz"])),
        optimizer=str(exp.get("optimizer", defaults["optimizer"])),
        lr0=float(exp.get("lr0", defaults["lr0"])),
        lrf=float(exp.get("lrf", defaults["lrf"])),
        weight_decay=float(exp.get("weight_decay", defaults["weight_decay"])),
        seed=resolved_seed,
        device=str(exp.get("device", defaults["device"])),
        workers=int(exp.get("workers", defaults["workers"])),
        amp=bool(exp.get("amp", defaults["amp"])),
        patience=int(exp.get("patience", defaults["patience"])),
        conf=float(exp.get("conf", defaults["conf"])),
        iou=float(exp.get("iou", defaults["iou"])),
        augmentation_preset=aug_preset,
        augment_params=aug_params,
        class_map_path=str(class_map_path),
        class_map_sha256=compute_sha256(class_map_path),
        canonical_names=canonical_names,
        mlflow_experiment=str(defaults["mlflow_experiment"]),
        applicable_tests=list(exp.get("applicable_tests", [])),
        locked_test_sets=dict(defaults.get("locked_test_sets", {})),
    )


def _git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _set_reproducible_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    spec: RunSpec,
    tracking_uri: str,
    summary_output_path: Path,
    candidate_dir: Path = Path("models/candidates"),
    register_model_name: str | None = None,
    register_stage: str = "Staging",
) -> dict[str, Any]:
    """Execute Ultralytics YOLO training and log full provenance to MLflow."""
    import torch
    from ultralytics import YOLO, settings as ultra_settings

    # Disable Ultralytics built-in MLflow logger so SafeGuard manages runs cleanly.
    ultra_settings.update({"mlflow": False})
    _set_reproducible_seeds(spec.seed)

    use_cuda = torch.cuda.is_available() and spec.device != "cpu"
    device_arg = spec.device if use_cuda else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(0)

    dvc_lock = Path("dvc.lock")
    dvc_lock_sha = compute_sha256(dvc_lock) if dvc_lock.exists() else "untracked"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(spec.mlflow_experiment)

    project_dir = Path("runs/train")
    t_start = time.perf_counter()
    model = YOLO(spec.model_weights)
    train_results = model.train(
        data=spec.data_yaml,
        epochs=spec.epochs,
        batch=spec.batch_size,
        imgsz=spec.imgsz,
        optimizer=spec.optimizer,
        lr0=spec.lr0,
        lrf=spec.lrf,
        weight_decay=spec.weight_decay,
        seed=spec.seed,
        device=device_arg,
        workers=spec.workers,
        amp=spec.amp,
        patience=spec.patience,
        project=str(project_dir),
        name=spec.run_name,
        exist_ok=True,
        verbose=True,
        **spec.augment_params,
    )
    duration_sec = round(time.perf_counter() - t_start, 2)

    peak_alloc_mb = (
        round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        if use_cuda
        else 0.0
    )
    peak_reserved_mb = (
        round(torch.cuda.max_memory_reserved(0) / (1024 * 1024), 2)
        if use_cuda
        else 0.0
    )

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    selected_ckpt = best_pt if best_pt.exists() else last_pt

    exp_cand_dir = candidate_dir / spec.experiment_id
    exp_cand_dir.mkdir(parents=True, exist_ok=True)
    archived_ckpt = exp_cand_dir / f"{spec.run_name}_best.pt"
    if selected_ckpt.exists():
        shutil.copy2(selected_ckpt, archived_ckpt)

    results_dict = getattr(train_results, "results_dict", {}) or {}
    val_metrics = {
        "val_precision": float(results_dict.get("metrics/precision(B)", 0.0)),
        "val_recall": float(results_dict.get("metrics/recall(B)", 0.0)),
        "val_mAP50": float(results_dict.get("metrics/mAP50(B)", 0.0)),
        "val_mAP50_95": float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(results_dict.get("fitness", 0.0)),
        "training_duration_sec": duration_sec,
        "peak_gpu_allocated_mb": peak_alloc_mb,
        "peak_gpu_reserved_mb": peak_reserved_mb,
    }

    with mlflow.start_run(run_name=spec.run_name) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(
            {
                "experiment_id": spec.experiment_id,
                "architecture": spec.architecture,
                "pretrained_weights": spec.model_weights,
                "dataset_scope": spec.dataset_scope,
                "data_yaml": spec.data_yaml,
                "dataset_dvc_lock_sha256": dvc_lock_sha,
                "class_map_sha256": spec.class_map_sha256,
                "canonical_classes": ",".join(spec.canonical_names),
                "epochs": spec.epochs,
                "batch_size": spec.batch_size,
                "imgsz": spec.imgsz,
                "optimizer": spec.optimizer,
                "lr0": spec.lr0,
                "lrf": spec.lrf,
                "weight_decay": spec.weight_decay,
                "seed": spec.seed,
                "amp": spec.amp,
                "augmentation_preset": spec.augmentation_preset,
                "gpu_device": gpu_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "none",
                "git_revision": _git_revision(),
                **{f"aug_{k}": v for k, v in spec.augment_params.items()},
            }
        )
        mlflow.log_metrics(val_metrics)

        artifact_candidates = [
            archived_ckpt,
            save_dir / "results.csv",
            save_dir / "results.png",
            save_dir / "confusion_matrix.png",
            save_dir / "confusion_matrix_normalized.png",
            save_dir / "BoxPR_curve.png",
            save_dir / "BoxF1_curve.png",
        ]
        for art in artifact_candidates:
            if art.exists():
                sub = "weights" if art.suffix == ".pt" else "training_plots"
                mlflow.log_artifact(str(art), artifact_path=sub)

        if register_model_name and archived_ckpt.exists():
            client = MlflowClient()
            try:
                client.create_registered_model(register_model_name)
            except Exception:
                pass
            mv = client.create_model_version(
                name=register_model_name,
                source=f"runs:/{run_id}/weights/{archived_ckpt.name}",
                run_id=run_id,
            )
            client.set_model_version_tag(
                register_model_name, mv.version, "stage", register_stage
            )
            client.set_registered_model_alias(
                register_model_name, register_stage.lower(), mv.version
            )

    summary = {
        "experiment_id": spec.experiment_id,
        "run_name": spec.run_name,
        "mlflow_run_id": run_id,
        "mlflow_tracking_uri": tracking_uri,
        "architecture": spec.architecture,
        "model_weights": spec.model_weights,
        "data_yaml": spec.data_yaml,
        "dataset_scope": spec.dataset_scope,
        "epochs": spec.epochs,
        "batch_size": spec.batch_size,
        "imgsz": spec.imgsz,
        "seed": spec.seed,
        "gpu_device": gpu_name,
        "checkpoint_path": str(archived_ckpt),
        "checkpoint_sha256": (
            compute_sha256(archived_ckpt) if archived_ckpt.exists() else None
        ),
        "save_dir": str(save_dir),
        "metrics": val_metrics,
        "applicable_tests": spec.applicable_tests,
        "locked_test_sets": spec.locked_test_sets,
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    exp_summary = Path("reports/experiments") / spec.experiment_id / "train_summary.json"
    exp_summary.parent.mkdir(parents=True, exist_ok=True)
    exp_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for configurable YOLO training."""
    parser = argparse.ArgumentParser(description="SafeGuard YOLO training runner.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiments/active_train_summary.json"),
    )
    parser.add_argument("--register-name", type=str, default=None)
    parser.add_argument("--register-stage", type=str, default="Staging")
    args = parser.parse_args()

    with args.params.open("r", encoding="utf-8") as handle:
        train_params = yaml.safe_load(handle).get("training", {})

    exp_id = args.experiment or str(train_params.get("active_experiment", "SMOKE"))
    spec = resolve_run_spec(
        config_path=args.config,
        experiment_id=exp_id,
        epochs=args.epochs if args.epochs is not None else train_params.get("epochs_override"),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else train_params.get("batch_size_override")
        ),
        imgsz=args.imgsz if args.imgsz is not None else train_params.get("imgsz_override"),
        seed=args.seed if args.seed is not None else train_params.get("seed_override"),
        run_name=args.run_name,
    )
    tracking_uri = args.tracking_uri or str(
        train_params.get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    )
    summary = run_training(
        spec=spec,
        tracking_uri=tracking_uri,
        summary_output_path=args.output,
        register_model_name=args.register_name,
        register_stage=args.register_stage,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
