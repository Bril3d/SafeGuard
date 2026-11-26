"""Prefect 3 DAG/flow orchestration wrapping DVC stages and drift-triggered CT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from prefect import flow, task

from src.monitoring.drift import (
    build_validation_reference_df,
    evaluate_proxy_drift,
    load_prediction_log_df,
)

DATA_STAGES: tuple[str, ...] = (
    "ingest",
    "validate",
    "harmonize",
    "deduplicate",
    "split",
    "statistics",
)
TRAINING_STAGES: tuple[str, ...] = ("train", "evaluate", "export", "benchmark")


@task(name="run-dvc-stage", retries=1, retry_delay_seconds=1)
def run_dvc_stage_task(stage_name: str, dry_run: bool = True) -> dict[str, Any]:
    """Execute or validate a single DVC pipeline stage."""
    cmd = [sys.executable, "-m", "dvc", "repro", "--single-item"]
    if dry_run:
        cmd.append("--dry")
    cmd.append(stage_name)

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"DVC stage '{stage_name}' failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return {
        "stage": stage_name,
        "dry_run": dry_run,
        "status": "ok",
        "output": proc.stdout.strip(),
    }


@task(name="check-proxy-drift")
def check_proxy_drift_task(
    stats_path: Path = Path("reports/dataset/final_split_statistics.json"),
    pred_log_path: Path = Path("reports/monitoring/predictions.jsonl"),
    output_json: Path = Path("reports/monitoring/drift_report.json"),
) -> dict[str, Any]:
    """Evaluate proxy distribution drift against validation statistics baseline."""
    ref_df = build_validation_reference_df(stats_path=stats_path, sample_size=200)
    cur_df = load_prediction_log_df(pred_log_path)
    if cur_df.empty:
        cur_df = build_validation_reference_df(
            stats_path=stats_path, sample_size=120, seed=101
        )
    summary = evaluate_proxy_drift(
        reference_df=ref_df,
        current_df=cur_df,
        output_json=output_json,
        output_html=None,
    )
    return summary.__dict__


@flow(name="safeguard-data-pipeline")
def safeguard_data_pipeline_flow(dry_run: bool = True) -> dict[str, Any]:
    """Orchestrate ingest -> validate -> harmonize -> deduplicate -> split -> stats."""
    results = [
        run_dvc_stage_task(stage_name=stage, dry_run=dry_run) for stage in DATA_STAGES
    ]
    return {"flow": "safeguard-data-pipeline", "dry_run": dry_run, "stages": results}


@flow(name="safeguard-training-ct-pipeline")
def safeguard_training_ct_flow(
    dry_run: bool = True,
    force_retrain: bool = False,
    drift_override: bool | None = None,
    pred_log_path: Path = Path("reports/monitoring/predictions.jsonl"),
    drift_report_path: Path = Path("reports/monitoring/drift_report.json"),
) -> dict[str, Any]:
    """Continuous Training flow: run drift check and trigger train/export if needed."""
    drift_result = check_proxy_drift_task(
        pred_log_path=pred_log_path,
        output_json=drift_report_path,
    )
    drift_detected = (
        bool(drift_override)
        if drift_override is not None
        else bool(drift_result["drift_detected"])
    )

    if not (drift_detected or force_retrain):
        return {
            "flow": "safeguard-training-ct-pipeline",
            "retraining_triggered": False,
            "reason": "no_proxy_drift_and_force_retrain_false",
            "drift_summary": drift_result,
            "stages": [],
        }

    stages = [
        run_dvc_stage_task(stage_name=stage, dry_run=dry_run)
        for stage in TRAINING_STAGES
    ]
    return {
        "flow": "safeguard-training-ct-pipeline",
        "retraining_triggered": True,
        "dry_run": dry_run,
        "drift_summary": drift_result,
        "stages": stages,
    }


def main() -> None:
    """CLI entrypoint for scheduled or manual Prefect orchestration."""
    parser = argparse.ArgumentParser(description="SafeGuard Prefect MLOps Orchestrator")
    parser.add_argument(
        "--mode",
        choices=("data", "ct", "dry-run"),
        default="dry-run",
    )
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dry = not args.execute
    if args.mode == "data":
        out = safeguard_data_pipeline_flow(dry_run=dry)
    elif args.mode == "ct":
        out = safeguard_training_ct_flow(dry_run=dry, force_retrain=args.force_retrain)
    else:
        data_out = safeguard_data_pipeline_flow(dry_run=True)
        ct_out = safeguard_training_ct_flow(dry_run=True, force_retrain=True)
        out = {"data": data_out, "ct": ct_out}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
