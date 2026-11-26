"""TDD unit tests for Prefect workflow orchestration and Evidently drift checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.monitoring.drift import evaluate_proxy_drift
from src.orchestration.prefect_flows import (
    DATA_STAGES,
    TRAINING_STAGES,
    safeguard_data_pipeline_flow,
    safeguard_training_ct_flow,
)


def test_evidently_proxy_drift_detects_severe_confidence_and_class_shift(
    tmp_path: Path,
) -> None:
    ref_df = pd.DataFrame(
        {
            "class_name": ["helmet", "person", "vest", "helmet", "person"] * 10,
            "confidence": [0.85, 0.88, 0.82, 0.86, 0.84] * 10,
            "bbox_area_norm": [0.003, 0.035, 0.008, 0.0025, 0.03] * 10,
        }
    )
    shifted_df = pd.DataFrame(
        {
            "class_name": ["no_helmet", "no_helmet", "gloves", "no_helmet", "person"]
            * 10,
            "confidence": [0.31, 0.29, 0.33, 0.28, 0.35] * 10,
            "bbox_area_norm": [0.0005, 0.0006, 0.0004, 0.0007, 0.005] * 10,
        }
    )

    out_json = tmp_path / "drift.json"
    out_html = tmp_path / "drift.html"
    summary = evaluate_proxy_drift(
        reference_df=ref_df,
        current_df=shifted_df,
        output_json=out_json,
        output_html=out_html,
        drift_share_threshold=0.5,
    )
    assert summary.assessment_scope == "proxy_statistical_only_without_ground_truth"
    assert summary.drift_detected is True
    assert summary.drifted_columns_share >= 0.5
    assert out_json.exists()
    assert out_html.exists()


def test_prefect_data_and_ct_flows_skip_and_trigger_cleanly(
    monkeypatch, tmp_path: Path
) -> None:
    executed: list[tuple[str, bool]] = []

    def _fake_stage(stage_name: str, dry_run: bool = True):
        executed.append((stage_name, dry_run))
        return {"stage": stage_name, "dry_run": dry_run, "status": "ok"}

    monkeypatch.setattr(
        "src.orchestration.prefect_flows.run_dvc_stage_task.fn", _fake_stage
    )
    # Run Prefect data flow cleanly via mock task callable.
    monkeypatch.setattr(
        "src.orchestration.prefect_flows.run_dvc_stage_task", _fake_stage
    )
    monkeypatch.setattr(
        "src.orchestration.prefect_flows.check_proxy_drift_task",
        lambda **_kwargs: {
            "drift_detected": False,
            "drifted_columns_share": 0.0,
        },
    )

    data_res = safeguard_data_pipeline_flow(dry_run=True)
    assert len(data_res["stages"]) == len(DATA_STAGES)
    assert [s for s, _ in executed] == list(DATA_STAGES)

    executed.clear()
    ct_skip = safeguard_training_ct_flow(
        dry_run=True, force_retrain=False, drift_override=False
    )
    assert ct_skip["retraining_triggered"] is False
    assert executed == []

    ct_run = safeguard_training_ct_flow(
        dry_run=True, force_retrain=False, drift_override=True
    )
    assert ct_run["retraining_triggered"] is True
    assert [s for s, _ in executed] == list(TRAINING_STAGES)
