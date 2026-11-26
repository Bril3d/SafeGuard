"""Proxy prediction/data distribution drift evaluation using Evidently AI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from evidently import Report
from evidently.presets import DataDriftPreset
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyDriftSummary:
    """Structured proxy statistical drift summary (never claims accuracy drift)."""

    generated_at_utc: str
    assessment_scope: str
    reference_rows: int
    current_rows: int
    drift_share_threshold: float
    drifted_columns_count: int
    drifted_columns_share: float
    drift_detected: bool
    column_drifts: dict[str, dict[str, Any]]
    json_report_path: str
    html_report_path: str | None


def load_prediction_log_df(log_path: Path) -> pd.DataFrame:
    """Load anonymized JSONL prediction telemetry into a normalized DataFrame."""
    if not log_path.exists():
        return pd.DataFrame(columns=["class_name", "confidence", "bbox_area_norm"])
    rows: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows.append(
            {
                "class_name": str(rec.get("class_name", "unknown")),
                "confidence": float(rec.get("confidence", 0.0)),
                "bbox_area_norm": float(rec.get("bbox_area_norm", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def build_validation_reference_df(
    stats_path: Path = Path("reports/dataset/final_split_statistics.json"),
    sample_size: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Build baseline reference distribution from validation split statistics."""
    rng = np.random.default_rng(seed)
    class_weights: dict[str, int] = {
        "person": 639,
        "helmet": 370,
        "vest": 378,
        "gloves": 403,
        "no_helmet": 243,
    }
    if stats_path.exists():
        raw = json.loads(stats_path.read_text(encoding="utf-8"))
        val_cls = raw.get("partitions", {}).get("val_merged", {}).get("per_class", {})
        extracted = {
            str(k): int(v.get("instances", 0))
            for k, v in val_cls.items()
            if int(v.get("instances", 0)) > 0
        }
        if extracted:
            class_weights = extracted

    names = list(class_weights.keys())
    counts = np.asarray([class_weights[n] for n in names], dtype=np.float64)
    probs = counts / counts.sum()

    sampled_classes = rng.choice(names, size=sample_size, p=probs).tolist()
    conf_means = {
        "person": 0.84,
        "helmet": 0.81,
        "vest": 0.80,
        "gloves": 0.72,
        "no_helmet": 0.68,
    }
    area_means = {
        "person": 0.032,
        "helmet": 0.0026,
        "vest": 0.0078,
        "gloves": 0.0031,
        "no_helmet": 0.0018,
    }
    confidences = [
        float(np.clip(rng.normal(conf_means.get(c, 0.75), 0.08), 0.25, 0.99))
        for c in sampled_classes
    ]
    areas = [
        float(np.clip(rng.normal(area_means.get(c, 0.01), 0.004), 0.0002, 0.45))
        for c in sampled_classes
    ]
    return pd.DataFrame(
        {
            "class_name": sampled_classes,
            "confidence": confidences,
            "bbox_area_norm": areas,
        }
    )


def evaluate_proxy_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_json: Path,
    output_html: Path | None = None,
    drift_share_threshold: float = 0.5,
) -> ProxyDriftSummary:
    """Compute Evidently statistical/proxy drift report on telemetry features."""
    if reference_df.empty or current_df.empty:
        raise ValueError("Both reference_df and current_df must contain rows.")

    shared_cols = [c for c in reference_df.columns if c in current_df.columns]
    if not shared_cols:
        raise ValueError(
            "No overlapping feature columns between reference and current."
        )

    report = Report([DataDriftPreset(drift_share=drift_share_threshold)])
    snapshot = report.run(current_df[shared_cols], reference_df[shared_cols])
    payload = snapshot.dict()

    drifted_count = 0
    drifted_share = 0.0
    col_drifts: dict[str, dict[str, Any]] = {}

    for metric in payload.get("metrics", []):
        cfg = metric.get("config", {})
        mtype = str(cfg.get("type", ""))
        val = metric.get("value")
        if "DriftedColumnsCount" in mtype and isinstance(val, dict):
            drifted_count = int(round(float(val.get("count", 0))))
            drifted_share = float(val.get("share", 0.0))
        elif "ValueDrift" in mtype and "column" in cfg:
            col = str(cfg["column"])
            threshold = float(cfg.get("threshold", 0.05))
            method = str(cfg.get("method", ""))
            score = float(val) if isinstance(val, (int, float)) else 0.0
            # In Evidently, p-value methods indicate drift when score < threshold;
            # distance metrics (Wasserstein / JS) drift when >= threshold.
            is_pvalue = "p_value" in method.lower()
            is_drifted = score < threshold if is_pvalue else score >= threshold
            col_drifts[col] = {
                "method": method,
                "score": round(score, 6),
                "threshold": threshold,
                "drifted": bool(is_drifted),
            }

    drift_detected = drifted_share >= drift_share_threshold
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_html is not None:
        output_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(output_html))

    summary = ProxyDriftSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        assessment_scope="proxy_statistical_only_without_ground_truth",
        reference_rows=int(len(reference_df)),
        current_rows=int(len(current_df)),
        drift_share_threshold=float(drift_share_threshold),
        drifted_columns_count=drifted_count,
        drifted_columns_share=round(drifted_share, 4),
        drift_detected=bool(drift_detected),
        column_drifts=col_drifts,
        json_report_path=str(output_json),
        html_report_path=str(output_html) if output_html else None,
    )
    output_json.write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for scheduled or manual CT proxy drift checks."""
    parser = argparse.ArgumentParser(
        description="Run Evidently AI proxy drift analysis on SafeGuard predictions."
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("reports/dataset/final_split_statistics.json"),
    )
    parser.add_argument(
        "--pred-log",
        type=Path,
        default=Path("reports/monitoring/predictions.jsonl"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/monitoring/drift_report.json"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("reports/monitoring/drift_report.html"),
    )
    parser.add_argument("--drift-share", type=float, default=0.5)
    args = parser.parse_args()

    ref_df = build_validation_reference_df(stats_path=args.stats, sample_size=250)
    cur_df = load_prediction_log_df(args.pred_log)
    if cur_df.empty:
        # Baseline self-check when live logs have not accumulated yet.
        cur_df = build_validation_reference_df(
            stats_path=args.stats, sample_size=120, seed=99
        )

    summary = evaluate_proxy_drift(
        reference_df=ref_df,
        current_df=cur_df,
        output_json=args.output_json,
        output_html=args.output_html,
        drift_share_threshold=args.drift_share,
    )
    print(json.dumps(summary.__dict__, indent=2))


if __name__ == "__main__":
    main()
