"""Stage 6 (statistics): Final processed split & class distribution reporting."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.data.inspect_dataset import parse_yolo_label_file  # noqa: E402


def compute_processed_statistics(
    split_manifest_path: Path,
    mapping_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    """Compute final per-split and per-class statistics for processed datasets."""
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_raw = yaml.safe_load(handle)

    canonical_names = {
        int(c["id"]): str(c["name"])
        for c in mapping_raw.get("canonical_classes", [])
        if c.get("status") == "approved"
    }

    base_parts = split_manifest["partitions"]
    aggregate_parts: dict[str, list[dict[str, Any]]] = {
        **base_parts,
        "train_merged": base_parts["train_roboflow"] + base_parts["train_ultralytics"],
        "val_merged": base_parts["val_roboflow"] + base_parts["val_ultralytics"],
    }

    rows: list[dict[str, Any]] = []
    partition_summaries: dict[str, Any] = {}

    for part_name, recs in aggregate_parts.items():
        inst_counts: dict[int, int] = defaultdict(int)
        img_sets: dict[int, set[str]] = defaultdict(set)
        empty_count = 0
        total_boxes = 0

        for rec in recs:
            boxes = parse_yolo_label_file(Path(rec["label_path"]))
            if not boxes:
                empty_count += 1
            for cls_id, *_ in boxes:
                inst_counts[cls_id] += 1
                img_sets[cls_id].add(rec["unique_id"])
                total_boxes += 1

        partition_summaries[part_name] = {
            "images": len(recs),
            "empty_label_images": empty_count,
            "total_boxes": total_boxes,
            "per_class": {},
        }

        for cls_id, cls_name in sorted(canonical_names.items()):
            cnt = inst_counts.get(cls_id, 0)
            imgs = len(img_sets.get(cls_id, set()))
            partition_summaries[part_name]["per_class"][cls_name] = {
                "instances": cnt,
                "images": imgs,
            }
            rows.append(
                {
                    "partition": part_name,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "instance_count": cnt,
                    "image_count": imgs,
                    "partition_images": len(recs),
                }
            )

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "final_split_statistics.csv"
    json_path = reports_dir / "final_split_statistics.json"
    plot_path = reports_dir / "final_split_distribution.png"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    summary_payload = {
        "stage": "statistics",
        "leakage_check": split_manifest["leakage_check"],
        "partitions": partition_summaries,
    }
    json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    pivot = df.pivot(
        index="class_name", columns="partition", values="instance_count"
    ).fillna(0)
    ordered_cols = [
        c
        for c in [
            "train_merged",
            "val_merged",
            "test_roboflow",
            "test_ultralytics",
        ]
        if c in pivot.columns
    ]
    ax = pivot[ordered_cols].plot(kind="bar", figsize=(10, 5))
    ax.set_title("Final Harmonized Dataset — Instances per Class & Partition")
    ax.set_xlabel("Canonical Class")
    ax.set_ylabel("Instance Count")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "plot_path": str(plot_path),
    }


def main() -> None:
    """CLI entrypoint for DVC statistics stage."""
    parser = argparse.ArgumentParser(description="SafeGuard DVC statistics stage.")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/processed/split_manifest.json"),
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/class_mapping.yaml")
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=Path("reports/dataset")
    )
    args = parser.parse_args()
    compute_processed_statistics(
        args.split_manifest, args.mapping, args.reports_dir
    )


if __name__ == "__main__":
    main()
