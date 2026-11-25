"""Execute the full read-only dataset audit on data/raw/ and generate reports."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.analyze_distribution import (
    compute_bbox_metrics,
    compute_class_distribution,
    export_distribution_csv,
    plot_bbox_analysis,
    plot_class_distribution,
)
from src.data.check_duplicates import find_duplicates
from src.data.inspect_dataset import discover_dataset_records, load_source_yaml
from src.data.validate_annotations import validate_dataset
from src.data.visualize_annotations import render_class_gallery

ROBOFLOW_DIR = Path("data/raw/roboflow/construction-site-safety-v30")
ULTRALYTICS_DIR = Path("data/raw/ultralytics/construction-ppe")
REPORTS_DIR = Path("reports/dataset")

CANDIDATE_MAPPING_BY_SOURCE: dict[str, dict[int, str]] = {
    "roboflow_v30": {
        8: "person",
        2: "helmet",
        11: "vest",
        1: "gloves",
        5: "no_helmet",
        7: "no_vest",
    },
    "ultralytics_ppe": {
        5: "person",
        0: "helmet",
        2: "vest",
        1: "gloves",
        6: "no_helmet",
        11: "no_vest",
    },
}


def main() -> None:
    """Run inspection, validation, distribution, dedup, and gallery generation."""
    rf_yaml = load_source_yaml(ROBOFLOW_DIR / "data.yaml")
    ul_yaml = load_source_yaml(ULTRALYTICS_DIR / "data.yaml")

    source_class_names = {
        "roboflow_v30": rf_yaml["names"],
        "ultralytics_ppe": ul_yaml["names"],
    }

    rf_records = discover_dataset_records("roboflow_v30", ROBOFLOW_DIR, "roboflow")
    ul_records = discover_dataset_records(
        "ultralytics_ppe", ULTRALYTICS_DIR, "ultralytics"
    )
    all_records = rf_records + ul_records

    rf_val = validate_dataset(
        "roboflow_v30",
        ROBOFLOW_DIR,
        "roboflow",
        rf_records,
        set(rf_yaml["names"].keys()),
    )
    ul_val = validate_dataset(
        "ultralytics_ppe",
        ULTRALYTICS_DIR,
        "ultralytics",
        ul_records,
        set(ul_yaml["names"].keys()),
    )

    dist_df = compute_class_distribution(all_records, source_class_names)
    export_distribution_csv(dist_df, REPORTS_DIR / "class_distribution.csv")
    plot_class_distribution(dist_df, REPORTS_DIR / "class_distribution.png")

    bbox_df = compute_bbox_metrics(all_records, source_class_names)
    candidate_names = {
        "roboflow_v30": {
            "Person",
            "Hardhat",
            "Safety Vest",
            "Gloves",
            "NO-Hardhat",
            "NO-Safety Vest",
        },
        "ultralytics_ppe": {
            "Person",
            "helmet",
            "vest",
            "gloves",
            "no_helmet",
            "no_vest",
        },
    }
    plot_bbox_analysis(bbox_df, REPORTS_DIR / "bbox_analysis.png", candidate_names)

    dup_report = find_duplicates(all_records, phash_threshold=6)

    gallery_paths = render_class_gallery(
        records=all_records,
        source_class_names=source_class_names,
        target_classes=CANDIDATE_MAPPING_BY_SOURCE,
        output_dir=REPORTS_DIR / "annotation_gallery",
        max_samples_per_class=6,
    )

    split_counts = {}
    for rec in all_records:
        split_counts.setdefault(rec.source, {}).setdefault(rec.split, 0)
        split_counts[rec.source][rec.split] += 1

    bbox_summary = {}
    for (src, cls_name), grp in bbox_df.groupby(["source", "class_name"]):
        if cls_name in candidate_names[src]:
            bbox_summary.setdefault(src, {})[cls_name] = {
                "count": int(len(grp)),
                "median_w": round(float(grp["width"].median()), 4),
                "median_h": round(float(grp["height"].median()), 4),
                "median_area": round(float(grp["area"].median()), 5),
                "median_ar": round(float(grp["aspect_ratio"].median()), 3),
                "p10_area": round(float(grp["area"].quantile(0.10)), 5),
                "p90_area": round(float(grp["area"].quantile(0.90)), 5),
            }

    summary = {
        "split_image_counts": split_counts,
        "yaml_classes": {
            "roboflow_v30": rf_yaml["names"],
            "ultralytics_ppe": ul_yaml["names"],
        },
        "validation": {
            "roboflow_v30": {
                "total_images": rf_val.total_images,
                "total_label_files": rf_val.total_label_files,
                "missing_labels": len(rf_val.missing_labels),
                "orphan_labels": len(rf_val.orphan_labels),
                "corrupted_images": len(rf_val.corrupted_images),
                "empty_label_files": len(rf_val.empty_label_files),
                "valid_box_count": rf_val.valid_box_count,
                "invalid_box_count": rf_val.invalid_box_count,
                "invalid_boxes_sample": rf_val.invalid_boxes[:10],
            },
            "ultralytics_ppe": {
                "total_images": ul_val.total_images,
                "total_label_files": ul_val.total_label_files,
                "missing_labels": len(ul_val.missing_labels),
                "orphan_labels": len(ul_val.orphan_labels),
                "corrupted_images": len(ul_val.corrupted_images),
                "empty_label_files": len(ul_val.empty_label_files),
                "valid_box_count": ul_val.valid_box_count,
                "invalid_box_count": ul_val.invalid_box_count,
                "invalid_boxes_sample": ul_val.invalid_boxes[:10],
            },
        },
        "duplicates": {
            "total_images": dup_report.total_images,
            "phash_threshold": dup_report.phash_threshold,
            "exact_groups": [
                {
                    "group_id": g.group_id,
                    "cross_source": g.cross_source,
                    "cross_split": g.cross_split,
                    "records": g.records,
                }
                for g in dup_report.exact_duplicate_groups
            ],
            "near_groups": [
                {
                    "group_id": g.group_id,
                    "min_hamming": g.min_hamming,
                    "cross_source": g.cross_source,
                    "cross_split": g.cross_split,
                    "records": g.records,
                }
                for g in dup_report.near_duplicate_groups
            ],
        },
        "candidate_bbox_summary": bbox_summary,
        "gallery_files": [str(p) for p in gallery_paths],
    }

    (REPORTS_DIR / "audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
