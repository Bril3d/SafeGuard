"""TDD unit tests for the SafeGuard dataset inspection and validation pipeline."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from src.data.analyze_distribution import (
    compute_bbox_metrics,
    compute_class_distribution,
    export_distribution_csv,
)
from src.data.check_duplicates import find_duplicates
from src.data.harmonize_labels import harmonize_label_lines, load_class_mapping
from src.data.inspect_dataset import discover_dataset_records, load_source_yaml
from src.data.validate_annotations import validate_dataset
from src.data.visualize_annotations import render_class_gallery


def _create_image(path: Path, size: tuple[int, int] = (100, 100), color=(20, 40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill=(220, 180, 40))
    img.save(path, format="JPEG")


def _write_roboflow_fixture(root: Path) -> Path:
    rf_dir = root / "roboflow" / "construction-site-safety-v30"
    rf_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": 3,
        "names": ["Hardhat", "Person", "Excavator"],
    }
    (rf_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")
    _create_image(rf_dir / "train" / "images" / "img1.jpg")
    (rf_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (rf_dir / "train" / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.6 0.4 0.6\n2 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )
    _create_image(rf_dir / "valid" / "images" / "img2.jpg", color=(80, 30, 30))
    (rf_dir / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    (rf_dir / "valid" / "labels" / "img2.txt").write_text(
        "1 0.4 0.4 0.3 0.5\n", encoding="utf-8"
    )
    _create_image(rf_dir / "test" / "images" / "img3.jpg", color=(30, 90, 40))
    (rf_dir / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (rf_dir / "test" / "labels" / "img3.txt").write_text(
        "0 0.3 0.3 0.1 0.1\n", encoding="utf-8"
    )
    return rf_dir


def _write_ultralytics_fixture(root: Path) -> Path:
    ul_dir = root / "ultralytics" / "construction-ppe"
    ul_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = {
        "path": "construction-ppe",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "helmet", 1: "Person", 2: "boots"},
    }
    (ul_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml), encoding="utf-8")
    _create_image(ul_dir / "images" / "train" / "u1.jpg", color=(15, 75, 110))
    (ul_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ul_dir / "labels" / "train" / "u1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n2 0.5 0.8 0.1 0.1\n", encoding="utf-8"
    )
    _create_image(ul_dir / "images" / "val" / "u2.jpg", color=(120, 20, 90))
    (ul_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (ul_dir / "labels" / "val" / "u2.txt").write_text(
        "1 0.5 0.5 0.3 0.6\n", encoding="utf-8"
    )
    return ul_dir


def test_inspect_dataset_discovers_both_layouts_and_provenance(tmp_path: Path) -> None:
    rf_dir = _write_roboflow_fixture(tmp_path)
    ul_dir = _write_ultralytics_fixture(tmp_path)

    rf_meta = load_source_yaml(rf_dir / "data.yaml")
    ul_meta = load_source_yaml(ul_dir / "data.yaml")
    assert rf_meta["names"] == {0: "Hardhat", 1: "Person", 2: "Excavator"}
    assert ul_meta["names"] == {0: "helmet", 1: "Person", 2: "boots"}

    rf_records = discover_dataset_records("roboflow_v30", rf_dir, "roboflow")
    ul_records = discover_dataset_records("ultralytics_ppe", ul_dir, "ultralytics")
    assert len(rf_records) == 3
    assert {r.split for r in rf_records} == {"train", "valid", "test"}
    assert len(ul_records) == 2
    assert {r.split for r in ul_records} == {"train", "val"}
    assert all(len(r.sha256) == 64 for r in rf_records + ul_records)


def test_validate_annotations_detects_corrupt_missing_and_bad_boxes(
    tmp_path: Path,
) -> None:
    rf_dir = _write_roboflow_fixture(tmp_path)
    # Missing label for extra image
    _create_image(rf_dir / "train" / "images" / "no_label.jpg")
    # Orphan label without image
    (rf_dir / "train" / "labels" / "orphan.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )
    # Corrupted image + invalid YOLO annotations
    corrupt_img = rf_dir / "train" / "images" / "corrupt.jpg"
    corrupt_img.write_bytes(b"not-a-jpeg")
    (rf_dir / "train" / "labels" / "corrupt.txt").write_text(
        "99 0.5 0.5 0.2 0.2\n"  # unknown class_id
        "0 1.2 0.5 0.2 0.2\n"  # cx > 1
        "0 0.5 0.5 -0.1 0.2\n"  # negative width
        "0 0.5 0.5\n",  # bad column count
        encoding="utf-8",
    )

    records = discover_dataset_records("roboflow_v30", rf_dir, "roboflow")
    report = validate_dataset("roboflow_v30", rf_dir, "roboflow", records, {0, 1, 2})

    assert len(report.missing_labels) == 1
    assert len(report.orphan_labels) == 1
    assert len(report.corrupted_images) == 1
    assert report.invalid_box_count >= 4


def test_analyze_distribution_and_bbox_metrics(tmp_path: Path) -> None:
    rf_dir = _write_roboflow_fixture(tmp_path)
    records = discover_dataset_records("roboflow_v30", rf_dir, "roboflow")
    class_names = {0: "Hardhat", 1: "Person", 2: "Excavator"}

    dist_df = compute_class_distribution(records, {"roboflow_v30": class_names})
    hardhat_row = dist_df[
        (dist_df["class_name"] == "Hardhat") & (dist_df["split"] == "train")
    ].iloc[0]
    assert hardhat_row["instance_count"] == 1
    assert hardhat_row["image_count"] == 1

    csv_path = tmp_path / "class_distribution.csv"
    export_distribution_csv(dist_df, csv_path)
    assert csv_path.exists()

    bbox_df = compute_bbox_metrics(records, {"roboflow_v30": class_names})
    assert len(bbox_df) == 5
    assert set(["width", "height", "area", "aspect_ratio"]).issubset(bbox_df.columns)


def test_check_duplicates_detects_exact_and_near_duplicates(tmp_path: Path) -> None:
    rf_dir = _write_roboflow_fixture(tmp_path)
    ul_dir = _write_ultralytics_fixture(tmp_path)

    # Create an exact byte copy from rf test -> ul train (cross-source & cross-split!)
    src_bytes = (rf_dir / "test" / "images" / "img3.jpg").read_bytes()
    (ul_dir / "images" / "train" / "dup_from_rf_test.jpg").write_bytes(src_bytes)
    (ul_dir / "labels" / "train" / "dup_from_rf_test.txt").write_text(
        "0 0.3 0.3 0.1 0.1\n", encoding="utf-8"
    )

    records = discover_dataset_records(
        "roboflow_v30", rf_dir, "roboflow"
    ) + discover_dataset_records("ultralytics_ppe", ul_dir, "ultralytics")
    dup_report = find_duplicates(records, phash_threshold=4)

    assert len(dup_report.exact_duplicate_groups) == 1
    assert dup_report.exact_duplicate_groups[0].cross_split is True
    assert dup_report.exact_duplicate_groups[0].cross_source is True


def test_visualize_annotations_generates_gallery(tmp_path: Path) -> None:
    rf_dir = _write_roboflow_fixture(tmp_path)
    records = discover_dataset_records("roboflow_v30", rf_dir, "roboflow")
    out_dir = tmp_path / "gallery"
    generated = render_class_gallery(
        records=records,
        source_class_names={"roboflow_v30": {0: "Hardhat", 1: "Person"}},
        target_classes={"roboflow_v30": {0: "helmet", 1: "person"}},
        output_dir=out_dir,
        max_samples_per_class=2,
    )
    assert len(generated) >= 1
    assert all(p.exists() and p.stat().st_size > 0 for p in generated)


def test_harmonize_labels_maps_approved_and_excludes_others(tmp_path: Path) -> None:
    mapping_file = tmp_path / "class_mapping.yaml"
    mapping_data = {
        "schema_version": "1.0",
        "status": "provisional",
        "canonical_classes": [
            {
                "id": 0,
                "name": "person",
                "status": "approved",
                "sources": {
                    "roboflow_v30": {"id": 8, "name": "Person"},
                    "ultralytics_ppe": {"id": 5, "name": "Person"},
                },
            },
            {
                "id": 1,
                "name": "helmet",
                "status": "approved",
                "sources": {
                    "roboflow_v30": {"id": 2, "name": "Hardhat"},
                    "ultralytics_ppe": {"id": 0, "name": "helmet"},
                },
            },
        ],
        "candidate_classes": [
            {
                "name": "no_vest",
                "status": "pending_approval",
                "sources": {
                    "roboflow_v30": {"id": 7, "name": "NO-Safety Vest"},
                    "ultralytics_ppe": {"id": 11, "name": "no_vest"},
                },
            }
        ],
    }
    mapping_file.write_text(yaml.safe_dump(mapping_data), encoding="utf-8")

    mapping = load_class_mapping(mapping_file)
    raw_lines = [
        "8 0.5 0.5 0.3 0.6",
        "2 0.5 0.2 0.1 0.1",
        "7 0.5 0.5 0.2 0.2",  # candidate only -> excluded unless approved
        "0 0.1 0.1 0.1 0.1",  # Excavator -> excluded
    ]
    harmonized = harmonize_label_lines(raw_lines, "roboflow_v30", mapping)
    assert harmonized == ["0 0.5 0.5 0.3 0.6", "1 0.5 0.2 0.1 0.1"]
