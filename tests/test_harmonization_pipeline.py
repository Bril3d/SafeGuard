"""TDD tests for the SafeGuard dataset harmonization and DVC pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

from src.data.check_duplicates import deduplicate_from_manifest
from src.data.harmonize_labels import harmonize_from_manifest
from src.data.ingest import run_ingest
from src.data.split_dataset import split_from_manifest
from src.data.stats import compute_processed_statistics
from src.data.validate_annotations import validate_from_manifest


def _make_img(
    path: Path, color: tuple[int, int, int], box_x: int = 10, seed_idx: int = 0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (120, 120), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([box_x, 10, box_x + 40, 60], fill=(240, 200, 30))
    for k in range((seed_idx % 5) + 1):
        draw.line(
            [(0, 20 * k + (seed_idx * 7) % 30), (120, 120 - 15 * k)],
            fill=(255 - color[0], (color[1] + 90) % 255, 220),
            width=4,
        )
    if seed_idx % 2 == 1:
        draw.ellipse([70, 65, 115, 110], fill=(30, 230, 200))
    img.save(path, format="JPEG")


def _setup_synthetic_workspace(root: Path) -> tuple[Path, Path, Path]:
    rf = root / "data/raw/roboflow/construction-site-safety-v30"
    ul = root / "data/raw/ultralytics/construction-ppe"
    rf.mkdir(parents=True, exist_ok=True)
    ul.mkdir(parents=True, exist_ok=True)

    (rf / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
                "nc": 3,
                "names": ["Excavator", "Hardhat", "Person"],
            }
        ),
        encoding="utf-8",
    )
    (rf / "README.dataset.txt").write_text("License: CC BY 4.0\n", encoding="utf-8")

    # RF train: one normal, one leaked copy of RF test, one non-intersection-only image
    _make_img(rf / "train/images/rf_tr_1.jpg", (20, 50, 90), box_x=10)
    (rf / "train/labels").mkdir(parents=True, exist_ok=True)
    (rf / "train/labels/rf_tr_1.txt").write_text(
        "1 0.5 0.2 0.2 0.2\n2 0.5 0.5 0.4 0.7\n0 0.2 0.2 0.1 0.1\n",
        encoding="utf-8",
    )

    _make_img(rf / "train/images/rf_tr_only_excavator.jpg", (110, 30, 40), box_x=35)
    (rf / "train/labels/rf_tr_only_excavator.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )

    _make_img(rf / "valid/images/rf_val_1.jpg", (40, 120, 50), box_x=20)
    (rf / "valid/labels").mkdir(parents=True, exist_ok=True)
    (rf / "valid/labels/rf_val_1.txt").write_text(
        "2 0.5 0.5 0.3 0.6\n", encoding="utf-8"
    )

    _make_img(rf / "test/images/rf_test_1.jpg", (160, 40, 130), box_x=50)
    (rf / "test/labels").mkdir(parents=True, exist_ok=True)
    (rf / "test/labels/rf_test_1.txt").write_text(
        "1 0.5 0.3 0.2 0.2\n", encoding="utf-8"
    )
    (rf / "train/images/rf_tr_leaked_test.jpg").write_bytes(
        (rf / "test/images/rf_test_1.jpg").read_bytes()
    )
    (rf / "train/labels/rf_tr_leaked_test.txt").write_text(
        "1 0.5 0.3 0.2 0.2\n", encoding="utf-8"
    )

    # Ultralytics fixture: 8 train images (across 4 scene groups) + 2 val images
    (ul / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "construction-ppe",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "helmet", 1: "Person", 2: "boots"},
            }
        ),
        encoding="utf-8",
    )
    (ul / "LICENSE.txt").write_text("CC BY 4.0\n", encoding="utf-8")
    (ul / "images/train").mkdir(parents=True, exist_ok=True)
    (ul / "labels/train").mkdir(parents=True, exist_ok=True)
    (ul / "images/val").mkdir(parents=True, exist_ok=True)
    (ul / "labels/val").mkdir(parents=True, exist_ok=True)

    for idx in range(8):
        color = (15 * idx + 10, 200 - 15 * idx, (40 * idx + 30) % 255)
        _make_img(
            ul / f"images/train/ul_tr_{idx}.jpg",
            color=color,
            box_x=10 + 5 * idx,
            seed_idx=idx + 1,
        )
        (ul / f"labels/train/ul_tr_{idx}.txt").write_text(
            "0 0.5 0.2 0.2 0.2\n1 0.5 0.5 0.4 0.7\n2 0.5 0.9 0.2 0.1\n",
            encoding="utf-8",
        )

    for idx in range(2):
        _make_img(
            ul / f"images/val/ul_val_{idx}.jpg",
            color=(220 - 30 * idx, 50 + 40 * idx, 180),
            box_x=25 + 10 * idx,
            seed_idx=idx + 10,
        )
        (ul / f"labels/val/ul_val_{idx}.txt").write_text(
            "1 0.5 0.5 0.3 0.6\n", encoding="utf-8"
        )

    data_cfg = root / "configs/data.yaml"
    data_cfg.parent.mkdir(parents=True, exist_ok=True)
    data_cfg.write_text(
        yaml.safe_dump(
            {
                "sources": {
                    "roboflow_v30": {
                        "name": "Roboflow Test Fixture",
                        "layout": "roboflow",
                        "root_dir": str(rf),
                        "yaml_path": str(rf / "data.yaml"),
                        "license": "CC BY 4.0",
                        "license_evidence": str(rf / "README.dataset.txt"),
                        "locked_test_split": "test",
                    },
                    "ultralytics_ppe": {
                        "name": "Ultralytics Test Fixture",
                        "layout": "ultralytics",
                        "root_dir": str(ul),
                        "yaml_path": str(ul / "data.yaml"),
                        "license": "CC BY 4.0",
                        "license_evidence": str(ul / "LICENSE.txt"),
                        "locked_test_split": None,
                    },
                },
                "class_mapping_path": str(root / "configs/class_mapping.yaml"),
                "paths": {
                    "interim_dir": str(root / "data/interim"),
                    "processed_dir": str(root / "data/processed"),
                    "reports_dir": str(root / "reports/dataset"),
                },
            }
        ),
        encoding="utf-8",
    )

    mapping_cfg = root / "configs/class_mapping.yaml"
    mapping_cfg.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "canonical_classes": [
                    {
                        "id": 0,
                        "name": "person",
                        "status": "approved",
                        "sources": {
                            "roboflow_v30": {"id": 2, "name": "Person"},
                            "ultralytics_ppe": {"id": 1, "name": "Person"},
                        },
                    },
                    {
                        "id": 1,
                        "name": "helmet",
                        "status": "approved",
                        "sources": {
                            "roboflow_v30": {"id": 1, "name": "Hardhat"},
                            "ultralytics_ppe": {"id": 0, "name": "helmet"},
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    params_cfg = root / "params.yaml"
    params_cfg.write_text(
        yaml.safe_dump(
            {
                "data_pipeline": {
                    "phash_threshold": 6,
                    "enable_scene_grouping": True,
                    "ultralytics_holdout_ratio": 0.25,
                    "random_seed": 42,
                    "keep_empty_labels_after_harmonization": True,
                }
            }
        ),
        encoding="utf-8",
    )
    return data_cfg, mapping_cfg, params_cfg


def test_end_to_end_harmonization_pipeline_preserves_raw_and_prevents_leakage(
    tmp_path: Path,
) -> None:
    data_cfg, mapping_cfg, params_cfg = _setup_synthetic_workspace(tmp_path)
    interim_dir = tmp_path / "data/interim"
    processed_dir = tmp_path / "data/processed"
    reports_dir = tmp_path / "reports/dataset"

    # 1. Ingest
    ingest_manifest = run_ingest(data_cfg, interim_dir / "ingest_manifest.json")
    assert ingest_manifest["total_images"] == 15

    # 2. Validate
    val_report = validate_from_manifest(
        interim_dir / "ingest_manifest.json",
        interim_dir / "validation_report.json",
    )
    assert val_report["total_invalid_boxes"] == 0
    assert val_report["total_corrupted_images"] == 0

    # 3. Harmonize (unique filenames, non-intersection dropped, empty-after-filter kept)
    harm_manifest = harmonize_from_manifest(
        interim_dir / "ingest_manifest.json",
        mapping_cfg,
        interim_dir / "harmonized",
        interim_dir / "harmonize_manifest.json",
    )
    assert harm_manifest["total_images"] == 15
    unique_ids = [r["unique_id"] for r in harm_manifest["records"]]
    assert len(unique_ids) == len(set(unique_ids))

    excavator_only = [
        r
        for r in harm_manifest["records"]
        if r["original_image_path"].endswith("rf_tr_only_excavator.jpg")
    ][0]
    assert excavator_only["empty_after_harmonization"] is True
    assert excavator_only["retained"] is True

    # 4. Deduplicate (removes rf_tr_leaked_test.jpg while keeping rf_test_1.jpg)
    dedup_manifest = deduplicate_from_manifest(
        interim_dir / "harmonize_manifest.json",
        params_cfg,
        interim_dir / "dedup_manifest.json",
        reports_dir / "deduplication_report.json",
    )
    kept_originals = {
        Path(r["original_image_path"]).name
        for r in dedup_manifest["records"]
        if r["decision"].startswith("keep")
    }
    dropped = [r for r in dedup_manifest["records"] if r["decision"].startswith("drop")]
    assert "rf_test_1.jpg" in kept_originals
    assert "rf_tr_leaked_test.jpg" not in kept_originals
    assert any(
        Path(r["original_image_path"]).name == "rf_tr_leaked_test.jpg"
        and r["decision"] == "drop_test_leakage"
        for r in dropped
    )

    # 5. Split (locked RF test, preserved UL val, group-aware UL holdout, merged)
    split_manifest = split_from_manifest(
        interim_dir / "dedup_manifest.json",
        mapping_cfg,
        params_cfg,
        processed_dir,
    )
    assert split_manifest["leakage_check"]["train_test_sha_overlap"] == 0
    assert split_manifest["leakage_check"]["train_val_sha_overlap"] == 0
    assert split_manifest["counts"]["test_roboflow"] == 1
    assert split_manifest["counts"]["val_ultralytics"] == 2
    assert split_manifest["counts"]["test_ultralytics"] == 2
    assert split_manifest["counts"]["train_ultralytics"] == 6
    assert (processed_dir / "merged.yaml").exists()
    assert (processed_dir / "test_roboflow.yaml").exists()
    assert (processed_dir / "test_ultralytics.yaml").exists()

    # 6. Statistics
    stats = compute_processed_statistics(
        processed_dir / "split_manifest.json",
        mapping_cfg,
        reports_dir,
    )
    assert Path(stats["csv_path"]).exists()
    assert Path(stats["json_path"]).exists()
    assert Path(stats["plot_path"]).exists()
    saved = json.loads(Path(stats["json_path"]).read_text(encoding="utf-8"))
    assert saved["partitions"]["test_roboflow"]["images"] == 1
