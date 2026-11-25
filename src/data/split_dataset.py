"""Stage 5 (split): Leakage-aware splitting and source-specific test preservation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
from typing import Any

import yaml

from src.data.harmonize_labels import load_class_mapping


def _group_aware_holdout(
    ul_train_records: list[dict[str, Any]],
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Ultralytics train records into train and test holdout by scene_group_id."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in ul_train_records:
        groups[rec["scene_group_id"]].append(rec)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    target_count = max(1, round(len(ul_train_records) * holdout_ratio))
    holdout_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []

    for gkey in group_keys:
        if len(holdout_records) < target_count:
            holdout_records.extend(groups[gkey])
        else:
            train_records.extend(groups[gkey])

    return train_records, holdout_records


def _copy_partition_files(
    records: list[dict[str, Any]],
    partition_dir: Path,
) -> None:
    img_dir = partition_dir / "images"
    lbl_dir = partition_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        src_img = Path(rec["harmonized_image_path"])
        src_lbl = Path(rec["harmonized_label_path"])
        shutil.copy2(src_img, img_dir / src_img.name)
        shutil.copy2(src_lbl, lbl_dir / src_lbl.name)


def _write_yolo_yaml(
    yaml_path: Path,
    base_dir: Path,
    names: dict[int, str],
    train: str | list[str] | None = None,
    val: str | list[str] | None = None,
    test: str | list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {"path": str(base_dir.resolve())}
    if train is not None:
        payload["train"] = train
    if val is not None:
        payload["val"] = val
    if test is not None:
        payload["test"] = test
    payload["nc"] = len(names)
    payload["names"] = names
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def split_from_manifest(
    dedup_manifest_path: Path,
    mapping_path: Path,
    params_path: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Create leakage-free processed partitions, YOLO YAMLs, and split manifest."""
    dedup = json.loads(dedup_manifest_path.read_text(encoding="utf-8"))
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle).get("data_pipeline", {})
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_raw = yaml.safe_load(handle)

    _ = load_class_mapping(mapping_path)
    canonical_names = {
        int(c["id"]): str(c["name"])
        for c in mapping_raw.get("canonical_classes", [])
        if c.get("status") == "approved"
    }

    holdout_ratio = float(params.get("ultralytics_holdout_ratio", 0.15))
    seed = int(params.get("random_seed", 42))

    eligible = [r for r in dedup["records"] if r["decision"].startswith("keep")]

    rf_train = [
        r
        for r in eligible
        if r["source"] == "roboflow_v30" and r["original_split"] == "train"
    ]
    rf_val = [
        r
        for r in eligible
        if r["source"] == "roboflow_v30" and r["original_split"] == "valid"
    ]
    rf_test = [
        r
        for r in eligible
        if r["source"] == "roboflow_v30" and r["original_split"] == "test"
    ]

    ul_train_pool = [
        r
        for r in eligible
        if r["source"] == "ultralytics_ppe" and r["original_split"] == "train"
    ]
    ul_val = [
        r
        for r in eligible
        if r["source"] == "ultralytics_ppe" and r["original_split"] == "val"
    ]
    ul_train, ul_test = _group_aware_holdout(ul_train_pool, holdout_ratio, seed)

    partitions: dict[str, list[dict[str, Any]]] = {
        "train_roboflow": rf_train,
        "val_roboflow": rf_val,
        "test_roboflow": rf_test,
        "train_ultralytics": ul_train,
        "val_ultralytics": ul_val,
        "test_ultralytics": ul_test,
    }

    if processed_dir.exists():
        for child in processed_dir.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    processed_dir.mkdir(parents=True, exist_ok=True)

    for part_name, part_records in partitions.items():
        _copy_partition_files(part_records, processed_dir / part_name)

    _write_yolo_yaml(
        processed_dir / "merged.yaml",
        processed_dir,
        canonical_names,
        train=["train_roboflow/images", "train_ultralytics/images"],
        val=["val_roboflow/images", "val_ultralytics/images"],
        test="test_roboflow/images",
    )
    _write_yolo_yaml(
        processed_dir / "roboflow_only.yaml",
        processed_dir,
        canonical_names,
        train="train_roboflow/images",
        val="val_roboflow/images",
        test="test_roboflow/images",
    )
    _write_yolo_yaml(
        processed_dir / "ultralytics_only.yaml",
        processed_dir,
        canonical_names,
        train="train_ultralytics/images",
        val="val_ultralytics/images",
        test="test_ultralytics/images",
    )
    _write_yolo_yaml(
        processed_dir / "test_roboflow.yaml",
        processed_dir,
        canonical_names,
        train="train_roboflow/images",
        val="val_roboflow/images",
        test="test_roboflow/images",
    )
    _write_yolo_yaml(
        processed_dir / "test_ultralytics.yaml",
        processed_dir,
        canonical_names,
        train="train_ultralytics/images",
        val="val_ultralytics/images",
        test="test_ultralytics/images",
    )

    train_shas = {r["sha256"] for r in rf_train + ul_train}
    val_shas = {r["sha256"] for r in rf_val + ul_val}
    test_shas = {r["sha256"] for r in rf_test + ul_test}

    train_groups = {r["scene_group_id"] for r in rf_train + ul_train}
    val_groups = {r["scene_group_id"] for r in rf_val + ul_val}
    test_groups = {r["scene_group_id"] for r in rf_test + ul_test}

    leakage_check = {
        "train_val_sha_overlap": len(train_shas & val_shas),
        "train_test_sha_overlap": len(train_shas & test_shas),
        "val_test_sha_overlap": len(val_shas & test_shas),
        "train_test_scene_group_overlap": len(train_groups & test_groups),
        "train_val_scene_group_overlap": len(train_groups & val_groups),
    }

    if (
        leakage_check["train_test_sha_overlap"] > 0
        or leakage_check["train_test_scene_group_overlap"] > 0
    ):
        raise RuntimeError(f"Split leakage detected: {leakage_check}")

    manifest = {
        "stage": "split",
        "ultralytics_holdout_ratio": holdout_ratio,
        "random_seed": seed,
        "canonical_names": {str(k): v for k, v in canonical_names.items()},
        "counts": {
            **{k: len(v) for k, v in partitions.items()},
            "train_merged": len(rf_train) + len(ul_train),
            "val_merged": len(rf_val) + len(ul_val),
        },
        "leakage_check": leakage_check,
        "partitions": {
            part_name: [
                {
                    "unique_id": r["unique_id"],
                    "source": r["source"],
                    "original_split": r["original_split"],
                    "scene_group_id": r["scene_group_id"],
                    "sha256": r["sha256"],
                    "harmonized_box_count": r["harmonized_box_count"],
                    "image_path": str(
                        processed_dir
                        / part_name
                        / "images"
                        / Path(r["harmonized_image_path"]).name
                    ),
                    "label_path": str(
                        processed_dir
                        / part_name
                        / "labels"
                        / Path(r["harmonized_label_path"]).name
                    ),
                }
                for r in part_records
            ]
            for part_name, part_records in partitions.items()
        },
    }

    manifest_path = processed_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint for DVC split stage."""
    parser = argparse.ArgumentParser(description="SafeGuard DVC split stage.")
    parser.add_argument(
        "--dedup-manifest",
        type=Path,
        default=Path("data/interim/dedup_manifest.json"),
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/class_mapping.yaml")
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/processed")
    )
    args = parser.parse_args()
    split_from_manifest(
        args.dedup_manifest, args.mapping, args.params, args.processed_dir
    )


if __name__ == "__main__":
    main()
