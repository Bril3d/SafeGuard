"""Label harmonization helpers and DVC stage driven by configs/class_mapping.yaml."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from src.data.inspect_dataset import compute_sha256


def load_class_mapping(mapping_path: Path) -> dict[str, dict[int, int]]:
    """Load approved source-class-id -> canonical-id mappings from YAML."""
    with mapping_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    mapping: dict[str, dict[int, int]] = {}
    for entry in raw.get("canonical_classes", []):
        if entry.get("status") != "approved":
            continue
        canonical_id = int(entry["id"])
        for source_name, source_info in entry.get("sources", {}).items():
            mapping.setdefault(source_name, {})[int(source_info["id"])] = canonical_id
    return mapping


def _validate_mapping_against_ingest(
    mapping_raw: dict[str, Any],
    ingest_sources: dict[str, Any],
) -> None:
    """Verify class_mapping.yaml IDs and names match actual source YAMLs."""
    for entry in mapping_raw.get("canonical_classes", []):
        if entry.get("status") != "approved":
            continue
        for source_name, src_info in entry.get("sources", {}).items():
            if source_name not in ingest_sources:
                raise KeyError(f"Unknown source '{source_name}' in class_mapping.yaml")
            yaml_names = ingest_sources[source_name]["names"]
            src_id = str(src_info["id"])
            expected_name = src_info["name"]
            if yaml_names.get(src_id) != expected_name:
                raise ValueError(
                    f"Source YAML drift for {source_name} id={src_id}: "
                    f"expected '{expected_name}', found '{yaml_names.get(src_id)}'"
                )


def harmonize_label_lines(
    lines: list[str],
    source_name: str,
    mapping: dict[str, dict[int, int]],
) -> list[str]:
    """Remap approved class IDs to canonical IDs and drop unapproved classes."""
    source_map = mapping.get(source_name, {})
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            continue
        source_cls_id = int(parts[0])
        if source_cls_id not in source_map:
            continue
        canonical_id = source_map[source_cls_id]
        output.append(f"{canonical_id} {' '.join(parts[1:])}")
    return output


def harmonize_from_manifest(
    ingest_manifest_path: Path,
    mapping_path: Path,
    output_dir: Path,
    output_manifest_path: Path,
) -> dict[str, Any]:
    """Harmonize labels into interim storage without altering raw datasets."""
    ingest = json.loads(ingest_manifest_path.read_text(encoding="utf-8"))
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping_raw = yaml.safe_load(handle)

    _validate_mapping_against_ingest(mapping_raw, ingest["sources"])
    mapping = load_class_mapping(mapping_path)

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    records_out: list[dict[str, Any]] = []
    used_unique_ids: set[str] = set()

    for idx, rec in enumerate(ingest["records"]):
        source = rec["source"]
        orig_split = rec["original_split"]
        orig_img = Path(rec["image_path"])
        orig_lbl = Path(rec["label_path"]) if rec["label_path"] else None
        sha = rec["sha256"]

        base_uid = f"{source}__{orig_split}__{sha[:12]}"
        unique_id = base_uid if base_uid not in used_unique_ids else f"{base_uid}__{idx}"
        used_unique_ids.add(unique_id)

        dest_img = images_out / f"{unique_id}{orig_img.suffix.lower()}"
        dest_lbl = labels_out / f"{unique_id}.txt"
        shutil.copy2(orig_img, dest_img)

        raw_lines = (
            orig_lbl.read_text(encoding="utf-8").splitlines()
            if orig_lbl and orig_lbl.exists()
            else []
        )
        raw_non_empty = [ln.strip() for ln in raw_lines if ln.strip()]
        harmonized_lines = harmonize_label_lines(raw_non_empty, source, mapping)

        dropped_by_class: dict[str, int] = defaultdict(int)
        source_map = mapping.get(source, {})
        source_names = ingest["sources"][source]["names"]
        for ln in raw_non_empty:
            cid = int(ln.split()[0])
            if cid not in source_map:
                cname = source_names.get(str(cid), str(cid))
                dropped_by_class[cname] += 1

        dest_lbl.write_text(
            "\n".join(harmonized_lines) + ("\n" if harmonized_lines else ""),
            encoding="utf-8",
        )

        records_out.append(
            {
                "unique_id": unique_id,
                "source": source,
                "original_split": orig_split,
                "original_image_path": str(orig_img),
                "original_label_path": str(orig_lbl) if orig_lbl else None,
                "sha256": sha,
                "harmonized_image_path": str(dest_img),
                "harmonized_label_path": str(dest_lbl),
                "raw_box_count": len(raw_non_empty),
                "harmonized_box_count": len(harmonized_lines),
                "empty_after_harmonization": len(harmonized_lines) == 0,
                "dropped_non_intersection_boxes": dict(dropped_by_class),
                "retained": True,
                "retention_reason": (
                    "retained_background_after_non_intersection_filter"
                    if len(raw_non_empty) > 0 and len(harmonized_lines) == 0
                    else "retained"
                ),
            }
        )

    manifest = {
        "stage": "harmonize",
        "class_mapping_path": str(mapping_path),
        "class_mapping_sha256": compute_sha256(mapping_path),
        "approved_classes": [
            {"id": int(c["id"]), "name": c["name"]}
            for c in mapping_raw.get("canonical_classes", [])
            if c.get("status") == "approved"
        ],
        "total_images": len(records_out),
        "records": records_out,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint for DVC harmonize stage."""
    parser = argparse.ArgumentParser(description="SafeGuard DVC harmonize stage.")
    parser.add_argument(
        "--ingest-manifest",
        type=Path,
        default=Path("data/interim/ingest_manifest.json"),
    )
    parser.add_argument(
        "--mapping", type=Path, default=Path("configs/class_mapping.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/interim/harmonized")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/interim/harmonize_manifest.json"),
    )
    args = parser.parse_args()
    harmonize_from_manifest(
        args.ingest_manifest, args.mapping, args.output_dir, args.output_manifest
    )


if __name__ == "__main__":
    main()
