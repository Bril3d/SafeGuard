"""Stage 1 (ingest): Discover raw sources, read YAMLs, and record provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.data.inspect_dataset import (
    compute_sha256,
    discover_dataset_records,
    load_source_yaml,
)


def run_ingest(data_config_path: Path, output_manifest_path: Path) -> dict[str, Any]:
    """Read source YAML metadata dynamically and write machine-readable ingest manifest."""
    with data_config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    sources_meta: dict[str, Any] = {}
    records_out: list[dict[str, Any]] = []

    for source_key, src_cfg in cfg.get("sources", {}).items():
        root_dir = Path(src_cfg["root_dir"])
        yaml_path = Path(src_cfg["yaml_path"])
        layout = str(src_cfg["layout"])
        yaml_meta = load_source_yaml(yaml_path)

        license_path = Path(src_cfg["license_evidence"])
        license_sha256 = compute_sha256(license_path) if license_path.exists() else None

        sources_meta[source_key] = {
            "name": src_cfg["name"],
            "layout": layout,
            "root_dir": str(root_dir),
            "yaml_path": str(yaml_path),
            "yaml_sha256": compute_sha256(yaml_path),
            "license": src_cfg["license"],
            "license_evidence": str(license_path),
            "license_sha256": license_sha256,
            "locked_test_split": src_cfg.get("locked_test_split"),
            "nc": yaml_meta["nc"],
            "names": {str(k): v for k, v in yaml_meta["names"].items()},
        }

        discovered = discover_dataset_records(source_key, root_dir, layout)
        for rec in discovered:
            records_out.append(
                {
                    "source": rec.source,
                    "layout": rec.layout,
                    "original_split": rec.split,
                    "image_path": str(rec.image_path),
                    "label_path": str(rec.label_path) if rec.label_path else None,
                    "sha256": rec.sha256,
                }
            )

    manifest = {
        "stage": "ingest",
        "data_config_path": str(data_config_path),
        "total_images": len(records_out),
        "sources": sources_meta,
        "records": records_out,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint for DVC ingest stage."""
    parser = argparse.ArgumentParser(description="SafeGuard DVC ingest stage.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/interim/ingest_manifest.json")
    )
    args = parser.parse_args()
    run_ingest(args.config, args.output)


if __name__ == "__main__":
    main()
