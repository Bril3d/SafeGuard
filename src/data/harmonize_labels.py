"""Label harmonization helpers driven by configs/class_mapping.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml


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
