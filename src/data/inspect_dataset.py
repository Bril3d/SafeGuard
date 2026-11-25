"""Dataset discovery and YAML metadata inspection for SafeGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)


@dataclass(frozen=True)
class DatasetRecord:
    """Provenance and annotation record for a single source image."""

    source: str
    layout: str
    split: str
    image_path: Path
    label_path: Path | None
    sha256: str
    annotations: list[tuple[int, float, float, float, float]] = field(
        default_factory=list
    )


def compute_sha256(file_path: Path) -> str:
    """Compute hex SHA-256 digest for a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_source_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load and normalize a dataset YAML configuration file."""
    with yaml_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    names_raw = raw.get("names", {})
    if isinstance(names_raw, list):
        names = {idx: str(name) for idx, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        raise ValueError(f"Unsupported 'names' format in {yaml_path}")

    return {
        "yaml_path": str(yaml_path),
        "nc": int(raw.get("nc", len(names))),
        "names": names,
        "raw": raw,
    }


def parse_yolo_label_file(
    label_path: Path | None,
) -> list[tuple[int, float, float, float, float]]:
    """Parse well-formed YOLO bbox rows from a label file."""
    if label_path is None or not label_path.exists():
        return []

    parsed: list[tuple[int, float, float, float, float]] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return parsed

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(val) for val in parts[1:])
        except ValueError:
            continue
        parsed.append((class_id, cx, cy, w, h))
    return parsed


def get_split_directories(
    root_dir: Path, layout: str
) -> list[tuple[str, Path, Path]]:
    """Return (split_name, images_dir, labels_dir) tuples for a dataset layout."""
    if layout == "roboflow":
        candidates = ["train", "valid", "test"]
        return [
            (split, root_dir / split / "images", root_dir / split / "labels")
            for split in candidates
            if (root_dir / split / "images").exists()
        ]
    if layout == "ultralytics":
        candidates = ["train", "val", "test"]
        return [
            (split, root_dir / "images" / split, root_dir / "labels" / split)
            for split in candidates
            if (root_dir / "images" / split).exists()
        ]
    raise ValueError(f"Unsupported dataset layout: {layout}")


def discover_dataset_records(
    source_name: str, root_dir: Path, layout: str
) -> list[DatasetRecord]:
    """Discover all images, paired label files, and provenance hashes."""
    records: list[DatasetRecord] = []
    for split, images_dir, labels_dir in get_split_directories(root_dir, layout):
        image_files = sorted(
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img_path in image_files:
            candidate_label = labels_dir / f"{img_path.stem}.txt"
            label_path = candidate_label if candidate_label.exists() else None
            records.append(
                DatasetRecord(
                    source=source_name,
                    layout=layout,
                    split=split,
                    image_path=img_path,
                    label_path=label_path,
                    sha256=compute_sha256(img_path),
                    annotations=parse_yolo_label_file(label_path),
                )
            )
    return records
