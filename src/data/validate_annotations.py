"""Annotation, image integrity, and bounding-box validation for SafeGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.data.inspect_dataset import DatasetRecord, get_split_directories


@dataclass
class ValidationReport:
    """Structured validation summary for a single source dataset."""

    source: str
    total_images: int
    total_label_files: int
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    corrupted_images: list[str] = field(default_factory=list)
    empty_label_files: list[str] = field(default_factory=list)
    valid_box_count: int = 0
    invalid_box_count: int = 0
    invalid_boxes: list[dict[str, Any]] = field(default_factory=list)
    image_dimensions: dict[str, tuple[int, int]] = field(default_factory=dict)


def _check_image_integrity(image_path: Path) -> tuple[bool, tuple[int, int] | None]:
    """Verify image readability and return pixel dimensions."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            width, height = img.size
        if width <= 0 or height <= 0:
            return False, None
        return True, (width, height)
    except (UnidentifiedImageError, OSError, ValueError):
        return False, None


def _validate_yolo_line(
    line: str,
    valid_class_ids: set[int],
    eps: float = 1e-4,
) -> str | None:
    """Validate a single YOLO annotation row; return error reason or None."""
    parts = line.strip().split()
    if len(parts) != 5:
        return f"expected 5 columns, got {len(parts)}"

    try:
        class_id = int(parts[0])
    except ValueError:
        return f"non-integer class_id '{parts[0]}'"

    if class_id not in valid_class_ids:
        return f"unknown class_id {class_id}"

    try:
        cx, cy, w, h = (float(val) for val in parts[1:])
    except ValueError:
        return "non-numeric coordinates"

    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return "non-finite coordinate"
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return f"center out of [0,1]: ({cx}, {cy})"
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return f"width/height out of (0,1]: ({w}, {h})"

    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
        return f"box extends outside [0,1]: [{x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}]"

    return None


def validate_dataset(
    source_name: str,
    root_dir: Path,
    layout: str,
    records: list[DatasetRecord],
    valid_class_ids: set[int],
) -> ValidationReport:
    """Validate image/label pairs, image files, and YOLO bounding boxes."""
    report = ValidationReport(
        source=source_name,
        total_images=len(records),
        total_label_files=0,
    )

    image_stems_by_split: dict[str, set[str]] = {}
    for rec in records:
        image_stems_by_split.setdefault(rec.split, set()).add(rec.image_path.stem)

    for split, _, labels_dir in get_split_directories(root_dir, layout):
        if not labels_dir.exists():
            continue
        label_files = sorted(
            p for p in labels_dir.iterdir() if p.is_file() and p.suffix == ".txt"
        )
        report.total_label_files += len(label_files)
        known_stems = image_stems_by_split.get(split, set())
        for lbl_path in label_files:
            if lbl_path.stem not in known_stems:
                report.orphan_labels.append(str(lbl_path))

    for rec in records:
        ok, dims = _check_image_integrity(rec.image_path)
        if not ok or dims is None:
            report.corrupted_images.append(str(rec.image_path))
        else:
            report.image_dimensions[str(rec.image_path)] = dims

        if rec.label_path is None:
            report.missing_labels.append(str(rec.image_path))
            continue

        content = rec.label_path.read_text(encoding="utf-8").strip()
        if not content:
            report.empty_label_files.append(str(rec.label_path))
            continue

        for line_no, line in enumerate(content.splitlines(), start=1):
            reason = _validate_yolo_line(line, valid_class_ids)
            if reason is not None:
                report.invalid_box_count += 1
                report.invalid_boxes.append(
                    {
                        "label_path": str(rec.label_path),
                        "line": line_no,
                        "reason": reason,
                        "raw": line.strip(),
                    }
                )
            else:
                report.valid_box_count += 1

    return report
