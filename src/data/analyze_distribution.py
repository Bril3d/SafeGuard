"""Class distribution and bounding-box geometry analysis for SafeGuard."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.inspect_dataset import DatasetRecord  # noqa: E402


def compute_class_distribution(
    records: list[DatasetRecord],
    source_class_names: dict[str, dict[int, str]],
) -> pd.DataFrame:
    """Compute per-source, per-split, per-class instance and image counts."""
    instance_counts: dict[tuple[str, str, int], int] = defaultdict(int)
    image_sets: dict[tuple[str, str, int], set[Path]] = defaultdict(set)
    split_totals: dict[tuple[str, str], int] = defaultdict(int)

    for rec in records:
        for class_id, *_ in rec.annotations:
            key = (rec.source, rec.split, class_id)
            instance_counts[key] += 1
            image_sets[key].add(rec.image_path)
            split_totals[(rec.source, rec.split)] += 1

    rows: list[dict[str, object]] = []
    for (source, split, class_id), inst_count in sorted(instance_counts.items()):
        class_name = source_class_names.get(source, {}).get(
            class_id, f"unknown_{class_id}"
        )
        total_in_split = split_totals[(source, split)]
        rows.append(
            {
                "source": source,
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": inst_count,
                "image_count": len(image_sets[(source, split, class_id)]),
                "pct_of_split_instances": round(
                    100.0 * inst_count / total_in_split if total_in_split else 0.0, 2
                ),
            }
        )
    return pd.DataFrame(rows)


def export_distribution_csv(dist_df: pd.DataFrame, output_path: Path) -> Path:
    """Write class distribution DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dist_df.to_csv(output_path, index=False)
    return output_path


def compute_bbox_metrics(
    records: list[DatasetRecord],
    source_class_names: dict[str, dict[int, str]],
) -> pd.DataFrame:
    """Compute normalized bounding-box dimensions, area, and aspect ratio."""
    rows: list[dict[str, object]] = []
    for rec in records:
        for class_id, cx, cy, w, h in rec.annotations:
            if w <= 0 or h <= 0:
                continue
            class_name = source_class_names.get(source := rec.source, {}).get(
                class_id, f"unknown_{class_id}"
            )
            rows.append(
                {
                    "source": source,
                    "split": rec.split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "cx": cx,
                    "cy": cy,
                    "width": w,
                    "height": h,
                    "area": w * h,
                    "aspect_ratio": w / h,
                }
            )
    return pd.DataFrame(rows)


def plot_class_distribution(dist_df: pd.DataFrame, output_path: Path) -> Path:
    """Generate a bar chart comparing class instance counts per source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted(dist_df["source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 6))
    if len(sources) == 1:
        axes = [axes]

    for ax, source in zip(axes, sources):
        sub = (
            dist_df[dist_df["source"] == source]
            .groupby("class_name", as_index=False)["instance_count"]
            .sum()
            .sort_values("instance_count", ascending=True)
        )
        ax.barh(sub["class_name"], sub["instance_count"], color="#2563eb")
        ax.set_title(f"{source} — Instances per Class")
        ax.set_xlabel("Instance Count")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_bbox_analysis(
    bbox_df: pd.DataFrame,
    output_path: Path,
    candidate_names: dict[str, set[str]] | None = None,
) -> Path:
    """Plot bounding-box normalized area and aspect-ratio distributions by source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_names:
        mask = bbox_df.apply(
            lambda r: r["class_name"] in candidate_names.get(r["source"], set()),
            axis=1,
        )
        plot_df = bbox_df[mask].copy()
    else:
        plot_df = bbox_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for source, group in plot_df.groupby("source"):
        clipped_area = group["area"].clip(upper=0.5)
        clipped_ar = group["aspect_ratio"].clip(upper=4.0)
        axes[0].hist(clipped_area, bins=35, alpha=0.55, label=source, density=True)
        axes[1].hist(clipped_ar, bins=35, alpha=0.55, label=source, density=True)

    axes[0].set_title("Normalized BBox Area Distribution (clipped at 0.5)")
    axes[0].set_xlabel("Normalized Area (w * h)")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("BBox Aspect Ratio Distribution (w / h, clipped at 4.0)")
    axes[1].set_xlabel("Aspect Ratio (w / h)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
