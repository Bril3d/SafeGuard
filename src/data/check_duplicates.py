"""Exact SHA-256, perceptual-hash, and scene-group deduplication for SafeGuard."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import imagehash
from PIL import Image, UnidentifiedImageError
import yaml

from src.data.inspect_dataset import DatasetRecord

_VIDEO_STEM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(IMG_\d+_(?:mp4|MOV))-\d+", re.IGNORECASE),
    re.compile(r"^(Movie-on-[^_]+-at-[^_]+_mov)-\d+", re.IGNORECASE),
    re.compile(r"^(autox4_mp4)-\d+", re.IGNORECASE),
    re.compile(r"^(construction-\d+-_mp4)-\d+", re.IGNORECASE),
)


@dataclass(frozen=True)
class DuplicateGroup:
    """Group of exact or near-duplicate images across sources/splits."""

    group_id: int
    match_type: str
    min_hamming: int
    cross_source: bool
    cross_split: bool
    records: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DuplicateReport:
    """Audit report of exact and near-duplicate image clusters."""

    total_images: int
    phash_threshold: int
    exact_duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    near_duplicate_groups: list[DuplicateGroup] = field(default_factory=list)


def _build_group(
    group_id: int,
    match_type: str,
    min_hamming: int,
    items: list[DatasetRecord],
) -> DuplicateGroup:
    sources = {r.source for r in items}
    splits = {(r.source, r.split) for r in items}
    return DuplicateGroup(
        group_id=group_id,
        match_type=match_type,
        min_hamming=min_hamming,
        cross_source=len(sources) > 1,
        cross_split=len(splits) > 1,
        records=[
            {
                "source": r.source,
                "split": r.split,
                "image_path": str(r.image_path),
                "sha256": r.sha256,
            }
            for r in items
        ],
    )


def find_duplicates(
    records: list[DatasetRecord],
    phash_threshold: int = 6,
) -> DuplicateReport:
    """Detect exact SHA-256 duplicates and perceptual-hash near-duplicates."""
    report = DuplicateReport(
        total_images=len(records),
        phash_threshold=phash_threshold,
    )

    by_sha: dict[str, list[DatasetRecord]] = defaultdict(list)
    for rec in records:
        by_sha[rec.sha256].append(rec)

    representatives: list[DatasetRecord] = []
    exact_id = 1
    for _, group in sorted(by_sha.items()):
        representatives.append(group[0])
        if len(group) > 1:
            report.exact_duplicate_groups.append(
                _build_group(exact_id, "exact_sha256", 0, group)
            )
            exact_id += 1

    phashes: list[tuple[DatasetRecord, imagehash.ImageHash]] = []
    for rec in representatives:
        try:
            with Image.open(rec.image_path) as img:
                phashes.append((rec, imagehash.phash(img.convert("RGB"))))
        except (UnidentifiedImageError, OSError):
            continue

    n = len(phashes)
    parent = list(range(n))
    min_dist_for_node: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        _, hi = phashes[i]
        for j in range(i + 1, n):
            dist = int(hi - phashes[j][1])
            if dist <= phash_threshold:
                union(i, j)
                min_dist_for_node[i] = min(min_dist_for_node.get(i, dist), dist)
                min_dist_for_node[j] = min(min_dist_for_node.get(j, dist), dist)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    near_id = 1
    for indices in clusters.values():
        if len(indices) > 1:
            cluster_records = [phashes[idx][0] for idx in indices]
            min_h = min(min_dist_for_node[idx] for idx in indices)
            report.near_duplicate_groups.append(
                _build_group(near_id, "near_phash", min_h, cluster_records)
            )
            near_id += 1

    return report


def _extract_scene_prefix(source: str, orig_path: str) -> str | None:
    stem = Path(orig_path).stem
    for pat in _VIDEO_STEM_PATTERNS:
        m = pat.match(stem)
        if m:
            return f"{source}:{m.group(1).lower()}"
    return None


def _split_priority(split: str) -> int:
    if split == "test":
        return 3
    if split in {"val", "valid"}:
        return 2
    return 1


def deduplicate_from_manifest(
    harmonize_manifest_path: Path,
    params_path: Path,
    output_manifest_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Remove duplicate/near-duplicate leakage while preserving locked test splits."""
    harmonize = json.loads(harmonize_manifest_path.read_text(encoding="utf-8"))
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle).get("data_pipeline", {})

    phash_threshold = int(params.get("phash_threshold", 6))
    enable_scene = bool(params.get("enable_scene_grouping", True))
    records: list[dict[str, Any]] = harmonize["records"]

    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1. Exact SHA-256 clustering
    sha_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        sha_to_indices[rec["sha256"]].append(idx)
    for idxs in sha_to_indices.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # 2. Perceptual hash clustering
    phashes: dict[int, imagehash.ImageHash] = {}
    for idx, rec in enumerate(records):
        with Image.open(rec["harmonized_image_path"]) as img:
            phashes[idx] = imagehash.phash(img.convert("RGB"))

    rep_indices = [idxs[0] for idxs in sha_to_indices.values()]
    for i_pos, idx_i in enumerate(rep_indices):
        for idx_j in rep_indices[i_pos + 1 :]:
            if int(phashes[idx_i] - phashes[idx_j]) <= phash_threshold:
                union(idx_i, idx_j)

    # 3. Assign scene_group_id (pHash clusters + video-prefix groups)
    dup_clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        dup_clusters[find(idx)].append(idx)

    scene_parent = list(range(n))

    def sfind(x: int) -> int:
        while scene_parent[x] != x:
            scene_parent[x] = scene_parent[scene_parent[x]]
            x = scene_parent[x]
        return x

    def sunion(a: int, b: int) -> None:
        ra, rb = sfind(a), sfind(b)
        if ra != rb:
            scene_parent[rb] = ra

    for idxs in dup_clusters.values():
        for j in idxs[1:]:
            sunion(idxs[0], j)

    if enable_scene:
        by_prefix: dict[str, list[int]] = defaultdict(list)
        for idx, rec in enumerate(records):
            pref = _extract_scene_prefix(rec["source"], rec["original_image_path"])
            if pref:
                by_prefix[pref].append(idx)
        for idxs in by_prefix.values():
            for j in idxs[1:]:
                sunion(idxs[0], j)

    # 4. Decide keep/drop per duplicate cluster (locked test > val/valid > train)
    decisions: dict[int, tuple[str, str]] = {}
    dropped_log: list[dict[str, Any]] = []

    for cluster_id, idxs in dup_clusters.items():
        if len(idxs) == 1:
            idx = idxs[0]
            split = records[idx]["original_split"]
            decisions[idx] = (
                "keep_locked_test" if split == "test" else "keep_unique",
                "no_exact_or_near_duplicate",
            )
            continue

        has_test = any(records[i]["original_split"] == "test" for i in idxs)
        has_val = any(records[i]["original_split"] in {"val", "valid"} for i in idxs)

        sorted_idxs = sorted(
            idxs,
            key=lambda i: (
                -_split_priority(records[i]["original_split"]),
                -int(records[i]["harmonized_box_count"]),
                records[i]["sha256"],
                records[i]["unique_id"],
            ),
        )
        keeper = sorted_idxs[0]

        for idx in idxs:
            split = records[idx]["original_split"]
            if split == "test":
                decisions[idx] = ("keep_locked_test", "locked_test_split_preserved")
            elif idx == keeper:
                decisions[idx] = (
                    "keep_cluster_representative",
                    f"representative_for_cluster_{cluster_id}",
                )
            elif has_test:
                decisions[idx] = (
                    "drop_test_leakage",
                    f"duplicate_of_locked_test_in_cluster_{cluster_id}",
                )
            elif has_val and split == "train":
                decisions[idx] = (
                    "drop_val_leakage",
                    f"duplicate_of_validation_image_in_cluster_{cluster_id}",
                )
            else:
                decisions[idx] = (
                    "drop_within_split_duplicate",
                    f"redundant_near_duplicate_of_{records[keeper]['unique_id']}",
                )

            if decisions[idx][0].startswith("drop"):
                dropped_log.append(
                    {
                        "unique_id": records[idx]["unique_id"],
                        "source": records[idx]["source"],
                        "original_split": split,
                        "original_image_path": records[idx]["original_image_path"],
                        "decision": decisions[idx][0],
                        "reason": decisions[idx][1],
                        "kept_representative": records[keeper]["unique_id"],
                    }
                )

    out_records: list[dict[str, Any]] = []
    test_scene_groups = {
        f"sg_{sfind(idx):04d}"
        for idx, rec in enumerate(records)
        if rec["original_split"] == "test"
    }

    for idx, rec in enumerate(records):
        decision, reason = decisions[idx]
        sg_id = f"sg_{sfind(idx):04d}"
        if (
            decision.startswith("keep")
            and rec["original_split"] == "train"
            and sg_id in test_scene_groups
        ):
            decision = "drop_scene_group_test_leakage"
            reason = f"shares_video_scene_group_{sg_id}_with_locked_test"
            dropped_log.append(
                {
                    "unique_id": rec["unique_id"],
                    "source": rec["source"],
                    "original_split": rec["original_split"],
                    "original_image_path": rec["original_image_path"],
                    "decision": decision,
                    "reason": reason,
                    "kept_representative": sg_id,
                }
            )
        rec_copy = dict(rec)
        rec_copy["duplicate_cluster_id"] = int(find(idx))
        rec_copy["scene_group_id"] = sg_id
        rec_copy["decision"] = decision
        rec_copy["decision_reason"] = reason
        out_records.append(rec_copy)

    kept_count = sum(1 for r in out_records if r["decision"].startswith("keep"))
    manifest = {
        "stage": "deduplicate",
        "phash_threshold": phash_threshold,
        "enable_scene_grouping": enable_scene,
        "total_input_images": len(out_records),
        "total_kept": kept_count,
        "total_dropped": len(out_records) - kept_count,
        "records": out_records,
    }
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "total_input_images": len(out_records),
        "total_kept": kept_count,
        "total_dropped": len(dropped_log),
        "dropped_by_decision": {
            dec: sum(1 for d in dropped_log if d["decision"] == dec)
            for dec in sorted({d["decision"] for d in dropped_log})
        },
        "dropped_records": dropped_log,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """CLI entrypoint for DVC deduplicate stage."""
    parser = argparse.ArgumentParser(description="SafeGuard DVC deduplicate stage.")
    parser.add_argument(
        "--harmonize-manifest",
        type=Path,
        default=Path("data/interim/harmonize_manifest.json"),
    )
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/interim/dedup_manifest.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/dataset/deduplication_report.json"),
    )
    args = parser.parse_args()
    deduplicate_from_manifest(
        args.harmonize_manifest, args.params, args.output_manifest, args.report
    )


if __name__ == "__main__":
    main()
