"""Exact SHA-256 and perceptual-hash duplicate detection for SafeGuard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import imagehash
from PIL import Image, UnidentifiedImageError

from src.data.inspect_dataset import DatasetRecord


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
    for sha, group in sorted(by_sha.items()):
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
