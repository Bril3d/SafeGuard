"""Lightweight pure-Python ByteTrack-style anonymous person tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for two-stage ByteTrack IoU multi-object tracking."""

    high_conf_threshold: float = 0.45
    low_conf_threshold: float = 0.15
    match_iou_threshold: float = 0.25
    second_stage_iou_threshold: float = 0.15
    max_age: int = 15
    min_hits: int = 1
    velocity_alpha: float = 0.4
    velocity_damping: float = 0.75


@dataclass(frozen=True)
class TrackDetection:
    """Generic detector-agnostic bounding box input to the tracker."""

    bbox_xyxy: list[float]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class Track:
    """Public anonymous tracked person state (contains zero PII or pixels)."""

    track_id: int
    bbox_xyxy: list[float]
    confidence: float
    age: int
    hits: int
    time_since_update: int
    state: str


@dataclass
class _InternalTrackState:
    """Internal kinematic state using center-size coordinates and velocity."""

    track_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    vx: float = 0.0
    vy: float = 0.0
    vw: float = 0.0
    vh: float = 0.0
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    velocity_alpha: float = 0.4
    velocity_damping: float = 0.75

    @classmethod
    def from_detection(
        cls, track_id: int, det: TrackDetection, cfg: TrackerConfig
    ) -> _InternalTrackState:
        x1, y1, x2, y2 = det.bbox_xyxy
        return cls(
            track_id=track_id,
            cx=(x1 + x2) * 0.5,
            cy=(y1 + y2) * 0.5,
            w=max(1.0, x2 - x1),
            h=max(1.0, y2 - y1),
            confidence=det.confidence,
            velocity_alpha=cfg.velocity_alpha,
            velocity_damping=cfg.velocity_damping,
        )

    def predict(self) -> None:
        self.cx += self.vx
        self.cy += self.vy
        self.w = max(1.0, self.w + self.vw)
        self.h = max(1.0, self.h + self.vh)
        self.vx *= self.velocity_damping
        self.vy *= self.velocity_damping
        self.vw *= self.velocity_damping
        self.vh *= self.velocity_damping
        self.age += 1
        self.time_since_update += 1

    def update(self, det: TrackDetection) -> None:
        x1, y1, x2, y2 = det.bbox_xyxy
        ncx = (x1 + x2) * 0.5
        ncy = (y1 + y2) * 0.5
        nw = max(1.0, x2 - x1)
        nh = max(1.0, y2 - y1)

        a = self.velocity_alpha
        self.vx = a * (ncx - self.cx) + (1.0 - a) * self.vx
        self.vy = a * (ncy - self.cy) + (1.0 - a) * self.vy
        self.vw = a * (nw - self.w) + (1.0 - a) * self.vw
        self.vh = a * (nh - self.h) + (1.0 - a) * self.vh

        self.cx = ncx
        self.cy = ncy
        self.w = nw
        self.h = nh
        self.confidence = det.confidence
        self.hits += 1
        self.time_since_update = 0

    def to_xyxy(self) -> list[float]:
        return [
            round(self.cx - self.w * 0.5, 2),
            round(self.cy - self.h * 0.5, 2),
            round(self.cx + self.w * 0.5, 2),
            round(self.cy + self.h * 0.5, 2),
        ]

    def to_public(self, min_hits: int) -> Track:
        state_label = (
            "active"
            if self.time_since_update == 0 and self.hits >= min_hits
            else "lost"
        )
        return Track(
            track_id=self.track_id,
            bbox_xyxy=self.to_xyxy(),
            confidence=round(float(self.confidence), 6),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            state=state_label,
        )


def _compute_iou_matrix(
    tracks: list[_InternalTrackState],
    detections: list[TrackDetection],
) -> np.ndarray:
    if not tracks or not detections:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)

    t_boxes = np.asarray([t.to_xyxy() for t in tracks], dtype=np.float32)
    d_boxes = np.asarray([d.bbox_xyxy for d in detections], dtype=np.float32)

    tx1, ty1, tx2, ty2 = (
        t_boxes[:, 0:1],
        t_boxes[:, 1:2],
        t_boxes[:, 2:3],
        t_boxes[:, 3:4],
    )
    dx1, dy1, dx2, dy2 = (
        d_boxes[:, 0][None, :],
        d_boxes[:, 1][None, :],
        d_boxes[:, 2][None, :],
        d_boxes[:, 3][None, :],
    )

    ix1 = np.maximum(tx1, dx1)
    iy1 = np.maximum(ty1, dy1)
    ix2 = np.minimum(tx2, dx2)
    iy2 = np.minimum(ty2, dy2)

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    t_area = np.maximum(0.0, tx2 - tx1) * np.maximum(0.0, ty2 - ty1)
    d_area = np.maximum(0.0, dx2 - dx1) * np.maximum(0.0, dy2 - dy1)
    union = t_area + d_area - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _greedy_bipartite_match(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    num_tracks, num_dets = iou_matrix.shape
    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))

    candidates: list[tuple[float, int, int]] = []
    for t_idx in range(num_tracks):
        for d_idx in range(num_dets):
            val = float(iou_matrix[t_idx, d_idx])
            if val >= iou_threshold:
                candidates.append((val, t_idx, d_idx))
    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, t_idx, d_idx in candidates:
        if t_idx not in matched_tracks and d_idx not in matched_dets:
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)
            matches.append((t_idx, d_idx))

    unmatched_t = [i for i in range(num_tracks) if i not in matched_tracks]
    unmatched_d = [j for j in range(num_dets) if j not in matched_dets]
    return matches, unmatched_t, unmatched_d


def _is_valid_detection(det: TrackDetection) -> bool:
    if len(det.bbox_xyxy) != 4 or not math.isfinite(det.confidence):
        return False
    if not (0.0 <= det.confidence <= 1.0):
        return False
    x1, y1, x2, y2 = det.bbox_xyxy
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    return x2 > x1 and y2 > y1


@dataclass
class ByteTrackTracker:
    """Anonymous frame-to-frame two-stage IoU tracker following ByteTrack."""

    config: TrackerConfig = field(default_factory=TrackerConfig)
    _tracks: list[_InternalTrackState] = field(default_factory=list, init=False)
    _next_id: int = field(default=1, init=False)
    _frame_index: int = field(default=0, init=False)

    @property
    def active_track_count(self) -> int:
        """Number of currently retained tracks (active + within max_age)."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all active and lost tracks for a stream."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    def update(self, detections: list[TrackDetection]) -> list[Track]:
        """Advance tracker one frame and return matched active anonymous tracks."""
        self._frame_index += 1
        for trk in self._tracks:
            trk.predict()

        valid_dets = [d for d in detections if _is_valid_detection(d)]
        high_dets = [
            d for d in valid_dets if d.confidence >= self.config.high_conf_threshold
        ]
        low_dets = [
            d
            for d in valid_dets
            if self.config.low_conf_threshold
            <= d.confidence
            < self.config.high_conf_threshold
        ]

        # Stage 1: match tracks against high-confidence detections.
        iou_high = _compute_iou_matrix(self._tracks, high_dets)
        matches_1, unmatched_t1, unmatched_d1 = _greedy_bipartite_match(
            iou_high, self.config.match_iou_threshold
        )
        for t_idx, d_idx in matches_1:
            self._tracks[t_idx].update(high_dets[d_idx])

        # Stage 2 (ByteTrack): recover occluded tracks with low-confidence detections.
        remaining_tracks = [self._tracks[i] for i in unmatched_t1]
        iou_low = _compute_iou_matrix(remaining_tracks, low_dets)
        matches_2, _, _ = _greedy_bipartite_match(
            iou_low, self.config.second_stage_iou_threshold
        )
        for rem_idx, d_idx in matches_2:
            remaining_tracks[rem_idx].update(low_dets[d_idx])

        # Initialize new tracks only from unmatched high-confidence detections.
        for d_idx in unmatched_d1:
            new_trk = _InternalTrackState.from_detection(
                self._next_id, high_dets[d_idx], self.config
            )
            self._next_id += 1
            self._tracks.append(new_trk)

        # Expire tracks beyond max_age occlusion budget.
        self._tracks = [
            trk for trk in self._tracks if trk.time_since_update <= self.config.max_age
        ]

        return [
            trk.to_public(self.config.min_hits)
            for trk in self._tracks
            if trk.time_since_update == 0 and trk.hits >= self.config.min_hits
        ]
