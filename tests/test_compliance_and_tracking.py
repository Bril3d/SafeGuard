"""TDD unit and integration tests for compliance rules and anonymous tracking."""

from __future__ import annotations

import pytest

from src.compliance.rules import (
    ComplianceConfig,
    ComplianceEngine,
    DetectionInput,
)
from src.tracking.tracker import ByteTrackTracker, TrackDetection, TrackerConfig


def _det(
    class_id: int,
    class_name: str,
    conf: float,
    bbox: list[float],
) -> DetectionInput:
    return DetectionInput(
        class_id=class_id,
        class_name=class_name,
        confidence=conf,
        bbox_xyxy=bbox,
    )


def test_correct_ppe_association_marks_person_compliant() -> None:
    tracker = ByteTrackTracker(TrackerConfig(max_age=5))
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=2, cooldown_seconds=10.0), tracker=tracker
    )

    dets = [
        _det(0, "person", 0.92, [100.0, 50.0, 220.0, 350.0]),
        _det(1, "helmet", 0.90, [130.0, 48.0, 190.0, 105.0]),
        _det(2, "vest", 0.88, [115.0, 130.0, 205.0, 240.0]),
        _det(3, "gloves", 0.78, [98.0, 215.0, 125.0, 250.0]),
    ]
    res = engine.process_frame(dets, timestamp=1.0)
    assert len(res.associations) == 1
    assoc = res.associations[0]
    assert assoc.helmet_status == "compliant"
    assert assoc.vest_status == "compliant"
    assert assoc.overall_status == "compliant"
    assert len(assoc.helmet_boxes) == 1
    assert len(assoc.vest_boxes) == 1
    assert len(assoc.gloves_boxes) == 1
    assert res.events == []


def test_explicit_no_helmet_triggers_violation_after_temporal_confirmation() -> None:
    tracker = ByteTrackTracker(TrackerConfig(max_age=5))
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=3, cooldown_seconds=10.0), tracker=tracker
    )

    dets = [
        _det(0, "person", 0.90, [100.0, 50.0, 220.0, 350.0]),
        _det(4, "no_helmet", 0.85, [135.0, 52.0, 185.0, 105.0]),
        _det(2, "vest", 0.84, [115.0, 130.0, 205.0, 240.0]),
    ]

    # Frames 1 & 2: below confirm_frames=3 -> pending, no emitted alert event yet.
    r1 = engine.process_frame(dets, timestamp=0.0)
    assert r1.associations[0].helmet_status == "violation"
    assert r1.associations[0].overall_status == "pending_confirmation"
    assert r1.events == []

    r2 = engine.process_frame(dets, timestamp=0.5)
    assert r2.events == []

    # Frame 3: confirmed! Emits potential_no_helmet event requiring human review.
    r3 = engine.process_frame(dets, timestamp=1.0)
    assert r3.associations[0].overall_status == "potential_violation"
    assert len(r3.events) == 1
    evt = r3.events[0]
    assert evt.event_type == "potential_no_helmet"
    assert evt.requires_human_review is True
    assert evt.track_id == r1.associations[0].track_id
    assert evt.confirmed_frames == 3


def test_missing_vest_is_unknown_and_never_triggers_violation() -> None:
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(ComplianceConfig(confirm_frames=1), tracker=tracker)

    dets = [
        _det(0, "person", 0.91, [100.0, 50.0, 220.0, 350.0]),
        _det(1, "helmet", 0.89, [135.0, 50.0, 185.0, 100.0]),
    ]
    res = engine.process_frame(dets, timestamp=1.0)
    assoc = res.associations[0]
    assert assoc.helmet_status == "compliant"
    assert assoc.vest_status == "unknown"
    assert assoc.overall_status == "unknown"
    assert "vest_not_detected_no_vest_class_untrained" in assoc.uncertainty_reasons
    assert res.events == []


def test_unknown_and_ambiguous_helmet_cases_do_not_emit_false_violations() -> None:
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(ComplianceConfig(confirm_frames=1), tracker=tracker)

    # Case A: Both helmet and no_helmet with close confidences -> ambiguous.
    ambiguous_dets = [
        _det(0, "person", 0.90, [100.0, 50.0, 220.0, 350.0]),
        _det(1, "helmet", 0.62, [135.0, 50.0, 185.0, 100.0]),
        _det(4, "no_helmet", 0.58, [136.0, 52.0, 184.0, 102.0]),
        _det(2, "vest", 0.85, [115.0, 130.0, 205.0, 240.0]),
    ]
    res_a = engine.process_frame(ambiguous_dets, timestamp=1.0)
    assert res_a.associations[0].helmet_status == "ambiguous"
    assert res_a.associations[0].overall_status == "unknown"
    assert res_a.events == []

    # Case B: Neither helmet nor no_helmet detected -> unknown (never violation).
    neither_dets = [
        _det(0, "person", 0.90, [400.0, 50.0, 520.0, 350.0]),
        _det(2, "vest", 0.85, [415.0, 130.0, 505.0, 240.0]),
    ]
    res_b = engine.process_frame(neither_dets, timestamp=2.0)
    second_assoc = [a for a in res_b.associations if a.bbox_xyxy[0] > 300][0]
    assert second_assoc.helmet_status == "unknown"
    assert second_assoc.overall_status == "unknown"
    assert res_b.events == []


def test_tracker_id_persistence_and_temporary_occlusion_recovery() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(high_conf_threshold=0.5, low_conf_threshold=0.15, max_age=4)
    )

    # Frame 1 & 2: moving person detected at high confidence.
    t1 = tracker.update([TrackDetection([100.0, 50.0, 200.0, 300.0], 0.90, 0)])
    t2 = tracker.update([TrackDetection([110.0, 50.0, 210.0, 300.0], 0.88, 0)])
    assert len(t1) == 1 and len(t2) == 1
    assert t1[0].track_id == t2[0].track_id

    # Frame 3: complete occlusion (0 detections), track remains retained internally.
    t3 = tracker.update([])
    assert t3 == []
    assert tracker.active_track_count == 1

    # Frame 4: low-conf reappearance -> ByteTrack stage-2 recovers same ID.
    t4 = tracker.update([TrackDetection([122.0, 50.0, 222.0, 300.0], 0.28, 0)])
    assert len(t4) == 1
    assert t4[0].track_id == t1[0].track_id

    # Exceeding max_age retires the lost track and assigns a new ID later.
    for _ in range(6):
        tracker.update([])
    assert tracker.active_track_count == 0
    t_new = tracker.update([TrackDetection([122.0, 50.0, 222.0, 300.0], 0.90, 0)])
    assert t_new[0].track_id != t1[0].track_id


def test_violation_deduplication_and_cooldown() -> None:
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=2, cooldown_seconds=5.0), tracker=tracker
    )
    dets = [
        _det(0, "person", 0.90, [100.0, 50.0, 220.0, 350.0]),
        _det(4, "no_helmet", 0.85, [135.0, 52.0, 185.0, 105.0]),
    ]

    r1 = engine.process_frame(dets, timestamp=10.0)
    assert r1.events == []
    r2 = engine.process_frame(dets, timestamp=11.0)
    assert len(r2.events) == 1

    # Within cooldown (< 5.0s from t=11.0): suppressed duplicate.
    r3 = engine.process_frame(dets, timestamp=13.0)
    assert r3.events == []
    assert r3.suppressed_duplicates == 1

    # After cooldown expires (>= 5.0s): allowed to emit again.
    r4 = engine.process_frame(dets, timestamp=16.5)
    assert len(r4.events) == 1
    assert r4.suppressed_duplicates == 0


def test_empty_frames_and_invalid_detections_handled_safely() -> None:
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(ComplianceConfig(), tracker=tracker)

    empty_res = engine.process_frame([], timestamp=1.0)
    assert empty_res.associations == []
    assert empty_res.events == []
    assert empty_res.invalid_detections_filtered == 0

    bad_dets = [
        _det(0, "person", float("nan"), [10.0, 10.0, 100.0, 200.0]),
        _det(0, "person", 0.9, [100.0, 10.0, 50.0, 200.0]),  # inverted x2 < x1
        _det(1, "helmet", 1.5, [20.0, 20.0, 40.0, 40.0]),  # invalid conf > 1.0
        _det(4, "no_helmet", 0.8, [10.0, 10.0, 30.0, 30.0]),  # valid unmatched PPE
    ]
    bad_res = engine.process_frame(bad_dets, timestamp=2.0)
    assert bad_res.invalid_detections_filtered == 3
    assert len(bad_res.unmatched_ppe) == 1
    assert bad_res.events == []
    # Confirm privacy guarantee: no image arrays or biometrics in track snapshots.
    assert "image" not in str(bad_res).lower()
    assert pytest.approx(2.0) == bad_res.timestamp


def test_unapproved_classes_like_no_vest_rejected_by_canonical_taxonomy() -> None:
    from src.compliance.rules import ComplianceInterpreter, load_approved_class_names

    approved = load_approved_class_names()
    assert approved == frozenset({"person", "helmet", "vest", "gloves", "no_helmet"})

    tracker = ByteTrackTracker()
    interpreter = ComplianceInterpreter(ComplianceConfig(confirm_frames=1))
    dets = [
        _det(0, "person", 0.92, [100.0, 50.0, 220.0, 350.0]),
        _det(1, "helmet", 0.90, [130.0, 48.0, 190.0, 105.0]),
        _det(5, "no_vest", 0.94, [115.0, 130.0, 205.0, 240.0]),
        _det(99, "head", 0.91, [132.0, 50.0, 188.0, 102.0]),
    ]
    tracks = tracker.update(
        [
            TrackDetection(d.bbox_xyxy, d.confidence, d.class_id)
            for d in dets
            if d.class_name == "person"
        ]
    )
    interp = interpreter.evaluate(dets, tracks)
    assert interp.unapproved_detections_filtered == 2
    assert interp.associations[0].helmet_status == "compliant"
    assert interp.associations[0].vest_status == "unknown"
    assert interp.associations[0].overall_status == "unknown"


def test_overlapping_workers_distinct_helmets_vs_ambiguous_shared_ppe() -> None:
    # Two overlapping workers with distinct helmets -> 1-to-1 optimal match.
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=1, spatial_ambiguity_margin=0.08),
        tracker=tracker,
    )
    two_helmets = [
        _det(0, "person", 0.92, [100.0, 60.0, 220.0, 360.0]),
        _det(0, "person", 0.91, [175.0, 65.0, 295.0, 365.0]),
        _det(1, "helmet", 0.90, [115.0, 55.0, 165.0, 110.0]),
        _det(2, "vest", 0.88, [112.0, 140.0, 195.0, 245.0]),
        _det(1, "helmet", 0.89, [230.0, 60.0, 280.0, 115.0]),
        _det(2, "vest", 0.87, [198.0, 145.0, 282.0, 250.0]),
    ]
    res_ok = engine.process_frame(two_helmets, timestamp=1.0)
    assert len(res_ok.associations) == 2
    assert all(a.helmet_status == "compliant" for a in res_ok.associations)

    # Overlapping workers with one centered no_helmet -> must surface uncertain.
    engine.reset()
    ambiguous_overlap = [
        _det(0, "person", 0.92, [100.0, 60.0, 240.0, 360.0]),
        _det(0, "person", 0.91, [120.0, 60.0, 260.0, 360.0]),
        _det(4, "no_helmet", 0.88, [155.0, 62.0, 205.0, 118.0]),
    ]
    res_amb = engine.process_frame(ambiguous_overlap, timestamp=2.0)
    assert res_amb.events == []
    assert all(
        a.overall_status == "uncertain"
        and "ambiguous_spatial_overlap" in a.uncertainty_reasons
        for a in res_amb.associations
    )


def test_tracking_step_exposes_lost_and_retired_with_intermittent_gaps() -> None:
    tracker = ByteTrackTracker(
        TrackerConfig(high_conf_threshold=0.5, low_conf_threshold=0.15, max_age=2)
    )
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=2, cooldown_seconds=10.0), tracker=tracker
    )

    p_det = _det(0, "person", 0.91, [120.0, 60.0, 240.0, 360.0])
    nh_det = _det(4, "no_helmet", 0.84, [152.0, 62.0, 208.0, 118.0])

    # Frame 1: no_helmet seen (streak=1, pending)
    r1 = engine.process_frame([p_det, nh_det], timestamp=1.0)
    tid = r1.tracks[0].track_id
    assert r1.associations[0].violation_streak == 1

    # Frame 2: complete occlusion -> track appears in lost_tracks.
    step2 = tracker.step([])
    assert step2.active_tracks == []
    assert len(step2.lost_tracks) == 1
    assert step2.lost_tracks[0].track_id == tid
    assert step2.lost_tracks[0].state == "lost"

    # Frame 3: low-conf person returns -> recovered same ID, gap resets streak.
    r3 = engine.process_frame(
        [_det(0, "person", 0.28, [126.0, 60.0, 246.0, 360.0]), nh_det],
        timestamp=2.0,
    )
    assert r3.tracks[0].track_id == tid
    assert r3.events == []  # Did not confirm prematurely across occluded gap

    # Frame 4: consecutive confirmed no_helmet -> emits event
    r4 = engine.process_frame(
        [_det(0, "person", 0.90, [130.0, 60.0, 250.0, 360.0]), nh_det],
        timestamp=3.0,
    )
    assert len(r4.events) == 1

    # Drop for > max_age frames -> retired_track_ids reports retired anonymous ID
    tracker.step([])
    tracker.step([])
    step_retire = tracker.step([])
    assert tid in step_retire.retired_track_ids
    assert tracker.active_track_count == 0


def test_insufficient_visibility_on_tiny_or_edge_clipped_worker() -> None:
    tracker = ByteTrackTracker(TrackerConfig(high_conf_threshold=0.4))
    engine = ComplianceEngine(
        ComplianceConfig(
            confirm_frames=1,
            min_person_height_px=60.0,
            edge_clip_margin_px=4.0,
        ),
        tracker=tracker,
    )
    # Tiny distant box (height 35px) + top-edge clipped box -> insufficient visibility.
    tiny_and_clipped = [
        _det(0, "person", 0.88, [50.0, 100.0, 75.0, 135.0]),
        _det(0, "person", 0.89, [300.0, 1.0, 410.0, 220.0]),
        _det(4, "no_helmet", 0.82, [330.0, 2.0, 380.0, 35.0]),
    ]
    res = engine.process_frame(tiny_and_clipped, timestamp=1.0, frame_size=(640, 480))
    assert len(res.associations) == 2
    assert all(
        a.visibility_status == "insufficient_visibility" for a in res.associations
    )
    assert all(a.overall_status == "insufficient_visibility" for a in res.associations)
    assert res.events == []


def test_alert_manager_evidence_frame_status_and_stream_analytics() -> None:
    import cv2
    import numpy as np

    frame_bgr = np.full((480, 640, 3), 180, dtype=np.uint8)
    tracker = ByteTrackTracker()
    engine = ComplianceEngine(
        ComplianceConfig(confirm_frames=2, cooldown_seconds=8.0), tracker=tracker
    )
    dets = [
        _det(0, "person", 0.93, [120.0, 50.0, 240.0, 350.0]),
        _det(4, "no_helmet", 0.89, [155.0, 54.0, 205.0, 108.0]),
        _det(2, "vest", 0.86, [135.0, 130.0, 225.0, 240.0]),
    ]

    r1 = engine.process_frame(dets, timestamp=10.0, frame_bgr=frame_bgr)
    assert r1.events == []

    r2 = engine.process_frame(dets, timestamp=11.0, frame_bgr=frame_bgr)
    assert len(r2.events) == 1
    evt = r2.events[0]
    assert evt.violation_category == "no_helmet"
    assert evt.event_status == "pending_review"
    assert evt.evidence_frame_b64 is not None
    # Verify annotated evidence bytes decode as a real JPEG image.
    import base64

    jpg_bytes = base64.b64decode(evt.evidence_frame_b64)
    decoded = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[0] > 0 and decoded.shape[1] > 0

    # Status update & duplicate suppression check
    r3 = engine.process_frame(dets, timestamp=14.0, frame_bgr=frame_bgr)
    assert r3.suppressed_duplicates == 1
    updated = engine.update_alert_status(evt.event_id, "verified_potential_event")
    assert updated is not None and updated.event_status == "verified_potential_event"

    # Analytics verification
    analytics = engine.get_analytics()
    assert analytics.workers_observed == 1
    assert analytics.ppe_detections["no_helmet"] == 3
    assert analytics.ppe_detections["vest"] == 3
    assert analytics.potential_violations >= 1
    assert analytics.violation_categories["no_helmet"] == 1
    assert len(analytics.alerts_over_time) == 1
    assert analytics.alerts_over_time[0].event_status == "verified_potential_event"
