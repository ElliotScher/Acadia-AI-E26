import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.processing.video_entityprofiler import (
    VideoEntityRecord,
    calibrate_absolute_speeds,
    compute_relative_speeds,
    load_yolo_detections,
    process_video,
)
from utility import imgutils
from utility.geometryutils import Rectangle


def test_compute_iou():
    # Perfect overlap
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(0, 0, 10, 10)
    assert Rectangle.compute_iou(box1, box2) == pytest.approx(1.0)

    # Partial overlap: half width, identical height
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(5, 0, 10, 10)
    # intersection: 5 width, 10 height = 50 area
    # union: 100 + 100 - 50 = 150 area
    # IoU: 50 / 150 = 0.3333
    assert Rectangle.compute_iou(box1, box2) == pytest.approx(1.0 / 3.0)

    # No overlap
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(20, 20, 10, 10)
    assert Rectangle.compute_iou(box1, box2) == 0.0


def test_compute_sharpness():
    # Empty crop
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    assert imgutils.compute_sharpness(empty_crop) == 0.0

    # Constant crop (no edges -> variance of Laplacian is 0)
    flat_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    assert imgutils.compute_sharpness(flat_crop) == 0.0

    # Edge crop (sharp transitions -> positive variance)
    edge_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    edge_crop[:, 25:] = 255
    assert imgutils.compute_sharpness(edge_crop) > 0.0


def test_is_box_excluded_by_zones():
    # Test case 1: no zones
    assert (
        Rectangle.is_box_excluded_by_zones(Rectangle(10, 10, 20, 20), [], []) is False
    )

    # Test case 2: inclusion zone (box inside)
    inclusion_zones = [Rectangle(0, 0, 50, 50)]
    # Box (10, 10, 20, 20) inside inclusion zone (0, 0 to 50, 50) -> not excluded (False)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(10, 10, 20, 20), inclusion_zones, []
        )
        is False
    )

    # Box (60, 60, 20, 20) outside inclusion zone -> excluded (True)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(60, 60, 20, 20), inclusion_zones, []
        )
        is True
    )

    # Test case 3: exclusion zone (box inside)
    exclusion_zones = [Rectangle(0, 0, 50, 50)]
    # Box (10, 10, 20, 20) inside exclusion zone -> excluded (True)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(10, 10, 20, 20), [], exclusion_zones
        )
        is True
    )

    # Box (60, 60, 20, 20) outside exclusion zone -> not excluded (False)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(60, 60, 20, 20), [], exclusion_zones
        )
        is False
    )


def _make_record(entity_id, relative_speed, video_path="video.mp4"):
    # Minimal VideoEntityRecord stub - only entity_id/video_path/relative_speed
    # matter for compute_relative_speeds/calibrate_absolute_speeds, the rest
    # are just placeholders to satisfy the dataclass's required fields.
    return VideoEntityRecord(
        video_path=Path(video_path),
        entity_id=entity_id,
        best_frame_idx=0,
        best_frame=np.zeros((1, 1, 3), dtype=np.uint8),
        best_crop=np.zeros((1, 1, 3), dtype=np.uint8),
        best_box=Rectangle(0, 0, 1, 1),
        timestamp=0.0,
        hsv_hist=np.zeros((8, 8, 8)),
        aspect_ratio=1.0,
        direction="right",
        relative_speed=relative_speed,
    )


def test_compute_relative_speeds_normalizes_to_fastest():
    records = [_make_record(1, 10.0), _make_record(2, 40.0), _make_record(3, 20.0)]

    compute_relative_speeds(records)

    assert records[0].relative_speed == pytest.approx(0.25)
    assert records[1].relative_speed == pytest.approx(1.0)
    assert records[2].relative_speed == pytest.approx(0.5)


def test_compute_relative_speeds_all_zero_stays_zero():
    records = [_make_record(1, 0.0), _make_record(2, 0.0)]

    compute_relative_speeds(records)

    assert all(r.relative_speed == 0.0 for r in records)


def test_compute_relative_speeds_empty_list():
    assert compute_relative_speeds([]) == []


def test_calibrate_absolute_speeds_scales_linearly():
    records = [_make_record(1, 0.25), _make_record(2, 1.0), _make_record(3, 0.5)]

    calibrate_absolute_speeds(records, reference_entity_id=2, reference_speed=60.0)

    # entity 2 (relative_speed=1.0) is the reference at 60 mph, so scale = 60.
    assert records[0].absolute_speed == pytest.approx(15.0)
    assert records[1].absolute_speed == pytest.approx(60.0)
    assert records[2].absolute_speed == pytest.approx(30.0)


def test_calibrate_absolute_speeds_missing_entity_raises():
    records = [_make_record(1, 0.5)]

    with pytest.raises(ValueError):
        calibrate_absolute_speeds(records, reference_entity_id=99, reference_speed=60.0)


def test_calibrate_absolute_speeds_zero_reference_raises():
    records = [_make_record(1, 0.0)]

    with pytest.raises(ValueError):
        calibrate_absolute_speeds(records, reference_entity_id=1, reference_speed=60.0)


def test_calibrate_absolute_speeds_ambiguous_entity_id_requires_video():
    # entity_id resets per video, so the same ID can appear in two videos.
    records = [
        _make_record(1, 0.5, video_path="cam1.mp4"),
        _make_record(1, 0.25, video_path="cam2.mp4"),
    ]

    with pytest.raises(ValueError):
        calibrate_absolute_speeds(records, reference_entity_id=1, reference_speed=60.0)

    # Disambiguating with reference_video_path resolves it.
    calibrate_absolute_speeds(
        records,
        reference_entity_id=1,
        reference_speed=60.0,
        reference_video_path="cam1.mp4",
    )
    assert records[0].absolute_speed == pytest.approx(60.0)
    assert records[1].absolute_speed == pytest.approx(30.0)


def test_calibrate_absolute_speeds_video_mismatch_names_the_actual_video():
    # entity_id 1 exists, but not in the video the caller named - the error
    # should say so distinctly from "entity_id not found at all", and name
    # the video(s) it actually appears in so a wrong --reference-video is
    # obvious rather than reading as a missing/wrong entity_id.
    records = [_make_record(1, 0.5, video_path="cam1.mp4")]

    with pytest.raises(ValueError, match="cam1.mp4"):
        calibrate_absolute_speeds(
            records,
            reference_entity_id=1,
            reference_speed=60.0,
            reference_video_path="wrong_name.mp4",
        )


def test_process_video_computes_direction_and_relative_speed():
    # A box moving 5px/frame to the right across 5 frames, sourced from a
    # video_yolo.py-style frame_detections map instead of live detection.
    frame_w, frame_h = 200, 80
    box_w, box_h = 30, 25
    box_y = 20
    fps = 10.0

    frames = [np.zeros((frame_h, frame_w, 3), dtype=np.uint8) for _ in range(5)]
    frame_detections = {
        i: [(Rectangle(x=i * 5, y=box_y, w=box_w, h=box_h), 2)] for i in range(5)
    }  # COCO class id 2 = car

    import cv2

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: (
        fps
        if prop == cv2.CAP_PROP_FPS
        else (
            frame_w
            if prop == cv2.CAP_PROP_FRAME_WIDTH
            else frame_h if prop == cv2.CAP_PROP_FRAME_HEIGHT else 0
        )
    )
    mock_cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]

    with patch(
        "src.processing.video_entityprofiler.cv2.VideoCapture", return_value=mock_cap
    ):
        records = process_video(
            video_path=Path("nonexistent_video.mp4"),
            frame_detections=frame_detections,
        )

    assert len(records) == 1
    record = records[0]
    assert record.direction == "right"
    assert record.entity_type == 2
    # displacement = 20px (from center x=15 to center x=35) over 4 frames @ 10fps = 0.4s
    assert record.relative_speed == pytest.approx(50.0, rel=0.05)


def test_load_yolo_detections_groups_by_video_and_frame(tmp_path):
    report_path = tmp_path / "yolo_report.json"
    report_path.write_text(
        json.dumps(
            {
                "detections": {
                    "clip.mp4": [
                        {
                            "frame_index": 0,
                            "box": [10, 20, 40, 60],
                            "class_id": 2,
                            "label": "car",
                            "confidence": 0.9,
                        },
                        {
                            "frame_index": 0,
                            "box": [50, 20, 80, 60],
                            "class_id": 1,
                            "label": "bicycle",
                            "confidence": 0.5,
                        },
                        {
                            "frame_index": 3,
                            "box": [12, 22, 42, 62],
                            "class_id": 2,
                            "label": "car",
                            "confidence": 0.8,
                        },
                    ]
                }
            }
        )
    )

    detections_by_video = load_yolo_detections(report_path)

    assert set(detections_by_video.keys()) == {"clip.mp4"}
    frame_map = detections_by_video["clip.mp4"]
    assert set(frame_map.keys()) == {0, 3}
    assert len(frame_map[0]) == 2
    rect0, class_id0 = frame_map[0][0]
    assert (rect0.x, rect0.y, rect0.w, rect0.h) == (10, 20, 30, 40)
    assert class_id0 == 2
    assert frame_map[0][1][1] == 1
    rect3, class_id3 = frame_map[3][0]
    assert (rect3.x, rect3.y, rect3.w, rect3.h) == (12, 22, 30, 40)
    assert class_id3 == 2
