from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.detection.direction.vehicle_direction import (
    FRONT_KEYPOINTS,
    REAR_KEYPOINTS,
    VehicleDirection,
    detect_vehicle_pose,
    parse_vehicle_directions,
    select_best_match,
)


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = xy
        self.conf = conf


class _FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = xyxy

    def __len__(self):
        return len(self.xyxy)


class _FakeDatum:
    def __init__(self, keypoints=None, boxes=None):
        self.keypoints = keypoints
        self.boxes = boxes


def _make_datum(front_x=800.0, rear_x=100.0, front_conf=0.9, rear_conf=0.9, box=(10, 20, 300, 400)):
    """
    Builds a single-instance fake pose result: every FRONT_KEYPOINTS index
    sits at front_x with front_conf confidence, every REAR_KEYPOINTS index
    sits at rear_x with rear_conf confidence, and indices 8/13 (the excluded
    ones) are zeroed out entirely since they should never be read.
    """
    xy = torch.zeros((14, 2))
    conf = torch.zeros(14)
    for idx in FRONT_KEYPOINTS:
        xy[idx] = torch.tensor([front_x, 500.0])
        conf[idx] = front_conf
    for idx in REAR_KEYPOINTS:
        xy[idx] = torch.tensor([rear_x, 500.0])
        conf[idx] = rear_conf

    keypoints = _FakeKeypoints(xy=xy.unsqueeze(0), conf=conf.unsqueeze(0))
    boxes = _FakeBoxes(xyxy=torch.tensor([list(box)], dtype=torch.float32))
    return _FakeDatum(keypoints=keypoints, boxes=boxes)


def test_parse_vehicle_directions_front_right_of_rear_is_right(tmp_path):
    datum = _make_datum(front_x=800.0, rear_x=100.0)
    directions = parse_vehicle_directions([datum], tmp_path / "car.jpg")

    assert len(directions) == 1
    assert directions[0].label == "right"
    assert directions[0].box == (10, 20, 300, 400)
    assert directions[0].confidence == pytest.approx(1.0)


def test_parse_vehicle_directions_front_left_of_rear_is_left(tmp_path):
    datum = _make_datum(front_x=100.0, rear_x=800.0)
    directions = parse_vehicle_directions([datum], tmp_path / "car.jpg")

    assert directions[0].label == "left"


def test_parse_vehicle_directions_too_few_confident_points_is_unknown(tmp_path):
    datum = _make_datum(front_conf=0.9, rear_conf=0.9)
    # Drop all but one FRONT_KEYPOINTS confidence below the gate.
    for idx in FRONT_KEYPOINTS[1:]:
        datum.keypoints.conf[0][idx] = 0.0

    directions = parse_vehicle_directions([datum], tmp_path / "car.jpg")

    assert directions[0].label == "unknown"


def test_parse_vehicle_directions_excludes_indices_8_and_13(tmp_path):
    datum = _make_datum(front_x=800.0, rear_x=100.0)
    # Poison the excluded indices with values that would flip the verdict
    # if they were ever read.
    datum.keypoints.xy[0][8] = torch.tensor([100.0, 500.0])
    datum.keypoints.conf[0][8] = 1.0
    datum.keypoints.xy[0][13] = torch.tensor([800.0, 500.0])
    datum.keypoints.conf[0][13] = 1.0

    directions = parse_vehicle_directions([datum], tmp_path / "car.jpg")

    assert directions[0].label == "right"


def test_parse_vehicle_directions_skips_datum_with_no_keypoints(tmp_path):
    datum = _FakeDatum(keypoints=None, boxes=None)
    assert parse_vehicle_directions([datum], tmp_path / "car.jpg") == []


def test_parse_vehicle_directions_confidence_is_fraction_of_confident_points(tmp_path):
    datum = _make_datum(front_conf=0.9, rear_conf=0.9)
    # Only 3 of the 6 front keypoints clear the gate.
    for idx in FRONT_KEYPOINTS[3:]:
        datum.keypoints.conf[0][idx] = 0.0

    directions = parse_vehicle_directions([datum], tmp_path / "car.jpg")

    # 3 front + 6 rear = 9 of 12 total front/rear keypoints confident.
    assert directions[0].confidence == pytest.approx(9 / 12)


def _direction(box, label="right"):
    return VehicleDirection(box=box, label=label, confidence=1.0, image_path=None)


def test_select_best_match_picks_highest_iou():
    close_match = _direction((10, 10, 110, 110))
    far_match = _direction((500, 500, 600, 600))

    best = select_best_match([far_match, close_match], reference_box=(10, 10, 110, 110))

    assert best is close_match


def test_select_best_match_empty_list_returns_none():
    assert select_best_match([], reference_box=(0, 0, 10, 10)) is None


def test_detect_vehicle_pose_runs_on_full_image_no_crop(tmp_path):
    img_path = tmp_path / "car.jpg"
    Image.new("RGB", (200, 100), (0, 0, 0)).save(img_path)

    fake_model = MagicMock()
    fake_model.predict.return_value = ["fake results"]

    result = detect_vehicle_pose(fake_model, img_path, conf=0.01)

    assert result == ["fake results"]
    call_kwargs = fake_model.predict.call_args.kwargs
    assert call_kwargs["conf"] == 0.01
    assert call_kwargs["classes"] == [0]
    # The image handed to predict() must be the full, uncropped frame.
    assert call_kwargs["source"].size == (200, 100)
