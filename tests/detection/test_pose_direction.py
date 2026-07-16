from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from src.detection.pose_direction import (
    Direction,
    parse_poses,
    save_annotated_results,
    process_single_image,
)


def _make_result(xy: list, conf: list):
    """Builds a fake ultralytics Results-like object with one entry per person."""
    result = MagicMock()
    result.keypoints = MagicMock()
    result.keypoints.xy = torch.tensor(xy, dtype=torch.float32)
    result.keypoints.conf = torch.tensor(conf, dtype=torch.float32)
    return result


def _single_person_keypoints(left_x=100, right_x=200, left_y=30, right_y=70, nose=(150, 5)):
    """
    Builds 17 COCO keypoints for one person.
    Indices 1,3,5,7,9,11,13,15 (left_points) get (left_x, left_y).
    Indices 2,4,6,8,10,12,14,16 (right_points) get (right_x, right_y).
    Index 0 (nose, uncategorized) gets `nose`.
    """
    kps = [None] * 17
    kps[0] = nose
    for i in (1, 3, 5, 7, 9, 11, 13, 15):
        kps[i] = (left_x, left_y)
    for i in (2, 4, 6, 8, 10, 12, 14, 16):
        kps[i] = (right_x, right_y)
    return kps


def test_parse_poses_no_keypoints_returns_empty():
    result = MagicMock()
    result.keypoints = None
    assert parse_poses([result], "img.jpg") == []


def test_parse_poses_all_equal_gives_unknown_label():
    kps = _single_person_keypoints(left_x=150, right_x=150, left_y=50, right_y=50, nose=(150, 50))
    result = _make_result([kps], [[1.0] * 17])

    directions = parse_poses([result], "img.jpg")

    assert len(directions) == 1
    d = directions[0]
    assert d.left_right == 0
    assert d.front_back == 0
    assert d.label == "unknown"
    assert d.box == (150, 50, 150, 50)
    assert d.image_path == Path("img.jpg")


def test_parse_poses_computes_label_and_box_from_keypoints():
    kps = _single_person_keypoints()
    result = _make_result([kps], [[1.0] * 17])

    directions = parse_poses([result], "img.jpg")

    assert len(directions) == 1
    d = directions[0]
    # front_points{1,2,3,9,10} avg=140, back_points{5,6,11,12} avg=150 -> front<back -> left_right=-1
    assert d.left_right == -1
    # left_points avg=100, right_points avg=200 -> left<right -> front_back=-1
    assert d.front_back == -1
    # abs(left-right)=100 > abs(front-back)=10 -> label uses left_right; -1 -> "left"
    assert d.label == "left"
    # bbox spans all confident keypoints including the uncategorized nose point
    assert d.box == (100, 5, 200, 70)


def test_parse_poses_low_confidence_points_are_excluded():
    kps = _single_person_keypoints()
    conf = [1.0] * 17
    # drop confidence on the nose point below threshold
    conf[0] = 0.1
    result = _make_result([kps], [conf])

    directions = parse_poses([result], "img.jpg", min_conf=0.25)

    d = directions[0]
    # nose point excluded entirely, so bbox no longer reflects its (150, 5) coordinates
    assert d.box == (100, 30, 200, 70)


def test_parse_poses_non_positive_coordinates_excluded_from_averages_but_not_bbox():
    kps = _single_person_keypoints()
    kps[7] = (0, 0)  # a left_point coordinate at the origin
    result = _make_result([kps], [[1.0] * 17])

    directions = parse_poses([result], "img.jpg")

    d = directions[0]
    # (0, 0) still updates the bbox mins even though it's excluded from left_total tally
    assert d.box[0] == 0
    assert d.box[1] == 0


def test_parse_poses_min_points_threshold_forces_unknown_component():
    kps = _single_person_keypoints()
    result = _make_result([kps], [[1.0] * 17])

    # front_points has only 5 members; requiring 6 forces left_right (lr) to 0
    directions = parse_poses([result], "img.jpg", min_points=6)

    d = directions[0]
    assert d.left_right == 0
    assert d.front_back == -1


def test_parse_poses_explicit_box_overrides_computed_bbox():
    kps = _single_person_keypoints()
    result = _make_result([kps], [[1.0] * 17])

    explicit_box = (0, 0, 5, 5)
    directions = parse_poses([result], "img.jpg", box=explicit_box)

    assert directions[0].box == explicit_box


def test_parse_poses_multiple_people_produce_multiple_directions():
    person1 = _single_person_keypoints(left_x=100, right_x=200)
    person2 = _single_person_keypoints(left_x=200, right_x=100)
    result = _make_result([person1, person2], [[1.0] * 17, [1.0] * 17])

    directions = parse_poses([result], "img.jpg")

    assert len(directions) == 2


def test_parse_poses_string_image_path_converted_to_path():
    kps = _single_person_keypoints()
    result = _make_result([kps], [[1.0] * 17])

    directions = parse_poses([result], "some/dir/img.jpg")

    assert isinstance(directions[0].image_path, Path)
    assert directions[0].image_path == Path("some/dir/img.jpg")


def _write_dummy_image(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((100, 100, 3), dtype=np.uint8))


def test_save_annotated_results_no_directions_saves_original_copy(tmp_path):
    input_folder = tmp_path / "in"
    output_folder = tmp_path / "out"
    img_path = input_folder / "cam1" / "img.jpg"
    _write_dummy_image(img_path)

    save_annotated_results(img_path, [], input_folder, output_folder)

    expected = output_folder / "cam1" / "img.jpg"
    assert expected.exists()


def test_save_annotated_results_writes_one_file_per_direction(tmp_path):
    input_folder = tmp_path / "in"
    output_folder = tmp_path / "out"
    img_path = input_folder / "img.jpg"
    _write_dummy_image(img_path)

    directions = [
        Direction((0, 0, 10, 10), 1, 1, img_path, "right"),
        Direction((20, 20, 30, 30), -1, -1, img_path, "left"),
    ]

    save_annotated_results(img_path, directions, input_folder, output_folder)

    assert (output_folder / "img-0-right.jpg").exists()
    assert (output_folder / "img-1-left.jpg").exists()
    assert not (output_folder / "img.jpg").exists()


def test_save_annotated_results_missing_source_image_is_a_noop(tmp_path):
    input_folder = tmp_path / "in"
    output_folder = tmp_path / "out"
    missing_path = input_folder / "missing.jpg"
    input_folder.mkdir(parents=True)

    save_annotated_results(missing_path, [], input_folder, output_folder)

    assert not output_folder.exists()


def test_process_single_image_runs_detection_and_saves(tmp_path):
    input_folder = tmp_path / "in"
    output_folder = tmp_path / "out"
    img_path = input_folder / "img.jpg"
    _write_dummy_image(img_path)

    kps = _single_person_keypoints()
    fake_result = _make_result([kps], [[1.0] * 17])

    mock_model = MagicMock()
    mock_model.predict.return_value = [fake_result]

    directions = process_single_image(
        model=mock_model,
        img_path=img_path,
        input_folder=input_folder,
        output_folder=output_folder,
        save_images=True,
    )

    assert len(directions) == 1
    assert directions[0].label == "left"
    mock_model.predict.assert_called_once()
    assert (output_folder / "img-0-left.jpg").exists()


def test_process_single_image_skip_save(tmp_path):
    input_folder = tmp_path / "in"
    output_folder = tmp_path / "out"
    img_path = input_folder / "img.jpg"
    _write_dummy_image(img_path)

    kps = _single_person_keypoints()
    fake_result = _make_result([kps], [[1.0] * 17])

    mock_model = MagicMock()
    mock_model.predict.return_value = [fake_result]

    process_single_image(
        model=mock_model,
        img_path=img_path,
        input_folder=input_folder,
        output_folder=output_folder,
        save_images=False,
    )

    assert not output_folder.exists()
