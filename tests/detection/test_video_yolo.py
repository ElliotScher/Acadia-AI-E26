import sys
from typing import List

import cv2
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.detection.video_yolo import process_videos, DetectionResult, main
from utility.geometryutils import Rectangle


def test_process_videos_basic():
    # Mocking YOLO, VideoCapture
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [2]  # COCO class 2 (car)
    mock_box.xyxy = [[10, 20, 100, 200]]
    mock_box.conf = [0.85]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.names = {2: "car"}
    mock_model.predict.return_value = [mock_result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Return a frame once, then False to terminate the loop
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]

    mock_progress = MagicMock()

    with (
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
    ):

        results = process_videos(
            video_paths=[Path("dummy_input.mp4")],
            model_name="dummy_model.pt",
            progress_bar=mock_progress,
            conf_threshold=0.25,
            target_classes=[2],
        )

        # Assertions
        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        assert results[0].video_path == Path("dummy_input.mp4")
        assert len(results[0].boxes) == 1
        frame_idx, rect, coco_id, conf, label = results[0].boxes[0]
        assert frame_idx == 0
        assert rect.x == 10
        assert rect.y == 20
        assert rect.w == 90
        assert rect.h == 180
        assert coco_id == 2
        assert conf == pytest.approx(0.85)
        assert label == "car"

        # Verify calls
        mock_model.predict.assert_called_once()
        assert mock_cap.read.call_count == 2
        mock_progress.update.assert_called_once_with(1)


def test_process_videos_inclusion_region():
    # Mocking YOLO, VideoCapture
    mock_model = MagicMock()
    mock_box_inside = MagicMock()
    mock_box_inside.cls = [2]
    mock_box_inside.xyxy = [[10, 10, 20, 20]]  # intersects inclusion region
    mock_box_inside.conf = [0.85]

    mock_box_outside = MagicMock()
    mock_box_outside.cls = [2]
    mock_box_outside.xyxy = [
        [100, 100, 120, 120]
    ]  # does not intersect inclusion region
    mock_box_outside.conf = [0.90]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box_inside, mock_box_outside]
    mock_model.names = {2: "car"}
    mock_model.predict.return_value = [mock_result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]

    mock_progress = MagicMock()

    # Inclusion region: x=0, y=0, w=50, h=50
    inclusion = Rectangle(0, 0, 50, 50)

    with (
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
    ):

        results = process_videos(
            video_paths=[Path("dummy_input.mp4")],
            model_name="dummy_model.pt",
            progress_bar=mock_progress,
            inclusion_region=inclusion,
            conf_threshold=0.25,
            target_classes=[2],
        )

        # Assertions: only mock_box_inside should be detected (1 detection)
        assert len(results) == 1
        assert len(results[0].boxes) == 1
        frame_idx, rect, coco_id, conf, label = results[0].boxes[0]
        assert frame_idx == 0
        assert rect.x == 10
        assert rect.y == 10
        assert rect.w == 10
        assert rect.h == 10
        assert coco_id == 2
        assert conf == pytest.approx(0.85)
        assert label == "car"

        # Verify calls
        mock_model.predict.assert_called_once()
        assert mock_cap.read.call_count == 2
        mock_progress.update.assert_called_once_with(1)


def test_process_videos_vehicle_merge_default_folds_truck_into_car():
    # vehicle_merge defaults to True, so a detected truck (COCO id 7) should
    # be reported as a merged car (id 2).
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [7]  # COCO class 7 (truck)
    mock_box.xyxy = [[10, 20, 100, 200]]
    mock_box.conf = [0.9]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.names = {7: "truck"}
    mock_model.predict.return_value = [mock_result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]

    mock_progress = MagicMock()

    with (
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
    ):
        results = process_videos(
            video_paths=[Path("dummy_input.mp4")],
            model_name="dummy_model.pt",
            progress_bar=mock_progress,
            conf_threshold=0.25,
            target_classes=[7],
        )

        assert len(results[0].boxes) == 1
        _, _, coco_id, _, label = results[0].boxes[0]
        assert coco_id == 2
        assert label == "car"


def test_process_videos_vehicle_merge_disabled_keeps_truck_distinct():
    # With vehicle_merge=False, a detected truck (COCO id 7) should be kept
    # as its own class rather than folded into car.
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [7]  # COCO class 7 (truck)
    mock_box.xyxy = [[10, 20, 100, 200]]
    mock_box.conf = [0.9]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.names = {7: "truck"}
    mock_model.predict.return_value = [mock_result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]

    mock_progress = MagicMock()

    with (
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
    ):
        results = process_videos(
            video_paths=[Path("dummy_input.mp4")],
            model_name="dummy_model.pt",
            progress_bar=mock_progress,
            conf_threshold=0.25,
            target_classes=[7],
            vehicle_merge=False,
        )

        assert len(results[0].boxes) == 1
        _, _, coco_id, _, label = results[0].boxes[0]
        assert coco_id == 7
        assert label == "truck"


def test_process_videos_vehicle_merge_disabled_leaves_car_unchanged():
    # Class ids that aren't bus/truck (e.g. car, id 2) should pass through
    # unchanged regardless of vehicle_merge.
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.cls = [2]  # COCO class 2 (car)
    mock_box.xyxy = [[10, 20, 100, 200]]
    mock_box.conf = [0.9]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.names = {2: "car"}
    mock_model.predict.return_value = [mock_result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]

    mock_progress = MagicMock()

    with (
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
    ):
        results = process_videos(
            video_paths=[Path("dummy_input.mp4")],
            model_name="dummy_model.pt",
            progress_bar=mock_progress,
            conf_threshold=0.25,
            target_classes=[2],
            vehicle_merge=False,
        )

        assert len(results[0].boxes) == 1
        _, _, coco_id, _, label = results[0].boxes[0]
        assert coco_id == 2
        assert label == "car"


def _run_main_and_capture_process_videos_kwargs(cli_args, tmp_path):
    """Runs main() with the given extra CLI flags and returns the kwargs
    process_videos() was called with, verifying the --merge flag is wired
    from the CLI through to the real detection path end-to-end."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "video.mp4").write_bytes(b"")

    mock_model = MagicMock()
    mock_model.names = {cls: str(cls) for cls in [0, 1, 2, 3, 5, 7]}

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 0

    captured_kwargs = {}

    def fake_process_videos(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return []

    argv = ["video_yolo.py", str(input_dir), str(output_dir), "--no-save", *cli_args]

    with (
        patch.object(sys, "argv", argv),
        patch("src.detection.video_yolo.load_model", return_value=mock_model),
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch(
            "src.detection.video_yolo.process_videos",
            side_effect=fake_process_videos,
        ),
    ):
        main()

    return captured_kwargs


def test_main_merge_flag_defaults_to_true(tmp_path):
    kwargs = _run_main_and_capture_process_videos_kwargs([], tmp_path)
    assert kwargs["vehicle_merge"] is True


def test_main_no_merge_flag_disables_merge(tmp_path):
    kwargs = _run_main_and_capture_process_videos_kwargs(["--no-merge"], tmp_path)
    assert kwargs["vehicle_merge"] is False


def test_main_explicit_merge_flag_enables_merge(tmp_path):
    kwargs = _run_main_and_capture_process_videos_kwargs(["--merge"], tmp_path)
    assert kwargs["vehicle_merge"] is True
