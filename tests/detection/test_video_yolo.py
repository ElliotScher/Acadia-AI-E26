from typing import List

import cv2
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.detection.video_yolo import process_videos, DetectionResult
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
        patch("src.detection.video_yolo.YOLO", return_value=mock_model),
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
        frame_idx, rect, label, conf = results[0].boxes[0]
        assert frame_idx == 0
        assert rect.x == 10
        assert rect.y == 20
        assert rect.w == 90
        assert rect.h == 180
        assert label == "car"
        assert conf == pytest.approx(0.85)

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
        patch("src.detection.video_yolo.YOLO", return_value=mock_model),
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
        frame_idx, rect, label, conf = results[0].boxes[0]
        assert frame_idx == 0
        assert rect.x == 10
        assert rect.y == 10
        assert rect.w == 10
        assert rect.h == 10
        assert label == "car"
        assert conf == pytest.approx(0.85)

        # Verify calls
        mock_model.predict.assert_called_once()
        assert mock_cap.read.call_count == 2
        mock_progress.update.assert_called_once_with(1)