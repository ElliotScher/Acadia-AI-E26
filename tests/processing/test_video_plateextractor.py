import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.processing.video_plateextractor import (
    crop_with_padding,
    extract_plate_crops,
    find_videos,
    process_video,
    run_video_plate_extraction,
)


def test_find_videos_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "sub" / "b.mov").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")

    videos = find_videos(tmp_path)

    assert videos == sorted([tmp_path / "a.mp4", tmp_path / "sub" / "b.mov"])


def test_crop_with_padding_expands_and_clamps():
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = crop_with_padding(img, 40, 40, 60, 60, padding_pct=0.1)

    # Box is 20x20; 10% padding on each side adds 2px per side.
    assert crop.shape[:2] == (24, 24)

    # Padding that would go out of bounds is clamped to the image edge.
    edge_crop = crop_with_padding(img, 0, 0, 10, 10, padding_pct=0.5)
    assert edge_crop.shape[:2] == (15, 15)


def test_extract_plate_crops_returns_one_entry_per_detection():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    box1 = MagicMock()
    box1.xyxy = [[10, 10, 50, 40]]
    box1.conf = [0.9]

    box2 = MagicMock()
    box2.xyxy = [[100, 100, 150, 140]]
    box2.conf = [0.6]

    result = MagicMock()
    result.boxes = [box1, box2]

    model = MagicMock()
    model.predict.return_value = [result]

    crops = extract_plate_crops(model, frame, conf_threshold=0.25)

    assert len(crops) == 2
    assert crops[0]["confidence"] == pytest.approx(0.9)
    assert crops[1]["confidence"] == pytest.approx(0.6)
    assert crops[0]["crop"].size > 0
    assert crops[1]["crop"].size > 0


def test_extract_plate_crops_empty_when_no_detections():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    result = MagicMock()
    result.boxes = []
    model = MagicMock()
    model.predict.return_value = [result]

    assert extract_plate_crops(model, frame, conf_threshold=0.25) == []


def test_process_video_downsample_skips_frames(tmp_path):
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "clip.mp4"
    video_path.write_bytes(b"")
    output_folder = tmp_path / "out"
    output_folder.mkdir()

    box = MagicMock()
    box.xyxy = [[5, 5, 15, 15]]
    box.conf = [0.8]
    result = MagicMock()
    result.boxes = [box]

    model = MagicMock()
    model.predict.return_value = [result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0  # fps
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    # 4 frames total; downsample=2 should only run detection on frames 2 and 4.
    mock_cap.read.side_effect = [
        (True, frame),
        (True, frame),
        (True, frame),
        (True, frame),
        (False, None),
    ]

    manifest = {}
    with (
        patch(
            "src.processing.video_plateextractor.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("src.processing.video_plateextractor.get_timestamp", return_value=1000.0),
    ):
        saved = process_video(
            video_path=video_path,
            input_folder=input_folder,
            output_folder=output_folder,
            model=model,
            conf_threshold=0.25,
            downsample_factor=2,
            manifest=manifest,
        )

    assert saved == 2
    assert model.predict.call_count == 2
    assert len(manifest) == 2
    for entry in manifest.values():
        assert entry["source_video"] == str(video_path)
        assert entry["confidence"] == pytest.approx(0.8)


def test_run_video_plate_extraction_writes_manifest(tmp_path):
    input_folder = tmp_path / "videos"
    input_folder.mkdir()
    video_path = input_folder / "clip.mp4"
    video_path.write_bytes(b"")
    output_folder = tmp_path / "out"

    box = MagicMock()
    box.xyxy = [[5, 5, 15, 15]]
    box.conf = [0.8]
    result = MagicMock()
    result.boxes = [box]

    model = MagicMock()
    model.predict.return_value = [result]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, frame), (False, None)]

    with (
        patch("src.processing.video_plateextractor.YOLO", return_value=model),
        patch(
            "src.processing.video_plateextractor.cv2.VideoCapture",
            return_value=mock_cap,
        ),
        patch("src.processing.video_plateextractor.get_timestamp", return_value=1000.0),
    ):
        summary = run_video_plate_extraction(
            input_dir=input_folder,
            output_dir=output_folder,
            plate_model="dummy_plate_model.pt",
        )

    assert summary["statistics"]["videos_processed"] == 1
    assert summary["statistics"]["plates_extracted"] == 1

    manifest_path = output_folder / "plate_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert len(manifest) == 1
