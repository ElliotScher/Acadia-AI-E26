import pytest
import numpy as np
import datetime
from pathlib import Path
from src.utility.imgutils import (
    get_center_crop,
    get_hsv_hist,
    detect_entities,
    get_timestamp,
    load_video_start_times,
    validate_video_start_times,
    _normalize_plate_text,
)


def test_get_center_crop(dummy_bgr_image):
    # For a 100x100 image, 10% margin should crop 10 pixels from all sides -> 80x80
    cropped = get_center_crop(dummy_bgr_image, 0.10)
    assert cropped.shape == (80, 80, 3)


def test_get_center_crop_empty_margin(dummy_bgr_image):
    # For a margin of 0.5 (or greater), it might result in crop.size == 0, falling back to original
    cropped = get_center_crop(dummy_bgr_image, 0.50)
    assert cropped.shape == (100, 100, 3)


def test_get_hsv_hist(dummy_bgr_image):
    hist = get_hsv_hist(dummy_bgr_image)
    assert hist.shape == (8, 8, 8)
    # The max value should be normalized to 1.0 (or at least close/exact depending on normalization)
    assert np.isclose(hist.max(), 1.0) or hist.max() == 1.0


def test_detect_entities(dummy_green_box_image):
    boxes = detect_entities(dummy_green_box_image)
    assert len(boxes) == 1
    x, y, w, h = boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h
    # Bounding box should enclose the green region (25, 25) to (55, 55)
    assert abs(x - 25) <= 2
    assert abs(y - 25) <= 2
    assert abs(w - 31) <= 2
    assert abs(h - 31) <= 2


def test_detect_entities_no_green(dummy_bgr_image):
    # A completely black image has no green box
    boxes = detect_entities(dummy_bgr_image)
    assert len(boxes) == 0


def test_get_timestamp_from_filename():
    # Test file with format: YYYY-MM-DD/HH-MM-SS.jpg
    # Path doesn't need to exist on disk for the filename parser to try to run
    img_path = Path("2026-07-03/12-30-15.jpg")
    ts = get_timestamp(img_path)
    # 2026-07-03 12:30:15
    import datetime

    expected = datetime.datetime(2026, 7, 3, 12, 30, 15).timestamp()
    assert ts == expected


def test_get_timestamp_filename_fallback():
    # Path doesn't match standard parent date folder, so it falls back to seconds-in-day float
    img_path = Path("not-a-date/12-30-15.jpg")
    ts = get_timestamp(img_path)
    assert ts == float(12 * 3600 + 30 * 60 + 15)


def test_get_timestamp_not_found():
    # A path that does not exist and doesn't match filename pattern raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        get_timestamp(Path("non_existent_file.jpg"))


def test_load_video_start_times_parses_epoch_and_iso(tmp_path):
    json_path = tmp_path / "start_times.json"
    json_path.write_text('[1700000000, "2026-07-08T14:30:00"]')

    start_times = load_video_start_times(json_path)

    assert start_times[0] == datetime.datetime.fromtimestamp(1700000000)
    assert start_times[1] == datetime.datetime(2026, 7, 8, 14, 30, 0)
    assert all(isinstance(t, datetime.datetime) for t in start_times)


def test_load_video_start_times_rejects_invalid_value(tmp_path):
    json_path = tmp_path / "start_times.json"
    json_path.write_text("[null]")

    with pytest.raises(ValueError):
        load_video_start_times(json_path)


def test_load_video_start_times_rejects_non_array(tmp_path):
    json_path = tmp_path / "start_times.json"
    json_path.write_text('{"clip1.mp4": 1700000000}')

    with pytest.raises(ValueError):
        load_video_start_times(json_path)


def test_validate_video_start_times_none_is_always_ok():
    validate_video_start_times(None, 0)
    validate_video_start_times(None, 5)


def test_validate_video_start_times_matching_length_is_ok():
    validate_video_start_times([1.0, 2.0, 3.0], 3)


def test_validate_video_start_times_mismatched_length_raises():
    with pytest.raises(ValueError):
        validate_video_start_times([1.0, 2.0], 3)

    with pytest.raises(ValueError):
        validate_video_start_times([1.0, 2.0, 3.0], 2)


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        ("ABC1234", "ABC1234"),
        ("abc-1234\n", "ABC1234"),
        ("  7 GXR 21 ", "7GXR21"),
        ("!!!", ""),
    ],
)
def test_normalize_plate_text(raw_text, expected):
    assert _normalize_plate_text(raw_text) == expected


@pytest.mark.parametrize(
    "filename,expected_dt",
    [
        ("entity_1_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 11, 42)),
        ("entity_2_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 13, 7)),
        ("entity_3_left_car.jpg", datetime.datetime(2026, 7, 1, 7, 13, 59)),
        ("entity_4_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 14, 59)),
        ("entity_5_left_car.jpg", datetime.datetime(2026, 7, 1, 7, 17, 42)),
        ("entity_6_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 17, 59)),
        ("entity_7_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 22, 20)),
        ("entity_8_right_car.jpg", datetime.datetime(2026, 7, 1, 7, 22, 58)),
    ],
)
def test_extract_timestamp_via_ocr_parameterized(filename, expected_dt):
    from src.utility.imgutils import extract_timestamp_via_ocr

    img_path = Path(__file__).parent.parent / "data" / "images" / "OCR" / filename
    ts = extract_timestamp_via_ocr(img_path)

    if expected_dt is None:
        assert ts is None
    else:
        assert ts is not None
        assert ts == expected_dt.timestamp()
