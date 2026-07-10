from pathlib import Path

import pytest

from src.processing.plate_dwellprofiler import (
    PlateDetection,
    compute_plate_dwell_times,
)


def test_compute_plate_dwell_times_simple_pair():
    detections = [
        PlateDetection(
            plate_text="ABC123", timestamp=10.0, img_path=Path("entry1.jpg")
        ),
        PlateDetection(plate_text="ABC123", timestamp=50.0, img_path=Path("exit1.jpg")),
    ]

    matches, single_sightings = compute_plate_dwell_times(detections)

    assert len(matches) == 1
    assert matches[0]["plate_text"] == "ABC123"
    assert matches[0]["entry_image"] == str(Path("entry1.jpg"))
    assert matches[0]["exit_image"] == str(Path("exit1.jpg"))
    assert matches[0]["dwell_time"] == pytest.approx(40.0)
    assert matches[0]["num_sightings"] == 2
    assert single_sightings == []


def test_compute_plate_dwell_times_different_plates_dont_match():
    detections = [
        PlateDetection(plate_text="AAA111", timestamp=10.0, img_path=Path("img1.jpg")),
        PlateDetection(plate_text="BBB222", timestamp=50.0, img_path=Path("img2.jpg")),
    ]

    matches, single_sightings = compute_plate_dwell_times(detections)

    assert matches == []
    assert len(single_sightings) == 2


def test_compute_plate_dwell_times_uses_earliest_and_latest_regardless_of_source():
    # No entry/exit split by directory or direction - plain pooled sightings.
    # The middle reading shouldn't affect the computed entry/exit pairing.
    detections = [
        PlateDetection(plate_text="XYZ789", timestamp=10.0, img_path=Path("first.jpg")),
        PlateDetection(
            plate_text="XYZ789", timestamp=50.0, img_path=Path("middle.jpg")
        ),
        PlateDetection(plate_text="XYZ789", timestamp=100.0, img_path=Path("last.jpg")),
    ]

    matches, single_sightings = compute_plate_dwell_times(detections)

    assert len(matches) == 1
    assert matches[0]["entry_image"] == str(Path("first.jpg"))
    assert matches[0]["exit_image"] == str(Path("last.jpg"))
    assert matches[0]["dwell_time"] == pytest.approx(90.0)
    assert matches[0]["num_sightings"] == 3
    assert single_sightings == []


def test_compute_plate_dwell_times_repeat_visits_use_overall_first_and_last():
    # A plate seen entering, leaving, and entering again (re-entry) still only
    # yields one dwell record spanning the very first to the very last sighting,
    # since there's no direction signal to split it into two crossings.
    detections = [
        PlateDetection(
            plate_text="XYZ789", timestamp=10.0, img_path=Path("entry_a.jpg")
        ),
        PlateDetection(
            plate_text="XYZ789", timestamp=50.0, img_path=Path("exit_a.jpg")
        ),
        PlateDetection(
            plate_text="XYZ789", timestamp=100.0, img_path=Path("entry_b.jpg")
        ),
        PlateDetection(
            plate_text="XYZ789", timestamp=150.0, img_path=Path("exit_b.jpg")
        ),
    ]

    matches, single_sightings = compute_plate_dwell_times(detections)

    assert len(matches) == 1
    assert matches[0]["entry_image"] == str(Path("entry_a.jpg"))
    assert matches[0]["exit_image"] == str(Path("exit_b.jpg"))
    assert matches[0]["dwell_time"] == pytest.approx(140.0)
    assert matches[0]["num_sightings"] == 4
    assert single_sightings == []


def test_compute_plate_dwell_times_single_sighting_reported_separately():
    detections = [
        PlateDetection(plate_text="ABC123", timestamp=100.0, img_path=Path("only.jpg")),
    ]

    matches, single_sightings = compute_plate_dwell_times(detections)

    assert matches == []
    assert len(single_sightings) == 1
    assert single_sightings[0].img_path == Path("only.jpg")
