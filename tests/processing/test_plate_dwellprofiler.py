from pathlib import Path

import pytest

from src.processing.plate_dwellprofiler import (
    PlateDetection,
    compute_average_dwell_time,
    compute_plate_dwell_times,
    levenshtein_distance,
    normalize_input_dirs,
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


def _match(dwell_time):
    return {"plate_text": "X", "dwell_time": dwell_time, "num_sightings": 2}


def test_compute_average_dwell_time_no_filtering_by_default():
    matches = [_match(5.0), _match(45.0), _match(100.0)]

    avg, excluded = compute_average_dwell_time(matches)

    assert avg == pytest.approx((5.0 + 45.0 + 100.0) / 3)
    assert excluded == 0
    assert all(m["counted_in_average"] for m in matches)


def test_compute_average_dwell_time_excludes_short_matches():
    matches = [_match(2.0), _match(3.0), _match(45.0), _match(100.0)]

    avg, excluded = compute_average_dwell_time(matches, min_dwell_time=10.0)

    assert avg == pytest.approx((45.0 + 100.0) / 2)
    assert excluded == 2
    assert [m["counted_in_average"] for m in matches] == [False, False, True, True]


def test_compute_average_dwell_time_threshold_is_inclusive():
    matches = [_match(10.0), _match(9.9)]

    avg, excluded = compute_average_dwell_time(matches, min_dwell_time=10.0)

    assert avg == pytest.approx(10.0)
    assert excluded == 1
    assert matches[0]["counted_in_average"] is True
    assert matches[1]["counted_in_average"] is False


def test_compute_average_dwell_time_all_excluded_returns_zero():
    matches = [_match(1.0), _match(2.0)]

    avg, excluded = compute_average_dwell_time(matches, min_dwell_time=10.0)

    assert avg == 0.0
    assert excluded == 2


def test_compute_average_dwell_time_no_matches():
    avg, excluded = compute_average_dwell_time([], min_dwell_time=10.0)

    assert avg == 0.0
    assert excluded == 0


def test_run_plate_dwell_profiling_min_dwell_time_excludes_short_crossings(tmp_path):
    import json

    import cv2
    import numpy as np
    from unittest.mock import patch

    from src.processing.plate_dwellprofiler import run_plate_dwell_profiling

    img = np.zeros((50, 100, 3), dtype="uint8")
    entries = [
        ("a1.jpg", "AAA111", 1.0),
        ("a2.jpg", "AAA111", 3.0),  # 2s dwell - short, should be excluded
        ("b1.jpg", "BBB222", 1.0),
        ("b2.jpg", "BBB222", 51.0),  # 50s dwell - kept
    ]
    for name, _, _ in entries:
        cv2.imwrite(str(tmp_path / name), img)

    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(name for name, _, _ in entries)

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_profiling(tmp_path, min_dwell_time=10.0)

    stats = report["statistics"]
    assert stats["matched_crossings"] == 2
    assert stats["min_dwell_time"] == 10.0
    assert stats["matches_excluded_from_average"] == 1
    assert stats["average_dwell_time"] == pytest.approx(50.0)

    matches_by_plate = {m["plate_text"]: m for m in report["dwell_time_matches"]}
    assert matches_by_plate["AAA111"]["counted_in_average"] is False
    assert matches_by_plate["BBB222"]["counted_in_average"] is True
    # Excluded match is still fully present in the report, not dropped.
    assert matches_by_plate["AAA111"]["dwell_time"] == pytest.approx(2.0)


def test_normalize_input_dirs_wraps_a_single_path():
    assert normalize_input_dirs(Path("a")) == [Path("a")]
    assert normalize_input_dirs("a") == [Path("a")]


def test_normalize_input_dirs_passes_through_a_list():
    assert normalize_input_dirs([Path("a"), "b"]) == [Path("a"), Path("b")]


def test_run_plate_dwell_profiling_pools_separate_entry_and_exit_directories(
    tmp_path,
):
    # Entry and exit cameras are commonly two entirely separate
    # video_plateextractor.py runs, each with its own output directory and
    # its own plate_manifest.json - even reusing the same crop filename.
    import json

    import cv2
    import numpy as np
    from unittest.mock import patch

    from src.processing.plate_dwellprofiler import run_plate_dwell_profiling

    entrance = tmp_path / "entrance_cam"
    exit_cam = tmp_path / "exit_cam"
    entrance.mkdir()
    exit_cam.mkdir()

    img = np.zeros((50, 100, 3), dtype="uint8")
    cv2.imwrite(str(entrance / "frame1.jpg"), img)
    cv2.imwrite(str(exit_cam / "frame1.jpg"), img)

    (entrance / "plate_manifest.json").write_text(
        json.dumps({"frame1.jpg": {"timestamp": 100.0}})
    )
    (exit_cam / "plate_manifest.json").write_text(
        json.dumps({"frame1.jpg": {"timestamp": 160.0}})
    )

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_dwell_profiling([entrance, exit_cam])

    assert report["statistics"]["matched_crossings"] == 1
    match = report["dwell_time_matches"][0]
    assert match["entry_time"] == pytest.approx(100.0)
    assert match["exit_time"] == pytest.approx(160.0)
    assert match["dwell_time"] == pytest.approx(60.0)
    assert report["metadata"]["input_dirs"] == [str(entrance), str(exit_cam)]


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("ABC123", "ABC123", 0),
        ("", "", 0),
        ("", "ABC", 3),
        ("ABC123", "ABD123", 1),
        ("KITTEN", "SITTING", 3),
    ],
)
def test_levenshtein_distance(a, b, expected):
    assert levenshtein_distance(a, b) == expected


def test_compute_plate_dwell_times_default_is_exact_match_unchanged():
    # max_edit_distance=0 (the default) must behave identically to the
    # original exact-match-only grouping - readings 1 char apart stay split.
    detections = [
        PlateDetection("ABC123", 1.0, Path("a1.jpg")),
        PlateDetection("ABD123", 50.0, Path("a2.jpg")),
    ]

    matches, singles = compute_plate_dwell_times(detections)

    assert matches == []
    assert len(singles) == 2


def test_compute_plate_dwell_times_fuzzy_reunifies_misread_crossing():
    # Mirrors a real long-duration crossing read three different (but
    # similar) ways across its presence - exact matching would fracture this
    # into three single sightings.
    detections = [
        PlateDetection("381KE4", 0.0, Path("c1.jpg")),
        PlateDetection("381KE9", 4000.0, Path("c2.jpg")),
        PlateDetection("38IKE4", 11949.89, Path("c3.jpg")),
    ]

    matches, singles = compute_plate_dwell_times(detections, max_edit_distance=1)

    assert singles == []
    assert len(matches) == 1
    match = matches[0]
    assert match["num_sightings"] == 3
    assert match["dwell_time"] == pytest.approx(11949.89)
    assert match["entry_image"] == str(Path("c1.jpg"))
    assert match["exit_image"] == str(Path("c3.jpg"))
    assert set(match["plate_text_variants"]) == {"381KE4", "381KE9", "38IKE4"}


def test_compute_plate_dwell_times_fuzzy_chains_transitively():
    # A~B~C cluster into one crossing even though A and C aren't directly
    # within max_edit_distance of each other.
    detections = [
        PlateDetection("AAAAAA", 1.0, Path("a.jpg")),
        PlateDetection("AAAAAB", 2.0, Path("b.jpg")),
        PlateDetection("AAAABB", 3.0, Path("c.jpg")),
    ]

    matches, singles = compute_plate_dwell_times(detections, max_edit_distance=1)

    assert levenshtein_distance("AAAAAA", "AAAABB") == 2  # not directly within range
    assert singles == []
    assert len(matches) == 1
    assert matches[0]["num_sightings"] == 3


def test_compute_plate_dwell_times_fuzzy_picks_majority_vote_text():
    detections = [
        PlateDetection("ABC123", 1.0, Path("a.jpg")),
        PlateDetection("ABC123", 2.0, Path("b.jpg")),
        PlateDetection("ABD123", 3.0, Path("c.jpg")),
    ]

    matches, _ = compute_plate_dwell_times(detections, max_edit_distance=1)

    assert matches[0]["plate_text"] == "ABC123"


def test_compute_plate_dwell_times_max_time_gap_prevents_distant_merge():
    # Two different vehicles with similar plates, hours apart - should NOT
    # merge when max_time_gap is set tighter than their separation.
    detections = [
        PlateDetection("ABC123", 0.0, Path("a1.jpg")),
        PlateDetection("ABC123", 30.0, Path("a2.jpg")),
        PlateDetection("ABD123", 20000.0, Path("b1.jpg")),
        PlateDetection("ABD123", 20030.0, Path("b2.jpg")),
    ]

    unguarded, _ = compute_plate_dwell_times(detections, max_edit_distance=1)
    assert len(unguarded) == 1  # wrongly merged - the failure mode the guard prevents
    assert unguarded[0]["num_sightings"] == 4

    guarded, _ = compute_plate_dwell_times(
        detections, max_edit_distance=1, max_time_gap=3600.0
    )
    assert len(guarded) == 2
    dwell_times = sorted(m["dwell_time"] for m in guarded)
    assert dwell_times == [pytest.approx(30.0), pytest.approx(30.0)]


def test_compute_plate_dwell_times_fuzzy_single_sighting_reported_separately():
    detections = [PlateDetection("ABC123", 100.0, Path("only.jpg"))]

    matches, singles = compute_plate_dwell_times(detections, max_edit_distance=1)

    assert matches == []
    assert len(singles) == 1
