import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utility.geometryutils import Rectangle
from src.utility.vehicle_speed_benchmark import (
    GroundTruth,
    GroundTruthCar,
    VehicleTrack,
    compute_speed_accuracy,
    find_video_file,
    load_ground_truth,
    match_tracks_to_ground_truth,
    render_html_report,
    run_vehicle_speed_benchmark,
    track_vehicles,
)

# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------


def test_load_ground_truth_parses_pkl(tmp_path):
    gt_path = tmp_path / "gt_data.pkl"
    raw = {
        "fps": 50.0,
        "cars": [
            {
                "carId": 1,
                "valid": True,
                "speed": 42.5,
                "intersections": [{"videoTime": 3.0}, {"videoTime": 5.0}],
            },
            {
                "carId": 2,
                "valid": False,
                "speed": 30.0,
                "intersections": [
                    {"videoTime": 6.0},
                    {"videoTime": 6.4},
                    {"videoTime": 6.9},
                ],
            },
        ],
    }
    with open(gt_path, "wb") as f:
        pickle.dump(raw, f)

    ground_truth = load_ground_truth(gt_path)

    assert ground_truth.fps == pytest.approx(50.0)
    assert len(ground_truth.cars) == 2

    car1 = ground_truth.cars[0]
    assert car1.car_id == 1
    assert car1.valid is True
    assert car1.speed_kmh == pytest.approx(42.5)
    assert car1.first_crossing_time == pytest.approx(3.0)
    assert car1.last_crossing_time == pytest.approx(5.0)

    car2 = ground_truth.cars[1]
    assert car2.valid is False
    # last_crossing_time is the LAST intersection, not the second one.
    assert car2.last_crossing_time == pytest.approx(6.9)


def test_load_ground_truth_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ground_truth(tmp_path / "missing.pkl")


# ---------------------------------------------------------------------------
# find_video_file
# ---------------------------------------------------------------------------


def test_find_video_file_autodetects_single_video(tmp_path):
    (tmp_path / "video.avi").write_bytes(b"")
    (tmp_path / "gt_data.pkl").write_bytes(b"")
    (tmp_path / "screen.png").write_bytes(b"")

    assert find_video_file(tmp_path) == tmp_path / "video.avi"


def test_find_video_file_raises_when_none_found(tmp_path):
    (tmp_path / "gt_data.pkl").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_video_file(tmp_path)


def test_find_video_file_raises_when_multiple_found(tmp_path):
    (tmp_path / "video.avi").write_bytes(b"")
    (tmp_path / "video2.mp4").write_bytes(b"")
    with pytest.raises(ValueError):
        find_video_file(tmp_path)


# ---------------------------------------------------------------------------
# track_vehicles
# ---------------------------------------------------------------------------


def test_track_vehicles_computes_pixel_speed():
    # A vehicle box moving 10px/frame to the right across 5 frames.
    frame_w, frame_h = 300, 100
    box_w, box_h = 40, 30
    box_y = 20
    fps = 10.0

    boxes_per_frame = [Rectangle(x=i * 10, y=box_y, w=box_w, h=box_h) for i in range(5)]

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = fps
    mock_cap.read.side_effect = [
        (True, np.zeros((frame_h, frame_w, 3), dtype=np.uint8)) for _ in range(5)
    ] + [(False, None)]

    predict_results = []
    for box in boxes_per_frame:
        mock_box = MagicMock()
        mock_box.xyxy = [[box.x, box.y, box.x + box.w, box.y + box.h]]
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        predict_results.append([mock_result])

    mock_model = MagicMock()
    mock_model.predict.side_effect = predict_results

    with patch(
        "src.utility.vehicle_speed_benchmark.cv2.VideoCapture", return_value=mock_cap
    ):
        tracks, returned_fps = track_vehicles(Path("dummy.avi"), mock_model)

    assert returned_fps == pytest.approx(fps)
    assert len(tracks) == 1
    track = tracks[0]
    assert track.track_id == 1
    assert track.first_frame_idx == 1
    assert track.last_frame_idx == 5
    # displacement = 40px (center x=20 to center x=60) over 4 frames @ 10fps = 0.4s
    assert track.pixel_speed == pytest.approx(100.0, rel=0.05)


def test_track_vehicles_raises_when_video_cannot_be_opened():
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch(
        "src.utility.vehicle_speed_benchmark.cv2.VideoCapture", return_value=mock_cap
    ):
        with pytest.raises(RuntimeError):
            track_vehicles(Path("dummy.avi"), MagicMock())


# ---------------------------------------------------------------------------
# match_tracks_to_ground_truth
# ---------------------------------------------------------------------------


def _track(track_id, last_frame_idx, first_frame_idx=0, pixel_speed=10.0):
    box = Rectangle(0, 0, 10, 10)
    return VehicleTrack(
        track_id=track_id,
        first_frame_idx=first_frame_idx,
        last_frame_idx=last_frame_idx,
        first_box=box,
        last_box=box,
        pixel_speed=pixel_speed,
    )


def _gt_car(
    car_id, last_crossing_time, valid=True, speed_kmh=50.0, first_crossing_time=None
):
    return GroundTruthCar(
        car_id=car_id,
        valid=valid,
        speed_kmh=speed_kmh,
        first_crossing_time=(
            first_crossing_time
            if first_crossing_time is not None
            else last_crossing_time - 1.0
        ),
        last_crossing_time=last_crossing_time,
    )


def test_match_tracks_to_ground_truth_matches_nearest_within_tolerance():
    fps = 10.0
    tracks = [_track(1, last_frame_idx=100), _track(2, last_frame_idx=200)]
    # track 1 last-seen time = 10.0s, track 2 = 20.0s
    ground_truth = GroundTruth(
        fps=fps,
        cars=[_gt_car(1, last_crossing_time=10.2), _gt_car(2, last_crossing_time=19.8)],
    )

    matches, unmatched = match_tracks_to_ground_truth(
        tracks, ground_truth, fps, max_time_diff=1.0
    )

    assert unmatched == 0
    matched_by_car = {m.car_id: m.track_id for m in matches}
    assert matched_by_car == {1: 1, 2: 2}


def test_match_tracks_to_ground_truth_drops_out_of_tolerance():
    fps = 10.0
    tracks = [_track(1, last_frame_idx=100)]  # last-seen time = 10.0s
    ground_truth = GroundTruth(fps=fps, cars=[_gt_car(1, last_crossing_time=50.0)])

    matches, unmatched = match_tracks_to_ground_truth(
        tracks, ground_truth, fps, max_time_diff=1.0
    )

    assert matches == []
    assert unmatched == 1


def test_match_tracks_to_ground_truth_track_matched_at_most_once():
    fps = 10.0
    tracks = [_track(1, last_frame_idx=100)]  # last-seen time = 10.0s
    ground_truth = GroundTruth(
        fps=fps,
        cars=[
            _gt_car(1, last_crossing_time=10.05),
            _gt_car(2, last_crossing_time=10.5),
        ],
    )

    matches, unmatched = match_tracks_to_ground_truth(
        tracks, ground_truth, fps, max_time_diff=1.0
    )

    assert len(matches) == 1
    assert matches[0].car_id == 1
    assert matches[0].track_id == 1
    assert unmatched == 1


# ---------------------------------------------------------------------------
# compute_speed_accuracy
# ---------------------------------------------------------------------------


def test_compute_speed_accuracy_buckets_and_reasonable_rate():
    comparisons = [
        {"abs_error_kmh": 1.0, "relative_error": 0.02},  # <=5%
        {"abs_error_kmh": 5.0, "relative_error": 0.15},  # <=20%
        {"abs_error_kmh": 10.0, "relative_error": 0.35},  # <=50%
        {"abs_error_kmh": 20.0, "relative_error": 0.75},  # >50%
    ]

    accuracy = compute_speed_accuracy(comparisons, tolerance=0.2)

    assert accuracy["total_scored"] == 4
    assert accuracy["histogram"]["<=5%"]["count"] == 1
    assert accuracy["histogram"]["<=20%"]["count"] == 1
    assert accuracy["histogram"]["<=50%"]["count"] == 1
    assert accuracy["histogram"][">50%"]["count"] == 1
    assert accuracy["mean_abs_error_kmh"] == pytest.approx(9.0)
    assert accuracy["median_abs_error_kmh"] == pytest.approx(7.5)
    assert accuracy["rmse_kmh"] == pytest.approx(11.4673, rel=1e-3)
    assert accuracy["mean_relative_error"] == pytest.approx(0.3175)
    # Reasonable (<=20%) = 0.02 and 0.15 -> 2/4
    assert accuracy["reasonable_count"] == 2
    assert accuracy["reasonable_rate"] == pytest.approx(0.5)


def test_compute_speed_accuracy_handles_none_relative_error():
    comparisons = [{"abs_error_kmh": 3.0, "relative_error": None}]

    accuracy = compute_speed_accuracy(comparisons)

    assert accuracy["total_scored"] == 1
    assert sum(b["count"] for b in accuracy["histogram"].values()) == 0
    assert accuracy["reasonable_rate"] == 0.0


def test_compute_speed_accuracy_empty_comparisons():
    accuracy = compute_speed_accuracy([])

    assert accuracy["total_scored"] == 0
    assert accuracy["mean_abs_error_kmh"] == 0.0
    assert accuracy["reasonable_rate"] == 0.0


# ---------------------------------------------------------------------------
# run_vehicle_speed_benchmark
# ---------------------------------------------------------------------------


def _placeholder_track(track_id, pixel_speed, last_frame_idx):
    return VehicleTrack(
        track_id=track_id,
        first_frame_idx=0,
        last_frame_idx=last_frame_idx,
        first_box=Rectangle(0, 0, 10, 10),
        last_box=Rectangle(50, 0, 10, 10),
        pixel_speed=pixel_speed,
    )


def test_run_vehicle_speed_benchmark_scores_against_reference(tmp_path):
    fps = 10.0
    tracks = [
        _placeholder_track(
            1, pixel_speed=50.0, last_frame_idx=100
        ),  # last-seen @ 10.0s
        _placeholder_track(
            2, pixel_speed=100.0, last_frame_idx=150
        ),  # last-seen @ 15.0s
    ]
    ground_truth = GroundTruth(
        fps=fps,
        cars=[
            _gt_car(1, last_crossing_time=10.0, speed_kmh=40.0),
            _gt_car(2, last_crossing_time=15.0, speed_kmh=80.0),
        ],
    )

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, fps),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        report = run_vehicle_speed_benchmark(
            recording_dir=tmp_path,
            video_path=tmp_path / "video.avi",
            gt_path=tmp_path / "gt_data.pkl",
            model=object(),
        )

    # Reference defaults to the lowest-numbered valid matched car -> car 1
    # (track 1), which must be excluded from scoring.
    assert report["reference"]["car_id"] == 1
    assert report["reference"]["track_id"] == 1
    assert report["statistics"]["matched_cars"] == 2
    assert len(report["comparisons"]) == 1

    comparison = report["comparisons"][0]
    assert comparison["car_id"] == 2
    # max pixel_speed=100 (track 2) -> relative speeds: track1=0.5, track2=1.0.
    # track1 (relative=0.5) calibrated to its true 40 km/h -> scale=80.
    # track2's predicted speed = 1.0 * 80 = 80 km/h, exactly matching its own
    # ground truth by construction of this fixture.
    assert comparison["predicted_speed_kmh"] == pytest.approx(80.0)
    assert comparison["abs_error_kmh"] == pytest.approx(0.0, abs=1e-6)


def test_run_vehicle_speed_benchmark_reference_car_id_selects_explicitly(tmp_path):
    fps = 10.0
    tracks = [
        _placeholder_track(1, pixel_speed=50.0, last_frame_idx=100),
        _placeholder_track(2, pixel_speed=100.0, last_frame_idx=150),
    ]
    ground_truth = GroundTruth(
        fps=fps,
        cars=[
            _gt_car(1, last_crossing_time=10.0, speed_kmh=40.0),
            _gt_car(2, last_crossing_time=15.0, speed_kmh=80.0),
        ],
    )

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, fps),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        report = run_vehicle_speed_benchmark(
            recording_dir=tmp_path,
            video_path=tmp_path / "video.avi",
            gt_path=tmp_path / "gt_data.pkl",
            model=object(),
            reference_car_id=2,
        )

    assert report["reference"]["car_id"] == 2
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["car_id"] == 1


def test_run_vehicle_speed_benchmark_unknown_reference_car_id_raises(tmp_path):
    tracks = [_placeholder_track(1, pixel_speed=50.0, last_frame_idx=100)]
    ground_truth = GroundTruth(fps=10.0, cars=[_gt_car(1, last_crossing_time=10.0)])

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, 10.0),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        with pytest.raises(ValueError):
            run_vehicle_speed_benchmark(
                recording_dir=tmp_path,
                video_path=tmp_path / "video.avi",
                gt_path=tmp_path / "gt_data.pkl",
                model=object(),
                reference_car_id=99,
            )


def test_run_vehicle_speed_benchmark_raises_when_no_valid_matches(tmp_path):
    tracks = [_placeholder_track(1, pixel_speed=50.0, last_frame_idx=100)]
    ground_truth = GroundTruth(
        fps=10.0, cars=[_gt_car(1, last_crossing_time=10.0, valid=False)]
    )

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, 10.0),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        with pytest.raises(ValueError):
            run_vehicle_speed_benchmark(
                recording_dir=tmp_path,
                video_path=tmp_path / "video.avi",
                gt_path=tmp_path / "gt_data.pkl",
                model=object(),
            )


def test_run_vehicle_speed_benchmark_writes_report_html_and_chart(tmp_path):
    fps = 10.0
    tracks = [
        _placeholder_track(1, pixel_speed=50.0, last_frame_idx=100),
        _placeholder_track(2, pixel_speed=100.0, last_frame_idx=150),
    ]
    ground_truth = GroundTruth(
        fps=fps,
        cars=[
            _gt_car(1, last_crossing_time=10.0, speed_kmh=40.0),
            _gt_car(2, last_crossing_time=15.0, speed_kmh=80.0),
        ],
    )
    report_path = tmp_path / "report.json"
    html_path = tmp_path / "summary.html"
    chart_path = tmp_path / "chart.png"

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, fps),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        run_vehicle_speed_benchmark(
            recording_dir=tmp_path,
            video_path=tmp_path / "video.avi",
            gt_path=tmp_path / "gt_data.pkl",
            model=object(),
            reference_car_id=2,
            report=report_path,
            html_report=html_path,
            chart=chart_path,
        )

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["reference"]["car_id"] == 2
    assert len(saved["comparisons"]) == 1
    assert saved["comparisons"][0]["car_id"] == 1

    assert chart_path.exists()
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert html_path.exists()
    html_content = html_path.read_text()
    assert "<table" in html_content
    assert "Vehicle Speed Benchmark" in html_content
    assert "data:image/png;base64," in html_content


def test_render_html_report_contains_title_and_tables(tmp_path):
    fps = 10.0
    tracks = [
        _placeholder_track(1, pixel_speed=50.0, last_frame_idx=100),
        _placeholder_track(2, pixel_speed=100.0, last_frame_idx=150),
    ]
    ground_truth = GroundTruth(
        fps=fps,
        cars=[
            _gt_car(1, last_crossing_time=10.0, speed_kmh=40.0),
            _gt_car(2, last_crossing_time=15.0, speed_kmh=80.0),
        ],
    )

    with (
        patch(
            "src.utility.vehicle_speed_benchmark.track_vehicles",
            return_value=(tracks, fps),
        ),
        patch(
            "src.utility.vehicle_speed_benchmark.load_ground_truth",
            return_value=ground_truth,
        ),
    ):
        report = run_vehicle_speed_benchmark(
            recording_dir=tmp_path,
            video_path=tmp_path / "video.avi",
            gt_path=tmp_path / "gt_data.pkl",
            model=object(),
        )

    doc = render_html_report(report)

    assert "Vehicle Speed Benchmark" in doc
    assert doc.count("<table") == 3  # overview, accuracy, scored vehicles
    assert doc.count("</table>") == 3
