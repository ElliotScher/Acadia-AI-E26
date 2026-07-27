import json
from unittest.mock import patch

import pytest

from src.utility.vehicle_direction_benchmark import (
    NO_DETECTION,
    find_labeled_images,
    ground_truth_label_from_filename,
    render_html_report,
    run_vehicle_direction_benchmark,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("entity_1_right_car.jpg", "right"),  # suffix position
        ("left_entity_2.jpg", "left"),  # prefix position
        ("entity_3_left_car.jpg", "left"),  # middle position
        ("IMG_RIGHT.JPG", "right"),
        # Unlike pedestrians, cars have no front/back concept - these must
        # NOT be recognized as valid ground truth even though the token
        # itself would be recognized for a pedestrian.
        ("subject_front.jpg", None),
        ("subject_back.jpg", None),
        ("subject_unknown.jpg", None),
        ("no_direction_here.jpg", None),
        ("leftover_notes.jpg", None),  # "left" as a substring, not a token
        ("entity_left_and_right.jpg", None),  # ambiguous
    ],
)
def test_ground_truth_label_from_filename(filename, expected):
    assert ground_truth_label_from_filename(filename) == expected


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_labeled_images_recurses_and_skips_unrecognized(tmp_path):
    _touch(tmp_path / "entity_1_left_car.jpg")
    _touch(tmp_path / "sub" / "entity_2_right_car.png")
    _touch(tmp_path / "unlabeled.jpg")
    _touch(tmp_path / "entity_3_front_car.jpg")  # not a valid vehicle direction
    _touch(tmp_path / "notes.txt")  # not an image

    labeled = find_labeled_images(tmp_path)

    assert {p.name: label for p, label in labeled.items()} == {
        "entity_1_left_car.jpg": "left",
        "entity_2_right_car.png": "right",
    }


def test_run_vehicle_direction_benchmark_scores_matches(tmp_path):
    _touch(tmp_path / "entity_1_left_car.jpg")
    _touch(tmp_path / "entity_2_right_car.jpg")
    _touch(tmp_path / "entity_3_right_car.jpg")
    _touch(tmp_path / "unlabeled.jpg")

    # sorted labeled paths: entity_1_left_car.jpg (gt=left),
    # entity_2_right_car.jpg (gt=right), entity_3_right_car.jpg (gt=right)
    predictions = iter(["left", "right", "unknown"])
    with patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        side_effect=lambda *args, **kwargs: next(predictions),
    ):
        report = run_vehicle_direction_benchmark(tmp_path, model=object())

    stats = report["statistics"]
    assert stats["total_images_scanned"] == 4
    assert stats["skipped_unlabeled"] == 1
    assert stats["images_evaluated"] == 3
    assert stats["exact_matches"] == 2
    assert stats["accuracy"] == pytest.approx(2 / 3)

    results_by_image = {r["image"]: r for r in report["results"]}
    assert results_by_image["entity_1_left_car.jpg"]["exact_match"] is True
    assert results_by_image["entity_2_right_car.jpg"]["predicted"] == "right"
    assert results_by_image["entity_3_right_car.jpg"]["predicted"] == "unknown"

    # No "front"/"back" bucket should appear anywhere - only left/right/
    # no_detection/unknown are meaningful for vehicles.
    assert set(report["confusion_matrix"].keys()) <= {"left", "right"}


def test_run_vehicle_direction_benchmark_handles_no_detection(tmp_path):
    _touch(tmp_path / "entity_1_left_car.jpg")

    with patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        return_value=NO_DETECTION,
    ):
        report = run_vehicle_direction_benchmark(tmp_path, model=object())

    assert report["statistics"]["exact_matches"] == 0
    assert report["confusion_matrix"]["left"] == {NO_DETECTION: 1}


def test_run_vehicle_direction_benchmark_handles_unknown_prediction_bar_chart(
    tmp_path,
):
    # Regression test: "unknown" is a real prediction outcome (too few
    # confident keypoints) even though it's never valid ground truth for a
    # vehicle - render_direction_bar_chart/compute_label_counts must not
    # KeyError when a prediction lands outside VALID_LABELS + NO_DETECTION.
    _touch(tmp_path / "entity_1_left_car.jpg")
    bar_chart_path = tmp_path / "chart.png"

    with patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        return_value="unknown",
    ):
        report = run_vehicle_direction_benchmark(
            tmp_path, model=object(), bar_chart=bar_chart_path
        )

    assert report["confusion_matrix"]["left"] == {"unknown": 1}
    assert bar_chart_path.exists()
    assert bar_chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_run_vehicle_direction_benchmark_auto_fetches_weights_when_no_model_given(
    tmp_path,
):
    _touch(tmp_path / "entity_1_left_car.jpg")

    with patch(
        "src.utility.vehicle_direction_benchmark.fetch_vehicle_pose_weights",
        return_value=tmp_path / "fetched.pt",
    ) as mock_fetch, patch(
        "src.utility.vehicle_direction_benchmark.load_model"
    ) as mock_load_model, patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        return_value="left",
    ):
        report = run_vehicle_direction_benchmark(tmp_path)

    mock_fetch.assert_called_once()
    mock_load_model.assert_called_once_with(str(tmp_path / "fetched.pt"))
    assert report["metadata"]["model_name"] == str(tmp_path / "fetched.pt")


def test_run_vehicle_direction_benchmark_writes_report_html_and_bar_chart(tmp_path):
    _touch(tmp_path / "entity_1_left_car.jpg")
    report_path = tmp_path / "report.json"
    html_path = tmp_path / "summary.html"
    bar_chart_path = tmp_path / "chart.png"

    with patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        return_value="left",
    ):
        run_vehicle_direction_benchmark(
            tmp_path,
            model=object(),
            report=report_path,
            html_report=html_path,
            bar_chart=bar_chart_path,
        )

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["statistics"]["exact_matches"] == 1

    assert bar_chart_path.exists()
    assert bar_chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    assert html_path.exists()
    html_content = html_path.read_text()
    assert "<table" in html_content
    assert "data:image/png;base64," in html_content


def test_render_html_report_contains_vehicle_title_and_tables(tmp_path):
    _touch(tmp_path / "entity_1_left_car.jpg")
    _touch(tmp_path / "entity_2_right_car.jpg")

    predictions = iter(["left", "left"])
    with patch(
        "src.utility.vehicle_direction_benchmark.predict_direction",
        side_effect=lambda *args, **kwargs: next(predictions),
    ):
        report = run_vehicle_direction_benchmark(tmp_path, model=object())

    doc = render_html_report(report)

    assert doc.count("<table") == 3  # overview, per-direction metrics, confusion
    assert doc.count("</table>") == 3
    assert "Vehicle Direction Benchmark" in doc
    assert "50.00% (1/2)" in doc
