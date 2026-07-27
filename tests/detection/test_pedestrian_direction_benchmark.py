import json
from unittest.mock import patch

import pytest

from src.utility.pedestrian_direction_benchmark import (
    NO_DETECTION,
    build_confusion_matrix,
    compute_label_counts,
    compute_label_metrics,
    find_labeled_images,
    ground_truth_label_from_filename,
    render_direction_bar_chart,
    render_html_report,
    run_pedestrian_direction_benchmark,
)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("car_017_left.jpg", "left"),  # suffix position
        ("left_car_017.png", "left"),  # prefix position
        ("cam2_left_017.jpg", "left"),  # middle position
        ("car_017_right.png", "right"),
        ("person_front.jpeg", "front"),
        ("person_back.bmp", "back"),
        ("subject_unknown.jpg", "unknown"),
        ("IMG_RIGHT.JPG", "right"),
        ("no_direction_here.jpg", None),
        ("onlyleft.jpg", None),  # not a delimited token, just a substring
        ("leftover_notes.jpg", None),  # "left" as a substring, not a token
        ("background_check.jpg", None),  # "back" as a substring, not a token
        ("car_left_and_right.jpg", None),  # ambiguous - two distinct labels
        ("noext", None),
    ],
)
def test_ground_truth_label_from_filename(filename, expected):
    assert ground_truth_label_from_filename(filename) == expected


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_find_labeled_images_recurses_and_skips_unrecognized(tmp_path):
    _touch(tmp_path / "a_left.jpg")
    _touch(tmp_path / "sub" / "b_right.png")
    _touch(tmp_path / "unlabeled.jpg")
    _touch(tmp_path / "c_sideways.jpg")  # not a recognized direction
    _touch(tmp_path / "notes.txt")  # not an image

    labeled = find_labeled_images(tmp_path)

    assert {p.name: label for p, label in labeled.items()} == {
        "a_left.jpg": "left",
        "b_right.png": "right",
    }


def test_build_confusion_matrix_and_label_metrics():
    results = [
        {"ground_truth": "left", "predicted": "left"},
        {"ground_truth": "left", "predicted": "right"},
        {"ground_truth": "right", "predicted": "right"},
        {"ground_truth": "right", "predicted": "right"},
        {"ground_truth": "front", "predicted": NO_DETECTION},
    ]

    matrix = build_confusion_matrix(results)
    assert matrix["left"] == {"left": 1, "right": 1}
    assert matrix["right"] == {"right": 2}
    assert matrix["front"] == {NO_DETECTION: 1}

    metrics = compute_label_metrics(matrix)

    assert metrics["left"]["support"] == 2
    assert metrics["left"]["true_positives"] == 1
    assert metrics["left"]["recall"] == pytest.approx(0.5)
    # "left" was predicted once total, and it was correct -> precision 1.0
    assert metrics["left"]["precision"] == pytest.approx(1.0)

    # "right" predicted 3 times total (1 wrong from "left" + 2 correct)
    assert metrics["right"]["support"] == 2
    assert metrics["right"]["true_positives"] == 2
    assert metrics["right"]["recall"] == pytest.approx(1.0)
    assert metrics["right"]["precision"] == pytest.approx(2 / 3)

    assert metrics["front"]["support"] == 1
    assert metrics["front"]["true_positives"] == 0
    assert metrics["front"]["recall"] == pytest.approx(0.0)
    assert metrics["front"]["f1"] == pytest.approx(0.0)


def test_run_pedestrian_direction_benchmark_scores_matches(tmp_path):
    _touch(tmp_path / "a_left.jpg")
    _touch(tmp_path / "b_right.jpg")
    _touch(tmp_path / "c_front.jpg")
    _touch(tmp_path / "unlabeled.jpg")

    # sorted labeled paths: a_left.jpg, b_right.jpg, c_front.jpg
    predictions = iter(["left", "left", "unknown"])
    with patch(
        "src.utility.pedestrian_direction_benchmark.predict_direction",
        side_effect=lambda *args, **kwargs: next(predictions),
    ):
        report = run_pedestrian_direction_benchmark(tmp_path, model=object())

    stats = report["statistics"]
    assert stats["total_images_scanned"] == 4
    assert stats["skipped_unlabeled"] == 1
    assert stats["images_evaluated"] == 3
    assert stats["exact_matches"] == 1
    assert stats["accuracy"] == pytest.approx(1 / 3)

    results_by_image = {r["image"]: r for r in report["results"]}
    assert results_by_image["a_left.jpg"]["exact_match"] is True
    assert results_by_image["b_right.jpg"]["predicted"] == "left"
    assert results_by_image["c_front.jpg"]["predicted"] == "unknown"


def test_run_pedestrian_direction_benchmark_handles_no_detection(tmp_path):
    _touch(tmp_path / "a_left.jpg")

    with patch(
        "src.utility.pedestrian_direction_benchmark.predict_direction",
        return_value=NO_DETECTION,
    ):
        report = run_pedestrian_direction_benchmark(tmp_path, model=object())

    assert report["statistics"]["exact_matches"] == 0
    assert report["confusion_matrix"]["left"] == {NO_DETECTION: 1}
    assert NO_DETECTION in report["prediction_labels"]


def test_run_pedestrian_direction_benchmark_writes_report_and_html(tmp_path):
    _touch(tmp_path / "a_left.jpg")
    report_path = tmp_path / "report.json"
    html_path = tmp_path / "summary.html"

    with patch(
        "src.utility.pedestrian_direction_benchmark.predict_direction",
        return_value="left",
    ):
        run_pedestrian_direction_benchmark(
            tmp_path, model=object(), report=report_path, html_report=html_path
        )

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["statistics"]["exact_matches"] == 1

    assert html_path.exists()
    html_content = html_path.read_text()
    assert "<table" in html_content
    assert "100.00% (1/1)" in html_content
    # Bar chart embedded as a base64 PNG <img>, not just the tables.
    assert "data:image/png;base64," in html_content


def test_compute_label_counts_covers_every_bucket_with_zeros(tmp_path):
    results = [
        {"ground_truth": "left", "predicted": "left"},
        {"ground_truth": "left", "predicted": "right"},
        {"ground_truth": "front", "predicted": NO_DETECTION},
    ]

    gt_counts, pred_counts = compute_label_counts(results)

    assert gt_counts == {
        "left": 2,
        "right": 0,
        "front": 1,
        "back": 0,
        "unknown": 0,
        NO_DETECTION: 0,
    }
    assert pred_counts == {
        "left": 1,
        "right": 1,
        "front": 0,
        "back": 0,
        "unknown": 0,
        NO_DETECTION: 1,
    }


def test_render_direction_bar_chart_produces_png_bytes():
    results = [
        {"ground_truth": "left", "predicted": "left"},
        {"ground_truth": "right", "predicted": NO_DETECTION},
    ]

    png_bytes = render_direction_bar_chart(results)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_direction_bar_chart_empty_results_returns_empty_bytes():
    assert render_direction_bar_chart([]) == b""


def test_run_pedestrian_direction_benchmark_writes_bar_chart_png(tmp_path):
    _touch(tmp_path / "a_left.jpg")
    bar_chart_path = tmp_path / "chart.png"

    with patch(
        "src.utility.pedestrian_direction_benchmark.predict_direction",
        return_value="left",
    ):
        run_pedestrian_direction_benchmark(tmp_path, model=object(), bar_chart=bar_chart_path)

    assert bar_chart_path.exists()
    assert bar_chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_html_report_contains_confusion_matrix(tmp_path):
    _touch(tmp_path / "a_left.jpg")
    _touch(tmp_path / "b_right.jpg")

    predictions = iter(["left", "left"])
    with patch(
        "src.utility.pedestrian_direction_benchmark.predict_direction",
        side_effect=lambda *args, **kwargs: next(predictions),
    ):
        report = run_pedestrian_direction_benchmark(tmp_path, model=object())

    doc = render_html_report(report)

    assert doc.count("<table") == 3  # overview, per-direction metrics, confusion
    assert doc.count("</table>") == 3
    assert "Pedestrian/Bicycle Direction Benchmark" in doc
    assert "50.00% (1/2)" in doc
