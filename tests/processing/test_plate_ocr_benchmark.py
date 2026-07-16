import json
from unittest.mock import patch

import numpy as np
import cv2
import pytest

from src.utility.plate_ocr_benchmark import (
    levenshtein_distance,
    load_ground_truth,
    load_source_videos,
    render_html_report,
    run_plate_ocr_benchmark,
    summarize_miss_distances,
    video_group_for,
)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("ABC123", "ABC123", 0),
        ("", "", 0),
        ("", "ABC", 3),
        ("ABC", "", 3),
        ("ABC123", "ABD123", 1),
        ("KITTEN", "SITTING", 3),
    ],
)
def test_levenshtein_distance(a, b, expected):
    assert levenshtein_distance(a, b) == expected


def test_load_ground_truth_excludes_unreadable_and_unfinished(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "a.jpg": "ABC123",
                "b.jpg": "",
                "c.jpg": None,
                "d.jpg": "XYZ999",
            }
        )
    )

    loaded = load_ground_truth(gt_path)

    assert loaded == {"a.jpg": "ABC123", "d.jpg": "XYZ999"}


def _write_dummy_image(path):
    cv2.imwrite(str(path), np.zeros((50, 100, 3), dtype="uint8"))


def test_run_plate_ocr_benchmark_scores_matches_and_char_accuracy(tmp_path):
    _write_dummy_image(tmp_path / "a.jpg")
    _write_dummy_image(tmp_path / "b.jpg")
    _write_dummy_image(tmp_path / "e.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "a.jpg": "ABC123",  # OCR reads correctly
                "b.jpg": "ABD123",  # OCR misreads one character
                "c.jpg": "",  # unreadable - excluded from scoring
                "d.jpg": None,  # never finished - excluded from scoring
                "e.jpg": "XYZ999",  # OCR fails to read anything
            }
        )
    )

    # sorted eligible keys are a.jpg, b.jpg, e.jpg - predictions line up in that order
    predictions = iter(["ABC123", "ABC123", None])

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        side_effect=lambda crop: next(predictions),
    ):
        report = run_plate_ocr_benchmark(gt_path)

    stats = report["statistics"]
    assert stats["images_evaluated"] == 3
    assert stats["unique_plates_seen"] == 3
    assert stats["exact_matches"] == 1
    assert stats["match_rate"] == pytest.approx(1 / 3)
    # total edit distance = 0 + 1 + 6 = 7 over 18 ground-truth characters
    assert stats["char_accuracy"] == pytest.approx(1 - 7 / 18)
    # c.jpg ("") and d.jpg (null) are thrown out before OCR ever runs
    assert stats["ground_truth_entries"] == 5
    assert stats["thrown_out_unusable_ground_truth"] == 2
    assert stats["images_thrown_out"] == 2

    results_by_image = {r["image"]: r for r in report["results"]}
    assert results_by_image["a.jpg"]["exact_match"] is True
    assert results_by_image["b.jpg"]["char_errors"] == 1
    assert results_by_image["e.jpg"]["predicted"] == ""
    assert results_by_image["e.jpg"]["char_errors"] == 6


def test_run_plate_ocr_benchmark_skips_missing_images(tmp_path):
    _write_dummy_image(tmp_path / "exists.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"exists.jpg": "ABC123", "missing.jpg": "DEF456"}))

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    stats = report["statistics"]
    assert stats["images_evaluated"] == 1
    assert stats["skipped_missing_images"] == 1
    assert stats["match_rate"] == pytest.approx(1.0)
    assert stats["images_thrown_out"] == 1
    assert stats["thrown_out_unusable_ground_truth"] == 0


def test_run_plate_ocr_benchmark_thrown_out_combines_both_categories(tmp_path):
    _write_dummy_image(tmp_path / "exists.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "exists.jpg": "ABC123",  # evaluated
                "missing.jpg": "DEF456",  # thrown out: file missing
                "unreadable.jpg": "",  # thrown out: unusable ground truth
                "pending.jpg": None,  # thrown out: unusable ground truth
            }
        )
    )

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    stats = report["statistics"]
    assert stats["ground_truth_entries"] == 4
    assert stats["images_evaluated"] == 1
    assert stats["skipped_missing_images"] == 1
    assert stats["thrown_out_unusable_ground_truth"] == 2
    assert stats["images_thrown_out"] == 3


def test_run_plate_ocr_benchmark_counts_unique_plates_across_repeat_sightings(
    tmp_path,
):
    # Same physical plate photographed at entry and exit - two images, one plate.
    _write_dummy_image(tmp_path / "entry.jpg")
    _write_dummy_image(tmp_path / "exit.jpg")
    _write_dummy_image(tmp_path / "other.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "entry.jpg": "ABC123",
                "exit.jpg": "ABC123",
                "other.jpg": "XYZ999",
            }
        )
    )

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    stats = report["statistics"]
    assert stats["images_evaluated"] == 3
    assert stats["unique_plates_seen"] == 2


def test_run_plate_ocr_benchmark_writes_report(tmp_path):
    _write_dummy_image(tmp_path / "a.jpg")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"a.jpg": "ABC123"}))
    report_path = tmp_path / "report.json"

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        run_plate_ocr_benchmark(gt_path, report=report_path)

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["statistics"]["exact_matches"] == 1


def test_run_plate_ocr_benchmark_defaults_input_dir_to_ground_truth_parent(tmp_path):
    nested = tmp_path / "crops"
    nested.mkdir()
    _write_dummy_image(nested / "a.jpg")
    gt_path = nested / "ground_truth.json"
    gt_path.write_text(json.dumps({"a.jpg": "ABC123"}))

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    assert report["statistics"]["images_evaluated"] == 1
    assert report["metadata"]["input_dir"] == str(nested)


def _result(exact_match, char_errors):
    return {"exact_match": exact_match, "char_errors": char_errors}


def test_summarize_miss_distances_buckets_and_stats():
    results = [
        _result(True, 0),
        _result(False, 1),
        _result(False, 1),
        _result(False, 2),
        _result(False, 6),
    ]

    summary = summarize_miss_distances(results)

    assert summary["num_misses"] == 4
    assert summary["mean_edit_distance"] == pytest.approx((1 + 1 + 2 + 6) / 4)
    assert summary["median_edit_distance"] == pytest.approx(1.5)
    assert summary["histogram"]["1"] == {"count": 2, "share": pytest.approx(0.5)}
    assert summary["histogram"]["2"] == {"count": 1, "share": pytest.approx(0.25)}
    assert summary["histogram"]["3"] == {"count": 0, "share": pytest.approx(0.0)}
    assert summary["histogram"]["4+"] == {"count": 1, "share": pytest.approx(0.25)}


def test_summarize_miss_distances_no_misses():
    results = [_result(True, 0), _result(True, 0)]

    summary = summarize_miss_distances(results)

    assert summary["num_misses"] == 0
    assert summary["mean_edit_distance"] == 0.0
    assert summary["median_edit_distance"] == 0.0
    assert all(bucket["count"] == 0 for bucket in summary["histogram"].values())


def test_run_plate_ocr_benchmark_includes_miss_distance_distribution(tmp_path):
    for name in ["a", "b", "c"]:
        _write_dummy_image(tmp_path / f"{name}.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "a.jpg": "ABC123",  # exact match
                "b.jpg": "ABD123",  # 1 char off
                "c.jpg": "XYZ999",  # completely different -> big miss
            }
        )
    )

    predictions = iter(["ABC123", "ABC123", "QQQ111"])
    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        side_effect=lambda crop: next(predictions),
    ):
        report = run_plate_ocr_benchmark(gt_path)

    miss_dist = report["miss_distance_distribution"]
    assert miss_dist["num_misses"] == 2
    assert miss_dist["histogram"]["1"]["count"] == 1
    assert miss_dist["histogram"]["4+"]["count"] == 1


def test_load_source_videos_reads_manifest(tmp_path):
    manifest = {
        "cam1/frame1.jpg": {
            "timestamp": 1.0,
            "confidence": 0.9,
            "source_video": "/videos/cam1.mp4",
            "frame_index": 1,
        },
        "cam2/frame1.jpg": {
            "timestamp": 1.0,
            "confidence": 0.9,
            "source_video": "/videos/cam2.mp4",
            "frame_index": 1,
        },
    }
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    loaded = load_source_videos(tmp_path)

    assert loaded == {
        "cam1/frame1.jpg": "cam1.mp4",
        "cam2/frame1.jpg": "cam2.mp4",
    }


def test_load_source_videos_missing_manifest_returns_empty(tmp_path):
    assert load_source_videos(tmp_path) == {}


def test_video_group_for_prefers_manifest_then_falls_back_to_parent_dir():
    source_videos = {"camA/frame1.jpg": "camA.mp4"}

    assert video_group_for("camA/frame1.jpg", source_videos) == "camA.mp4"
    assert video_group_for("camB/frame1.jpg", source_videos) == "camB"
    assert video_group_for("top_level.jpg", source_videos) == "(unknown video)"


def test_run_plate_ocr_benchmark_groups_by_source_video_from_manifest(tmp_path):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    _write_dummy_image(tmp_path / "cam1" / "frame1.jpg")
    _write_dummy_image(tmp_path / "cam1" / "frame2.jpg")
    _write_dummy_image(tmp_path / "cam2" / "frame1.jpg")

    manifest = {
        "cam1/frame1.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam1/frame2.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam2/frame1.jpg": {"source_video": "/videos/cam2.mp4"},
    }
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "cam1/frame1.jpg": "ABC123",
                "cam1/frame2.jpg": "ABC123",
                "cam2/frame1.jpg": "XYZ999",
            }
        )
    )

    # cam1's two readings correct, cam2's one reading wrong
    predictions = iter(["ABC123", "ABC123", "WRONGG"])
    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        side_effect=lambda crop: next(predictions),
    ):
        report = run_plate_ocr_benchmark(gt_path)

    by_video = {row["video"]: row for row in report["by_video"]}
    assert by_video.keys() == {"cam1.mp4", "cam2.mp4"}
    assert by_video["cam1.mp4"]["images_evaluated"] == 2
    assert by_video["cam1.mp4"]["unique_plates_seen"] == 1
    assert by_video["cam1.mp4"]["match_rate"] == pytest.approx(1.0)
    assert by_video["cam2.mp4"]["images_evaluated"] == 1
    assert by_video["cam2.mp4"]["match_rate"] == pytest.approx(0.0)

    assert {r["image"]: r["video"] for r in report["results"]} == {
        "cam1/frame1.jpg": "cam1.mp4",
        "cam1/frame2.jpg": "cam1.mp4",
        "cam2/frame1.jpg": "cam2.mp4",
    }


def test_run_plate_ocr_benchmark_groups_by_parent_dir_without_manifest(tmp_path):
    (tmp_path / "camA").mkdir()
    _write_dummy_image(tmp_path / "camA" / "frame1.jpg")
    _write_dummy_image(tmp_path / "toplevel.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps({"camA/frame1.jpg": "ABC123", "toplevel.jpg": "XYZ999"})
    )

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    by_video = {row["video"]: row for row in report["by_video"]}
    assert by_video.keys() == {"camA", "(unknown video)"}
    assert by_video["camA"]["match_rate"] == pytest.approx(1.0)
    assert by_video["(unknown video)"]["match_rate"] == pytest.approx(0.0)


def test_run_plate_ocr_benchmark_attributes_thrown_out_entries_to_their_video(
    tmp_path,
):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    _write_dummy_image(tmp_path / "cam1" / "a.jpg")
    _write_dummy_image(tmp_path / "cam2" / "x.jpg")
    # cam2/y.jpg intentionally not written - simulates a missing image file

    manifest = {
        "cam1/a.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam1/unreadable.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam2/x.jpg": {"source_video": "/videos/cam2.mp4"},
        "cam2/y.jpg": {"source_video": "/videos/cam2.mp4"},
        "cam2/pending.jpg": {"source_video": "/videos/cam2.mp4"},
    }
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "cam1/a.jpg": "ABC123",
                "cam1/unreadable.jpg": "",  # thrown out: unusable, belongs to cam1
                "cam2/x.jpg": "XYZ999",
                "cam2/y.jpg": "DEF456",  # thrown out: missing file, belongs to cam2
                "cam2/pending.jpg": None,  # thrown out: unusable, belongs to cam2
            }
        )
    )

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    assert report["statistics"]["images_thrown_out"] == 3

    by_video = {row["video"]: row for row in report["by_video"]}
    assert by_video["cam1.mp4"]["images_thrown_out"] == 1
    assert by_video["cam1.mp4"]["thrown_out_unusable_ground_truth"] == 1
    assert by_video["cam1.mp4"]["skipped_missing_images"] == 0

    assert by_video["cam2.mp4"]["images_thrown_out"] == 2
    assert by_video["cam2.mp4"]["thrown_out_unusable_ground_truth"] == 1
    assert by_video["cam2.mp4"]["skipped_missing_images"] == 1


def test_run_plate_ocr_benchmark_includes_miss_distance_distribution_per_video(
    tmp_path,
):
    (tmp_path / "cam1").mkdir()
    (tmp_path / "cam2").mkdir()
    _write_dummy_image(tmp_path / "cam1" / "a.jpg")
    _write_dummy_image(tmp_path / "cam1" / "b.jpg")
    _write_dummy_image(tmp_path / "cam2" / "c.jpg")

    manifest = {
        "cam1/a.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam1/b.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam2/c.jpg": {"source_video": "/videos/cam2.mp4"},
    }
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(
        json.dumps(
            {
                "cam1/a.jpg": "ABC123",  # exact match
                "cam1/b.jpg": "ABD123",  # 1 char off, cam1
                "cam2/c.jpg": "XYZ999",  # totally different, cam2
            }
        )
    )

    # sorted rel paths: cam1/a.jpg, cam1/b.jpg, cam2/c.jpg
    predictions = iter(["ABC123", "ABC123", "QQQ111"])
    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        side_effect=lambda crop: next(predictions),
    ):
        report = run_plate_ocr_benchmark(gt_path)

    by_video = {row["video"]: row for row in report["by_video"]}

    cam1_dist = by_video["cam1.mp4"]["miss_distance_distribution"]
    assert cam1_dist["num_misses"] == 1
    assert cam1_dist["histogram"]["1"]["count"] == 1
    assert cam1_dist["histogram"]["4+"]["count"] == 0

    cam2_dist = by_video["cam2.mp4"]["miss_distance_distribution"]
    assert cam2_dist["num_misses"] == 1
    assert cam2_dist["histogram"]["4+"]["count"] == 1


def test_render_html_report_contains_real_tables_and_key_figures(tmp_path):
    (tmp_path / "cam1").mkdir()
    _write_dummy_image(tmp_path / "cam1" / "a.jpg")
    _write_dummy_image(tmp_path / "cam1" / "b.jpg")

    manifest = {
        "cam1/a.jpg": {"source_video": "/videos/cam1.mp4"},
        "cam1/b.jpg": {"source_video": "/videos/cam1.mp4"},
    }
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"cam1/a.jpg": "ABC123", "cam1/b.jpg": "ABD123"}))

    predictions = iter(["ABC123", "ABC123"])
    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        side_effect=lambda crop: next(predictions),
    ):
        report = run_plate_ocr_benchmark(gt_path)

    doc = render_html_report(report)

    # Real <table> markup, not preformatted/monospace text, is what
    # survives copy-paste into Outlook as an actual table. 4 tables: overview,
    # miss-distance overview, miss-distance histogram, and by-video.
    assert doc.count("<table") == 4
    assert doc.count("</table>") == 4
    assert "<pre" not in doc

    assert "Plate OCR Benchmark" in doc
    assert "50.00% (1/2)" in doc  # exact match rate
    assert "cam1.mp4" in doc
    assert "1 char off" in doc


def test_render_html_report_escapes_video_names(tmp_path):
    (tmp_path / "cam<1>").mkdir()
    _write_dummy_image(tmp_path / "cam<1>" / "a.jpg")

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"cam<1>/a.jpg": "ABC123"}))

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_ocr_benchmark(gt_path)

    doc = render_html_report(report)

    assert "cam&lt;1&gt;" in doc
    assert "<1>" not in doc


def test_run_plate_ocr_benchmark_writes_html_report(tmp_path):
    _write_dummy_image(tmp_path / "a.jpg")
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps({"a.jpg": "ABC123"}))
    html_path = tmp_path / "summary.html"

    with patch(
        "src.utility.plate_ocr_benchmark.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        run_plate_ocr_benchmark(gt_path, html_report=html_path)

    assert html_path.exists()
    content = html_path.read_text()
    assert "<table" in content
    assert "100.00% (1/1)" in content
