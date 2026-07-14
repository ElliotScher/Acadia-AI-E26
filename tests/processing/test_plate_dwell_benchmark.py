import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.processing.plate_dwellprofiler import PlateDetection, compute_plate_dwell_times
from src.utility.plate_dwell_benchmark import (
    CrossingComparison,
    _mean_and_median_dwell,
    _status_summary,
    build_ground_truth_detections,
    compare_crossings,
    compute_dwell_accuracy,
    render_dwell_time_histogram,
    render_html_report,
    render_saved_report,
    run_plate_dwell_benchmark,
    split_fragment_texts,
)


def test_compare_crossings_exact_when_ocr_reproduces_same_images():
    gt_detections = [
        PlateDetection("AAA111", 1.0, Path("a1.jpg")),
        PlateDetection("AAA111", 50.0, Path("a2.jpg")),
    ]
    ocr_detections = [
        PlateDetection("AAA111", 1.0, Path("a1.jpg")),
        PlateDetection("AAA111", 50.0, Path("a2.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    assert len(comparisons) == 1
    c = comparisons[0]
    assert c.status == "exact"
    assert c.ocr_dwell_time == pytest.approx(49.0)
    assert c.dwell_time_delta == pytest.approx(0.0)


def test_compare_crossings_corrupted_when_a_sighting_is_dropped():
    gt_detections = [
        PlateDetection("BBB222", 2.0, Path("b1.jpg")),
        PlateDetection("BBB222", 60.0, Path("b2.jpg")),
    ]
    # b2.jpg failed OCR entirely - never produced a detection.
    ocr_detections = [
        PlateDetection("BBB222", 2.0, Path("b1.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    assert comparisons[0].status == "corrupted"
    assert comparisons[0].ocr_dwell_time is None
    assert comparisons[0].ocr_num_sightings == 1


def test_compare_crossings_corrupted_when_two_plates_merge_under_one_misread():
    gt_detections = [
        PlateDetection("FFF666", 5.0, Path("f1.jpg")),
        PlateDetection("FFF666", 90.0, Path("f2.jpg")),
        PlateDetection("GGG777", 6.0, Path("g1.jpg")),
        PlateDetection("GGG777", 100.0, Path("g2.jpg")),
    ]
    # All four images misread to the same text - a false merge.
    ocr_detections = [
        PlateDetection("FGF666", 5.0, Path("f1.jpg")),
        PlateDetection("FGF666", 90.0, Path("f2.jpg")),
        PlateDetection("FGF666", 6.0, Path("g1.jpg")),
        PlateDetection("FGF666", 100.0, Path("g2.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    comparisons = {
        c.true_plate: c
        for c in compare_crossings(gt_matches, gt_detections, ocr_detections)
    }

    assert comparisons["FFF666"].status == "corrupted"
    assert comparisons["GGG777"].status == "corrupted"
    assert comparisons["FFF666"].ocr_dwell_time == pytest.approx(95.0)
    assert comparisons["GGG777"].ocr_dwell_time == pytest.approx(95.0)
    assert comparisons["FFF666"].ocr_num_sightings == 4


def test_compare_crossings_split_when_readings_are_inconsistent():
    gt_detections = [
        PlateDetection("CCC333", 3.0, Path("c1.jpg")),
        PlateDetection("CCC333", 70.0, Path("c2.jpg")),
    ]
    ocr_detections = [
        PlateDetection("CXC333", 3.0, Path("c1.jpg")),
        PlateDetection("CCX333", 70.0, Path("c2.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    assert comparisons[0].status == "split"
    assert comparisons[0].ocr_dwell_time is None
    assert set(comparisons[0].ocr_texts_seen) == {"CXC333", "CCX333"}


def test_split_fragment_texts_collects_only_split_crossings():
    gt_detections = [
        PlateDetection("CCC333", 1.0, Path("c1.jpg")),
        PlateDetection("CCC333", 20.0, Path("c2.jpg")),
        PlateDetection("CCC333", 90.0, Path("c3.jpg")),
        PlateDetection("DDD444", 5.0, Path("d1.jpg")),
        PlateDetection("DDD444", 65.0, Path("d2.jpg")),
    ]
    # CCC333 splits into a 2-sighting fragment (CXC333) and a 1-sighting
    # leftover (CCX333); DDD444 reads correctly and shouldn't be flagged.
    ocr_detections = [
        PlateDetection("CXC333", 1.0, Path("c1.jpg")),
        PlateDetection("CXC333", 20.0, Path("c2.jpg")),
        PlateDetection("CCX333", 90.0, Path("c3.jpg")),
        PlateDetection("DDD444", 5.0, Path("d1.jpg")),
        PlateDetection("DDD444", 65.0, Path("d2.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)
    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    fragments = split_fragment_texts(comparisons)

    assert fragments == {"CXC333", "CCX333"}


def test_status_summary_min_dwell_time_excludes_short_gt_dwell_from_corrupted_delta():
    # NOISE: a 0.03s "crossing" that gets merged with a different plate's
    # sighting under OCR - its delta is mostly noise, not real error.
    noise_gt = [
        PlateDetection("NOISE1", 1.000, Path("n1.jpg")),
        PlateDetection("NOISE1", 1.030, Path("n2.jpg")),
    ]
    noise_ocr = [
        PlateDetection("MERGED", 1.000, Path("n1.jpg")),
        PlateDetection("MERGED", 1.030, Path("n2.jpg")),
        PlateDetection("MERGED", 500.0, Path("other.jpg")),
    ]
    # REAL: a genuine crossing corrupted by an unrelated image merging in
    # under the same misread, so it still has a computable (but wrong) delta.
    real_gt = [
        PlateDetection("REAL1", 10.0, Path("r1.jpg")),
        PlateDetection("REAL1", 70.0, Path("r2.jpg")),
    ]
    real_ocr = [
        PlateDetection("REAL1", 10.0, Path("r1.jpg")),
        PlateDetection("REAL1", 70.0, Path("r2.jpg")),
        PlateDetection("REAL1", 500.0, Path("extra.jpg")),
    ]

    gt_detections = noise_gt + real_gt
    ocr_detections = noise_ocr + real_ocr
    gt_matches, _ = compute_plate_dwell_times(gt_detections)
    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    assert {c.true_plate: c.status for c in comparisons} == {
        "NOISE1": "corrupted",
        "REAL1": "corrupted",
    }

    unfiltered = _status_summary(comparisons)
    assert unfiltered["excluded_from_corrupted_delta"] == 0

    filtered = _status_summary(comparisons, min_dwell_time=5.0)
    noise_delta = next(
        c for c in comparisons if c.true_plate == "NOISE1"
    ).dwell_time_delta
    assert filtered["excluded_from_corrupted_delta"] == 1
    assert filtered["mean_corrupted_delta"] != pytest.approx(
        unfiltered["mean_corrupted_delta"]
    )
    assert noise_delta is not None


def test_compare_crossings_dropped_when_all_sightings_fail_ocr():
    gt_detections = [
        PlateDetection("DDD444", 4.0, Path("d1.jpg")),
        PlateDetection("DDD444", 80.0, Path("d2.jpg")),
    ]
    ocr_detections = []  # both images failed OCR entirely
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    comparisons = compare_crossings(gt_matches, gt_detections, ocr_detections)

    assert comparisons[0].status == "dropped"
    assert comparisons[0].ocr_dwell_time is None
    assert comparisons[0].ocr_texts_seen == []


def test_build_ground_truth_detections_prefers_manifest_timestamp(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"")
    ground_truth = {"a.jpg": "ABC123"}
    manifest = {tmp_path / "a.jpg": 42.0}

    detections = build_ground_truth_detections(ground_truth, tmp_path, manifest)

    assert len(detections) == 1
    assert detections[0].plate_text == "ABC123"
    assert detections[0].timestamp == 42.0
    assert detections[0].img_path == tmp_path / "a.jpg"


def _write_dummy_image(path):
    cv2.imwrite(str(path), np.zeros((50, 100, 3), dtype="uint8"))


def test_run_plate_dwell_benchmark_end_to_end(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 1.0),
        ("a2.jpg", "AAA111", 50.0),
        ("b1.jpg", "BBB222", 2.0),
        ("b2.jpg", "BBB222", 60.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)

    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))

    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    # a1/a2 read correctly; b2 fails OCR entirely, corrupting BBB222's crossing.
    predictions = {
        "a1.jpg": "AAA111",
        "a2.jpg": "AAA111",
        "b1.jpg": "BBB222",
        "b2.jpg": None,
    }
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    report_path = tmp_path / "report.json"
    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", report=report_path
        )

    crossings_by_plate = {c["true_plate"]: c for c in report["crossings"]}
    assert crossings_by_plate["AAA111"]["status"] == "exact"
    assert crossings_by_plate["AAA111"]["dwell_time_delta"] == pytest.approx(0.0)
    assert crossings_by_plate["BBB222"]["status"] == "corrupted"
    assert crossings_by_plate["BBB222"]["ocr_dwell_time"] is None

    assert report["ground_truth_run"]["statistics"]["matched_crossings"] == 2
    assert report["ocr_run"]["statistics"]["matched_crossings"] == 1

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert len(saved["crossings"]) == 2


def test_run_plate_dwell_benchmark_excludes_split_fragments_from_ocr_average(
    tmp_path,
):
    entries = [
        ("c1.jpg", "CCC333", 1.0),
        ("c2.jpg", "CCC333", 20.0),
        ("c3.jpg", "CCC333", 90.0),
        ("d1.jpg", "DDD444", 5.0),
        ("d2.jpg", "DDD444", 65.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)

    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    # CCC333 (3 sightings) splits: c1/c2 share a misread (a real 2-sighting
    # OCR match with a bogus 19s dwell), c3 gets a different misread.
    # DDD444 reads correctly throughout.
    predictions = {
        "c1.jpg": "CXC333",
        "c2.jpg": "CXC333",
        "c3.jpg": "CCX333",
        "d1.jpg": "DDD444",
        "d2.jpg": "DDD444",
    }
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(tmp_path / "ground_truth.json")

    crossings_by_plate = {c["true_plate"]: c for c in report["crossings"]}
    assert crossings_by_plate["CCC333"]["status"] == "split"
    assert crossings_by_plate["DDD444"]["status"] == "exact"

    ocr_stats = report["ocr_run"]["statistics"]
    # The 2-sighting split fragment (CXC333, dwell=19s) is excluded, so the
    # average is purely DDD444's genuine 60s crossing.
    assert ocr_stats["matches_excluded_from_average"] == 1
    assert ocr_stats["excluded_by_split_fragment"] == 1
    assert ocr_stats["excluded_by_min_dwell_time"] == 0
    assert ocr_stats["average_dwell_time"] == pytest.approx(60.0)

    matches_by_text = {
        m["plate_text"]: m for m in report["ocr_run"]["dwell_time_matches"]
    }
    assert matches_by_text["CXC333"]["counted_in_average"] is False
    assert matches_by_text["DDD444"]["counted_in_average"] is True


def test_run_plate_dwell_benchmark_min_dwell_time_excludes_short_crossings(
    tmp_path,
):
    entries = [
        ("n1.jpg", "NNN111", 1.000),  # 0.03s dwell - frame noise, not a real crossing
        ("n2.jpg", "NNN111", 1.030),
        ("g1.jpg", "GGG222", 10.0),  # 600s dwell - genuine crossing
        ("g2.jpg", "GGG222", 610.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)

    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", min_dwell_time=5.0
        )

    gt_stats = report["ground_truth_run"]["statistics"]
    ocr_stats = report["ocr_run"]["statistics"]

    # Both averages exclude the 0.03s noise crossing, so both equal GGG222's
    # genuine 600s dwell - not the misleadingly-low blended average.
    assert gt_stats["excluded_by_min_dwell_time"] == 1
    assert gt_stats["average_dwell_time"] == pytest.approx(600.0)
    assert ocr_stats["excluded_by_min_dwell_time"] == 1
    assert ocr_stats["excluded_by_split_fragment"] == 0
    assert ocr_stats["average_dwell_time"] == pytest.approx(600.0)

    gt_matches_by_plate = {
        m["plate_text"]: m for m in report["ground_truth_run"]["dwell_time_matches"]
    }
    assert gt_matches_by_plate["NNN111"]["counted_in_average"] is False
    assert gt_matches_by_plate["GGG222"]["counted_in_average"] is True

    # Excluded crossings are still listed in full, not dropped from the table.
    crossings_by_plate = {c["true_plate"]: c for c in report["crossings"]}
    assert crossings_by_plate["NNN111"]["status"] == "exact"
    assert crossings_by_plate["NNN111"]["ground_truth_dwell_time"] == pytest.approx(
        0.03
    )


def test_run_plate_dwell_benchmark_defaults_input_dir_to_ground_truth_parent(
    tmp_path,
):
    nested = tmp_path / "crops"
    nested.mkdir()
    _write_dummy_image(nested / "a1.jpg")
    _write_dummy_image(nested / "a2.jpg")

    gt_path = nested / "ground_truth.json"
    gt_path.write_text(json.dumps({"a1.jpg": "ABC123", "a2.jpg": "ABC123"}))
    (nested / "plate_manifest.json").write_text(
        json.dumps(
            {
                "a1.jpg": {"timestamp": 1.0},
                "a2.jpg": {"timestamp": 30.0},
            }
        )
    )

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        return_value="ABC123",
    ):
        report = run_plate_dwell_benchmark(gt_path)

    assert report["metadata"]["input_dir"] == str(nested)
    assert report["crossings"][0]["status"] == "exact"


def test_render_html_report_contains_real_tables_and_key_figures(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 1.0),
        ("a2.jpg", "AAA111", 50.0),
        ("b1.jpg", "BBB222", 2.0),
        ("b2.jpg", "BBB222", 60.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)

    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {
        "a1.jpg": "AAA111",
        "a2.jpg": "AAA111",
        "b1.jpg": "BBB222",
        "b2.jpg": None,
    }
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(tmp_path / "ground_truth.json")

    doc = render_html_report(report)

    # Real <table> markup, not preformatted/monospace text, is what
    # survives copy-paste into Outlook as an actual table. 3 tables:
    # overview, dwell-time accuracy, and crossings.
    assert doc.count("<table") == 3
    assert doc.count("</table>") == 3
    assert "<pre" not in doc

    assert "Plate Dwell Time Benchmark" in doc
    assert "AAA111" in doc
    assert "exact" in doc
    assert "BBB222" in doc
    assert "corrupted" in doc


def test_render_html_report_escapes_plate_text():
    gt_detections = [
        PlateDetection("<AAA111>", 1.0, Path("a1.jpg")),
        PlateDetection("<AAA111>", 50.0, Path("a2.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)
    for m in gt_matches:
        m["counted_in_average"] = True
    comparisons = compare_crossings(gt_matches, gt_detections, gt_detections)

    summary_report = {
        "ground_truth_run": {
            "statistics": {
                "average_dwell_time": 49.0,
                "excluded_by_min_dwell_time": 0,
            },
            "dwell_time_matches": gt_matches,
        },
        "ocr_run": {
            "statistics": {
                "average_dwell_time": 49.0,
                "excluded_by_min_dwell_time": 0,
                "excluded_by_split_fragment": 0,
            },
            "dwell_time_matches": gt_matches,
        },
        "status_summary": _status_summary(comparisons),
        "crossings": [
            {
                "true_plate": c.true_plate,
                "status": c.status,
                "ground_truth_dwell_time": c.ground_truth_dwell_time,
                "ground_truth_num_sightings": c.ground_truth_num_sightings,
                "ocr_dwell_time": c.ocr_dwell_time,
                "ocr_num_sightings": c.ocr_num_sightings,
                "dwell_time_delta": c.dwell_time_delta,
                "ocr_texts_seen": c.ocr_texts_seen,
            }
            for c in comparisons
        ],
    }

    doc = render_html_report(summary_report)

    assert "&lt;AAA111&gt;" in doc
    assert "<AAA111>" not in doc


def test_run_plate_dwell_benchmark_writes_html_report(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 1.0),
        ("a2.jpg", "AAA111", 50.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    html_path = tmp_path / "summary.html"
    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        return_value="AAA111",
    ):
        run_plate_dwell_benchmark(tmp_path / "ground_truth.json", html_report=html_path)

    assert html_path.exists()
    content = html_path.read_text()
    assert "<table" in content
    assert "AAA111" in content


def test_mean_and_median_dwell_uses_only_counted_matches():
    matches = [
        {"dwell_time": 10.0, "counted_in_average": True},
        {"dwell_time": 20.0, "counted_in_average": True},
        {"dwell_time": 30.0, "counted_in_average": True},
        {"dwell_time": 12000.0, "counted_in_average": False},
    ]

    mean, median = _mean_and_median_dwell(matches)

    assert mean == pytest.approx(20.0)
    assert median == pytest.approx(20.0)


def test_mean_and_median_dwell_treats_missing_flag_as_counted():
    # A match with no counted_in_average key at all - e.g. a report saved
    # before that flag existed - should still be included, not silently
    # dropped from the average.
    matches = [
        {"dwell_time": 10.0},
        {"dwell_time": 30.0},
    ]

    mean, median = _mean_and_median_dwell(matches)

    assert mean == pytest.approx(20.0)
    assert median == pytest.approx(20.0)


def test_mean_and_median_dwell_empty_returns_zero():
    assert _mean_and_median_dwell([]) == (0.0, 0.0)


def test_run_plate_dwell_benchmark_reports_median_alongside_mean(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 0.0),
        ("a2.jpg", "AAA111", 10.0),
        ("b1.jpg", "BBB222", 0.0),
        ("b2.jpg", "BBB222", 20.0),
        ("c1.jpg", "CCC333", 0.0),
        ("c2.jpg", "CCC333", 30.0),
        ("d1.jpg", "DDD444", 0.0),
        ("d2.jpg", "DDD444", 12000.0),  # far outlier, pulls the mean way up
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(tmp_path / "ground_truth.json")

    gt_matches = report["ground_truth_run"]["dwell_time_matches"]
    mean, median = _mean_and_median_dwell(gt_matches)
    assert mean == pytest.approx((10.0 + 20.0 + 30.0 + 12000.0) / 4)
    assert median == pytest.approx((20.0 + 30.0) / 2)
    assert median < mean  # the whole point: median isn't dragged by the outlier


def test_render_saved_report_reproduces_console_and_html_without_ocr(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 0.0),
        ("a2.jpg", "AAA111", 10.0),
        ("b1.jpg", "BBB222", 0.0),
        ("b2.jpg", "BBB222", 20.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    report_path = tmp_path / "manifest.json"
    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        original = run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", report=report_path
        )

    html_path = tmp_path / "summary.html"
    # No OCR patch active here at all - proves no OCR/detection is re-run.
    reloaded = render_saved_report(report_path, html_report=html_path)

    assert reloaded["crossings"] == original["crossings"]
    assert html_path.exists()
    assert "<table" in html_path.read_text()


def test_render_saved_report_tolerates_report_missing_newer_fields(tmp_path):
    # Simulates a report saved by an older version of this script, before
    # excluded_by_*/excluded_from_corrupted_delta existed.
    old_report = {
        "ground_truth_run": {
            "statistics": {"average_dwell_time": 20.0},
            "dwell_time_matches": [
                {"plate_text": "AAA111", "dwell_time": 10.0, "num_sightings": 2},
                {"plate_text": "BBB222", "dwell_time": 30.0, "num_sightings": 2},
            ],
        },
        "ocr_run": {
            "statistics": {"average_dwell_time": 20.0},
            "dwell_time_matches": [
                {"plate_text": "AAA111", "dwell_time": 10.0, "num_sightings": 2},
                {"plate_text": "BBB222", "dwell_time": 30.0, "num_sightings": 2},
            ],
        },
        "status_summary": {
            "total_crossings": 2,
            "status_counts": {"dropped": 0, "split": 0, "corrupted": 0, "exact": 2},
            "mean_corrupted_delta": None,
        },
        "crossings": [
            {
                "true_plate": "AAA111",
                "status": "exact",
                "ground_truth_dwell_time": 10.0,
                "ground_truth_num_sightings": 2,
                "ocr_dwell_time": 10.0,
                "ocr_num_sightings": 2,
                "dwell_time_delta": 0.0,
                "ocr_texts_seen": ["AAA111"],
            },
        ],
    }
    report_path = tmp_path / "old_manifest.json"
    report_path.write_text(json.dumps(old_report))

    result = render_saved_report(report_path)

    assert result == old_report


def _judged(status, gt_dwell, ocr_dwell):
    return CrossingComparison(
        true_plate="X",
        ground_truth_dwell_time=gt_dwell,
        ground_truth_num_sightings=2,
        status=status,
        ocr_dwell_time=ocr_dwell,
        ocr_num_sightings=2,
        ocr_texts_seen=["X"],
    )


def test_compute_dwell_accuracy_ignores_split_and_dropped():
    comparisons = [
        _judged("exact", 100.0, 100.0),
        CrossingComparison("Y", 50.0, 2, status="split"),
        CrossingComparison("Z", 50.0, 2, status="dropped"),
    ]

    result = compute_dwell_accuracy(comparisons)

    assert result["total_judged"] == 1
    assert result["reasonable_count"] == 1


def test_compute_dwell_accuracy_buckets_by_relative_error():
    comparisons = [
        _judged("exact", 100.0, 100.0),  # 0% error
        _judged("corrupted", 100.0, 115.0),  # 15% error
        _judged("corrupted", 100.0, 140.0),  # 40% error
        _judged("corrupted", 100.0, 300.0),  # 200% error
    ]

    result = compute_dwell_accuracy(comparisons, tolerance=0.2)

    assert result["total_judged"] == 4
    assert result["reasonable_count"] == 2  # 0% and 15% are within 20%
    assert result["reasonable_rate"] == pytest.approx(0.5)
    assert result["histogram"]["<=5%"]["count"] == 1
    assert result["histogram"]["<=20%"]["count"] == 1
    assert result["histogram"]["<=50%"]["count"] == 1
    assert result["histogram"][">50%"]["count"] == 1


def test_compute_dwell_accuracy_excludes_short_gt_dwell():
    comparisons = [
        _judged("exact", 0.03, 500.0),  # near-zero GT dwell, huge relative error
        _judged("exact", 100.0, 100.0),
    ]

    unfiltered = compute_dwell_accuracy(comparisons, min_dwell_time=0.0)
    assert unfiltered["total_judged"] == 2
    assert unfiltered["excluded_short_dwell"] == 0

    filtered = compute_dwell_accuracy(comparisons, min_dwell_time=5.0)
    assert filtered["total_judged"] == 1
    assert filtered["excluded_short_dwell"] == 1
    assert filtered["reasonable_rate"] == pytest.approx(1.0)


def test_compute_dwell_accuracy_no_judgeable_crossings_returns_zero():
    comparisons = [CrossingComparison("Y", 50.0, 2, status="split")]

    result = compute_dwell_accuracy(comparisons)

    assert result["total_judged"] == 0
    assert result["reasonable_rate"] == 0.0
    assert all(b["count"] == 0 for b in result["histogram"].values())


def test_render_saved_report_min_dwell_time_override_excludes_noise_crossing(
    tmp_path,
):
    entries = [
        ("n1.jpg", "NOISE1", 1.000),  # 0.03s - noise
        ("n2.jpg", "NOISE1", 1.030),
        ("g1.jpg", "GOOD1", 10.0),  # 60s - genuine
        ("g2.jpg", "GOOD1", 70.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    report_path = tmp_path / "manifest.json"
    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", report=report_path, min_dwell_time=0.0
        )

    # Re-analyze the SAME saved report with a different threshold - no OCR
    # mock active here at all, proving nothing gets re-run.
    reloaded = render_saved_report(report_path, min_dwell_time=5.0)

    # Can't inspect dwell_accuracy directly from the return value (it's only
    # printed), so recompute the same way render_saved_report does internally
    # to confirm the noise crossing is excluded under the new threshold.
    from src.utility.plate_dwell_benchmark import _comparisons_from_json

    comparisons = _comparisons_from_json(reloaded["crossings"])
    unfiltered = compute_dwell_accuracy(comparisons, min_dwell_time=0.0)
    filtered = compute_dwell_accuracy(comparisons, min_dwell_time=5.0)

    assert unfiltered["total_judged"] == 2
    assert filtered["total_judged"] == 1
    assert filtered["excluded_short_dwell"] == 1


def _match(dwell_time, counted=True):
    return {"plate_text": "X", "dwell_time": dwell_time, "counted_in_average": counted}


def test_render_dwell_time_histogram_returns_valid_png():
    gt_matches = [_match(10.0), _match(100.0), _match(1000.0)]
    ocr_matches = [_match(12.0), _match(90.0)]

    png_bytes = render_dwell_time_histogram(gt_matches, ocr_matches)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")  # PNG magic number
    assert len(png_bytes) > 0


def test_render_dwell_time_histogram_respects_counted_in_average():
    gt_matches = [_match(10.0, counted=True), _match(99999.0, counted=False)]
    ocr_matches = [_match(10.0, counted=True)]

    # Should not raise/blow up the axis range with the excluded outlier, and
    # should still produce a valid image using only the counted values.
    png_bytes = render_dwell_time_histogram(gt_matches, ocr_matches)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_dwell_time_histogram_empty_returns_empty_bytes():
    assert render_dwell_time_histogram([], []) == b""
    # All excluded -> nothing countable either.
    assert render_dwell_time_histogram([_match(10.0, counted=False)], []) == b""


def test_render_dwell_time_histogram_single_value_does_not_crash():
    # A degenerate case where log-spacing bins from min==max would divide by
    # zero if not guarded.
    png_bytes = render_dwell_time_histogram([_match(50.0)], [_match(50.0)])

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_run_plate_dwell_benchmark_writes_histogram_file(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 0.0),
        ("a2.jpg", "AAA111", 10.0),
        ("b1.jpg", "BBB222", 0.0),
        ("b2.jpg", "BBB222", 20.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    histogram_path = tmp_path / "histogram.png"
    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", histogram=histogram_path
        )

    assert histogram_path.exists()
    assert histogram_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_render_html_report_embeds_histogram_image(tmp_path):
    entries = [
        ("a1.jpg", "AAA111", 0.0),
        ("a2.jpg", "AAA111", 10.0),
        ("b1.jpg", "BBB222", 0.0),
        ("b2.jpg", "BBB222", 20.0),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {name: text for name, text, _ in entries}
    call_order = iter(gt.keys())

    def fake_ocr(crop):
        return predictions[next(call_order)]

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=fake_ocr,
    ):
        report = run_plate_dwell_benchmark(tmp_path / "ground_truth.json")

    doc = render_html_report(report)

    assert '<img src="data:image/png;base64,' in doc
    assert "Dwell Time Distribution" in doc


def test_compare_crossings_max_edit_distance_reunifies_split_crossing():
    gt_detections = [
        PlateDetection("381KE4", 0.0, Path("c1.jpg")),
        PlateDetection("381KE4", 4000.0, Path("c2.jpg")),
        PlateDetection("381KE4", 11949.89, Path("c3.jpg")),
    ]
    # OCR reads the same long-duration plate three different (but similar) ways.
    ocr_detections = [
        PlateDetection("381KE4", 0.0, Path("c1.jpg")),
        PlateDetection("381KE9", 4000.0, Path("c2.jpg")),
        PlateDetection("38IKE4", 11949.89, Path("c3.jpg")),
    ]
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    exact = compare_crossings(gt_matches, gt_detections, ocr_detections)
    assert exact[0].status == "split"

    fuzzy = compare_crossings(
        gt_matches, gt_detections, ocr_detections, max_edit_distance=1
    )
    assert fuzzy[0].status == "exact"
    assert fuzzy[0].dwell_time_delta == pytest.approx(0.0)


def test_compare_crossings_max_time_gap_prevents_false_merge():
    gt_detections = [
        PlateDetection("ABC123", 0.0, Path("a1.jpg")),
        PlateDetection("ABC123", 30.0, Path("a2.jpg")),
        PlateDetection("ABD123", 20000.0, Path("b1.jpg")),
        PlateDetection("ABD123", 20030.0, Path("b2.jpg")),
    ]
    ocr_detections = list(gt_detections)  # OCR reads both correctly
    gt_matches, _ = compute_plate_dwell_times(gt_detections)

    unguarded = {
        c.true_plate: c
        for c in compare_crossings(
            gt_matches, gt_detections, ocr_detections, max_edit_distance=1
        )
    }
    assert unguarded["ABC123"].status == "corrupted"
    assert unguarded["ABD123"].status == "corrupted"

    guarded = {
        c.true_plate: c
        for c in compare_crossings(
            gt_matches,
            gt_detections,
            ocr_detections,
            max_edit_distance=1,
            max_time_gap=3600.0,
        )
    }
    assert guarded["ABC123"].status == "exact"
    assert guarded["ABD123"].status == "exact"


def test_run_plate_dwell_benchmark_max_edit_distance_end_to_end(tmp_path):
    entries = [
        ("c1.jpg", "381KE4", 0.0),
        ("c2.jpg", "381KE4", 4000.0),
        ("c3.jpg", "381KE4", 11949.89),
    ]
    for name, _, _ in entries:
        _write_dummy_image(tmp_path / name)
    gt = {name: text for name, text, _ in entries}
    (tmp_path / "ground_truth.json").write_text(json.dumps(gt))
    manifest = {name: {"timestamp": ts} for name, _, ts in entries}
    (tmp_path / "plate_manifest.json").write_text(json.dumps(manifest))

    predictions = {"c1.jpg": "381KE4", "c2.jpg": "381KE9", "c3.jpg": "38IKE4"}

    def make_fake_ocr():
        call_order = iter(gt.keys())

        def fake_ocr(crop):
            return predictions[next(call_order)]

        return fake_ocr

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=make_fake_ocr(),
    ):
        exact_report = run_plate_dwell_benchmark(tmp_path / "ground_truth.json")
    assert exact_report["crossings"][0]["status"] == "split"

    with patch(
        "src.processing.plate_dwellprofiler.extract_plate_text_via_ocr",
        side_effect=make_fake_ocr(),
    ):
        fuzzy_report = run_plate_dwell_benchmark(
            tmp_path / "ground_truth.json", max_edit_distance=1
        )
    assert fuzzy_report["crossings"][0]["status"] == "exact"
    assert fuzzy_report["crossings"][0]["dwell_time_delta"] == pytest.approx(0.0)
    assert fuzzy_report["metadata"]["max_edit_distance"] == 1
