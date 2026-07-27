import json

import numpy as np
import pytest

import src.utility.speed_distribution_comparator as speed_distribution_comparator
from src.utility.speed_distribution_comparator import (
    ComparisonResult,
    SpeedDistributionComparator,
    load_directional_csv,
    load_report_counts,
    plot_ridgeline,
)
from src.utility.speed_distribution_comparator import (
    _format_final_statistics,
    _format_intuitive_summary,
    _report_match_stats,
)


def test_init_rejects_empty_bins():
    with pytest.raises(ValueError, match="speed_bins"):
        SpeedDistributionComparator([], [[1, 2, 3]])


def test_init_rejects_no_rows():
    with pytest.raises(ValueError, match="At least one histogram row"):
        SpeedDistributionComparator([10, 20, 30], [])


def test_init_rejects_mismatched_row_length():
    with pytest.raises(ValueError, match="Row 0"):
        SpeedDistributionComparator([10, 20, 30], [[1, 2]])


def test_init_rejects_mismatched_label_count():
    with pytest.raises(ValueError, match="labels"):
        SpeedDistributionComparator([10, 20], [[1, 2], [3, 4]], labels=["only-one"])


def test_identical_rows_are_perfectly_correlated_and_zero_divergence():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [5, 10, 8, 2]],
    )
    result = comparator.compare(0, 1)
    assert result.pearson_correlation == pytest.approx(1.0)
    assert result.jensen_shannon_divergence == pytest.approx(0.0, abs=1e-9)
    assert result.overlap_coefficient == pytest.approx(1.0)
    assert result.is_stable()


def test_scaled_row_is_perfectly_correlated_but_js_divergence_is_zero_when_shape_matches():
    # Same shape, different sample size - JS divergence normalizes this out,
    # and Pearson correlation is scale-invariant too.
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [50, 100, 80, 20]],
    )
    result = comparator.compare(0, 1)
    assert result.pearson_correlation == pytest.approx(1.0)
    assert result.jensen_shannon_divergence == pytest.approx(0.0, abs=1e-9)
    assert result.overlap_coefficient == pytest.approx(1.0)


def test_disjoint_distributions_have_maximum_js_divergence_and_zero_overlap():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[10, 0, 0, 0], [0, 0, 0, 10]],
    )
    result = comparator.compare(0, 1)
    assert result.jensen_shannon_divergence == pytest.approx(1.0)
    assert result.overlap_coefficient == pytest.approx(0.0)


def test_overlap_coefficient_partial_overlap():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20],
        counts=[[8, 2], [2, 8]],
    )
    # normalized: [0.8, 0.2] vs [0.2, 0.8] -> shared mass = min(0.8,0.2) + min(0.2,0.8) = 0.4
    assert comparator.overlap_coefficient(0, 1) == pytest.approx(0.4)


def test_overlap_coefficient_rejects_all_zero_row():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[0, 0, 0], [1, 2, 3]],
    )
    with pytest.raises(ValueError, match="zero total count"):
        comparator.overlap_coefficient(0, 1)


def test_negatively_correlated_rows():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[1, 2, 3, 4], [4, 3, 2, 1]],
    )
    result = comparator.compare(0, 1)
    assert result.pearson_correlation == pytest.approx(-1.0)


def test_constant_rows_short_circuit_pearson():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 5, 5], [5, 5, 5]],
    )
    assert comparator.pearson_correlation(0, 1) == pytest.approx(1.0)

    comparator2 = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 5, 5], [1, 2, 3]],
    )
    assert comparator2.pearson_correlation(0, 1) == pytest.approx(0.0)


def test_jensen_shannon_divergence_rejects_all_zero_row():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[0, 0, 0], [1, 2, 3]],
    )
    with pytest.raises(ValueError, match="zero total count"):
        comparator.jensen_shannon_divergence(0, 1)


def test_compare_all_returns_every_unordered_pair():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20],
        counts=[[1, 2], [2, 1], [1, 1]],
        labels=["a", "b", "c"],
    )
    results = comparator.compare_all()
    pairs = {(r.label_a, r.label_b) for r in results}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}


def test_is_stable_true_when_all_pairs_pass():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
    )
    assert comparator.is_stable(js_threshold=0.5)


def test_is_stable_false_when_one_pair_fails():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[10, 0, 0, 0], [0, 0, 0, 10]],
    )
    assert not comparator.is_stable()


def test_tail_keeps_only_last_n_rows_and_labels():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20],
        counts=[[1, 2], [3, 4], [5, 6]],
        labels=["a", "b", "c"],
    )
    tail = comparator.tail(2)
    assert tail.labels == ["b", "c"]
    assert np.array_equal(tail.counts, [[3, 4], [5, 6]])
    assert np.array_equal(tail.speed_bins, comparator.speed_bins)


def test_tail_n_larger_than_row_count_keeps_everything():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20],
        counts=[[1, 2], [3, 4]],
        labels=["a", "b"],
    )
    tail = comparator.tail(10)
    assert tail.labels == ["a", "b"]


def test_tail_rejects_n_less_than_one():
    comparator = SpeedDistributionComparator(speed_bins=[10, 20], counts=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="at least 1"):
        comparator.tail(0)


def test_tail_can_reveal_stability_hidden_by_earlier_drift():
    # First two rows drift away from the rest, but the last three are stable.
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[
            [10, 0, 0, 0],
            [0, 0, 0, 10],
            [5, 10, 5, 0],
            [6, 11, 5, 0],
            [4, 9, 6, 0],
        ],
    )
    assert not comparator.is_stable()
    assert comparator.tail(3).is_stable(js_threshold=0.5)


def test_comparison_result_is_stable_respects_js_threshold():
    result = ComparisonResult(
        label_a="a",
        label_b="b",
        pearson_correlation=0.1,
        jensen_shannon_divergence=0.05,
        overlap_coefficient=0.95,
    )
    assert result.is_stable(js_threshold=0.1)
    assert not result.is_stable(js_threshold=0.01)


def test_comparison_result_is_stable_respects_overlap_threshold():
    result = ComparisonResult(
        label_a="a",
        label_b="b",
        pearson_correlation=0.1,
        jensen_shannon_divergence=0.01,
        overlap_coefficient=0.85,
    )
    assert result.is_stable(overlap_threshold=0.8)
    assert not result.is_stable(overlap_threshold=0.9)


def test_load_directional_csv_splits_rows_by_direction(tmp_path):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text(
        "date,direction,10,20,30,40\n"
        "2026-01-01,left,3,12,48,4\n"
        "2026-01-01,right,1,9,40,3\n"
        "2026-01-02,left,2,15,45,3\n"
        "2026-01-02,right,4,11,42,2\n"
    )
    comparators = load_directional_csv(csv_path)
    assert set(comparators) == {"left", "right"}

    left = comparators["left"]
    assert left.labels == ["2026-01-01", "2026-01-02"]
    assert np.array_equal(left.speed_bins, [10, 20, 30, 40])
    assert np.array_equal(left.counts[0], [3, 12, 48, 4])
    assert np.array_equal(left.counts[1], [2, 15, 45, 3])

    right = comparators["right"]
    assert right.labels == ["2026-01-01", "2026-01-02"]
    assert np.array_equal(right.counts[0], [1, 9, 40, 3])
    assert np.array_equal(right.counts[1], [4, 11, 42, 2])


def test_load_directional_csv_preserves_per_direction_order_when_interleaved_out_of_order(
    tmp_path,
):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text(
        "date,direction,10,20\n"
        "2026-01-01,left,1,1\n"
        "2026-01-02,left,2,2\n"
        "2026-01-01,right,3,3\n"
        "2026-01-02,right,4,4\n"
    )
    comparators = load_directional_csv(csv_path)
    # Each direction's own rows stay in file order regardless of how the
    # two directions are interleaved with each other.
    assert comparators["left"].labels == ["2026-01-01", "2026-01-02"]
    assert comparators["right"].labels == ["2026-01-01", "2026-01-02"]


def test_load_directional_csv_requires_header_and_data_row(tmp_path):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text("date,direction,10,20,30\n")
    with pytest.raises(ValueError, match="at least one data row"):
        load_directional_csv(csv_path)


def test_load_directional_csv_requires_three_header_columns(tmp_path):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text("date,direction\n2026-01-01,left\n")
    with pytest.raises(ValueError, match="at least 3 columns"):
        load_directional_csv(csv_path)


def test_load_directional_csv_requires_three_row_columns(tmp_path):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text("date,direction,10,20\n2026-01-01,left\n")
    with pytest.raises(ValueError, match="at least 3 columns"):
        load_directional_csv(csv_path)


def test_load_directional_csv_raises_on_unparseable_row(tmp_path):
    csv_path = tmp_path / "speeds.csv"
    csv_path.write_text("date,direction,10,20\n2026-01-01,left,not-a-number,5\n")
    with pytest.raises(ValueError, match="Could not parse row"):
        load_directional_csv(csv_path)


def test_distribution_normalizes_counts_to_sum_to_one():
    comparator = SpeedDistributionComparator(speed_bins=[10, 20, 30], counts=[[2, 6, 2]])
    p = comparator.distribution(0)
    assert p == pytest.approx([0.2, 0.6, 0.2])
    assert p.sum() == pytest.approx(1.0)


def test_distributions_stacks_every_row():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20], counts=[[1, 1], [1, 3]]
    )
    stacked = comparator.distributions()
    assert stacked.shape == (2, 2)
    assert stacked[0] == pytest.approx([0.5, 0.5])
    assert stacked[1] == pytest.approx([0.25, 0.75])


def test_variance_of_single_bin_distribution_is_zero():
    comparator = SpeedDistributionComparator(speed_bins=[10, 20, 30], counts=[[0, 5, 0]])
    assert comparator.variance(0) == pytest.approx(0.0, abs=1e-9)


def test_variance_matches_known_two_point_distribution():
    # p = [0.5, 0.5] over [10, 20] -> mean 15, variance = 0.5*25 + 0.5*25 = 25
    comparator = SpeedDistributionComparator(speed_bins=[10, 20], counts=[[1, 1]])
    assert comparator.variance(0) == pytest.approx(25.0)


def test_variance_is_scale_invariant_to_sample_size():
    comparator = SpeedDistributionComparator(speed_bins=[10, 20], counts=[[1, 1], [10, 10]])
    assert comparator.variance(0) == pytest.approx(comparator.variance(1))


def test_variance_rejects_all_zero_row():
    comparator = SpeedDistributionComparator(speed_bins=[10, 20, 30], counts=[[0, 0, 0]])
    with pytest.raises(ValueError, match="zero total count"):
        comparator.variance(0)


def test_variances_returns_one_value_per_row():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [1, 0, 0], [0, 0, 1]],
    )
    variances = comparator.variances()
    assert len(variances) == 3
    assert variances[1] == pytest.approx(0.0, abs=1e-9)
    assert variances[2] == pytest.approx(0.0, abs=1e-9)


def test_plot_ridgeline_writes_a_nonempty_png(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3], [4, 9, 9, 1]],
        labels=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    output_path = tmp_path / "ridgeline.png"

    plot_ridgeline(comparator, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridgeline_subsamples_when_over_max_rows(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[i + 1, i + 2, i + 3] for i in range(50)],
    )
    output_path = tmp_path / "ridgeline.png"

    plot_ridgeline(comparator, output_path, max_rows=10)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridgeline_raises_on_zero_total_row(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20],
        counts=[[0, 0], [1, 2]],
    )
    with pytest.raises(ValueError, match="zero total count"):
        plot_ridgeline(comparator, tmp_path / "ridgeline.png")


def test_plot_ridgeline_with_reference_counts_writes_a_nonempty_png(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3], [4, 9, 9, 1]],
    )
    output_path = tmp_path / "ridgeline.png"

    plot_ridgeline(
        comparator, output_path, reference_counts=[[3, 6, 4, 1]], reference_labels=["Report"]
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridgeline_with_multiple_reference_counts_writes_a_nonempty_png(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3], [4, 9, 9, 1]],
    )
    output_path = tmp_path / "ridgeline.png"

    plot_ridgeline(
        comparator,
        output_path,
        reference_counts=[[3, 6, 4, 1], [1, 2, 8, 5], [7, 3, 2, 0]],
        reference_labels=["Report A", "Report B", "Report C"],
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridgeline_defaults_reference_labels_when_not_given(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3], [4, 9, 9, 1]],
    )
    output_path = tmp_path / "ridgeline.png"

    plot_ridgeline(comparator, output_path, reference_counts=[[3, 6, 4, 1], [1, 2, 8, 5]])

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridgeline_rejects_mismatched_reference_counts_length():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3]],
    )
    with pytest.raises(ValueError, match=r"reference_counts\[0\]"):
        plot_ridgeline(comparator, "unused.png", reference_counts=[[1, 2, 3]])


def test_plot_ridgeline_rejects_mismatched_reference_labels_length():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[5, 10, 8, 2], [6, 11, 7, 3]],
    )
    with pytest.raises(ValueError, match="reference_labels"):
        plot_ridgeline(
            comparator,
            "unused.png",
            reference_counts=[[3, 6, 4, 1], [1, 2, 8, 5]],
            reference_labels=["Only One"],
        )


def _write_report(path, entities):
    path.write_text(json.dumps({"individual_entities": entities}))


def test_load_report_counts_bins_absolute_speed_to_nearest_bin(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "entity_type": 2},
            {"absolute_speed": 24.0, "entity_type": 2},
            {"absolute_speed": 34.0, "entity_type": 2},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30, 40])
    assert np.array_equal(counts, [2, 1, 0])


def test_load_report_counts_filters_by_entity_type(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "entity_type": 2},
            {"absolute_speed": 21.0, "entity_type": 1},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30], entity_type=2)
    assert np.array_equal(counts, [1, 0])


def test_load_report_counts_filters_by_direction(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "direction": "left"},
            {"absolute_speed": 21.0, "direction": "right"},
            {"absolute_speed": 29.0, "direction": "right"},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30], direction="right")
    assert np.array_equal(counts, [1, 1])


def test_load_report_counts_direction_filter_is_case_insensitive(tmp_path):
    # video_entityprofiler.py reports always write "left"/"right" lowercase,
    # but a CSV's own direction column might be typed in any case (e.g.
    # "Right") - the filter should still match.
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "direction": "left"},
            {"absolute_speed": 21.0, "direction": "right"},
            {"absolute_speed": 29.0, "direction": "right"},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30], direction="Right")
    assert np.array_equal(counts, [1, 1])


def test_load_report_counts_combines_entity_type_and_direction_filters(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "entity_type": 2, "direction": "left"},
            {"absolute_speed": 21.0, "entity_type": 1, "direction": "left"},
            {"absolute_speed": 21.0, "entity_type": 2, "direction": "right"},
        ],
    )
    counts = load_report_counts(
        report_path, speed_bins=[20, 30], entity_type=2, direction="left"
    )
    assert np.array_equal(counts, [1, 0])


def test_load_report_counts_ignores_null_absolute_speed(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": None, "entity_type": 2},
            {"absolute_speed": 21.0, "entity_type": 2},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30])
    assert np.array_equal(counts, [1, 0])


def test_load_report_counts_raises_on_missing_individual_entities(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({}))
    with pytest.raises(ValueError, match="individual_entities"):
        load_report_counts(report_path, speed_bins=[20, 30])


def test_load_report_counts_raises_when_no_entities_have_absolute_speed(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, [{"absolute_speed": None, "entity_type": 2}])
    with pytest.raises(ValueError, match="absolute_speed"):
        load_report_counts(report_path, speed_bins=[20, 30])


def test_load_report_counts_falls_back_to_relative_speed_when_uncalibrated(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": None, "relative_speed": 0.0, "entity_type": 2},
            {"absolute_speed": None, "relative_speed": 1.0, "entity_type": 2},
        ],
    )
    # relative_speed 0.0/1.0 rescaled onto [20, 30] -> 20 and 30, nearest-bin
    # to themselves.
    counts = load_report_counts(report_path, speed_bins=[20, 30])
    assert np.array_equal(counts, [1, 1])


def test_load_report_counts_relative_speed_fallback_handles_constant_values(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": None, "relative_speed": 0.5, "entity_type": 2},
            {"absolute_speed": None, "relative_speed": 0.5, "entity_type": 2},
        ],
    )
    # Constant relative_speed has no range to rescale - both map to the
    # midpoint of speed_bins, nearest to 30.
    counts = load_report_counts(report_path, speed_bins=[20, 30, 40])
    assert np.array_equal(counts, [0, 2, 0])


def test_load_report_counts_prefers_absolute_speed_over_relative_speed(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 21.0, "relative_speed": 1.0, "entity_type": 2},
        ],
    )
    # If any entity has absolute_speed, it's used as-is (not rescaled) - a
    # relative_speed of 1.0 would otherwise dominate and land in the top bin.
    counts = load_report_counts(report_path, speed_bins=[20, 30])
    assert np.array_equal(counts, [1, 0])


def test_load_report_counts_raises_when_neither_speed_field_present(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(report_path, [{"entity_type": 2}])
    with pytest.raises(ValueError, match="absolute_speed.*relative_speed"):
        load_report_counts(report_path, speed_bins=[20, 30])


def test_load_report_counts_trims_absolute_speed_outside_bins_range(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 5.0, "entity_type": 2},  # below range, trimmed
            {"absolute_speed": 25.0, "entity_type": 2},  # in range
            {"absolute_speed": 90.0, "entity_type": 2},  # above range, trimmed
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30, 40])
    assert np.array_equal(counts, [1, 0, 0])


def test_load_report_counts_raises_when_all_entities_outside_bins_range(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 5.0, "entity_type": 2},
            {"absolute_speed": 90.0, "entity_type": 2},
        ],
    )
    with pytest.raises(ValueError, match="within speed_bins"):
        load_report_counts(report_path, speed_bins=[20, 30, 40])


def test_load_report_counts_keeps_values_exactly_at_bin_edges(tmp_path):
    report_path = tmp_path / "report.json"
    _write_report(
        report_path,
        [
            {"absolute_speed": 20.0, "entity_type": 2},
            {"absolute_speed": 40.0, "entity_type": 2},
        ],
    )
    counts = load_report_counts(report_path, speed_bins=[20, 30, 40])
    assert np.array_equal(counts, [1, 0, 1])


def test_format_final_statistics_reports_csv_variance_only_by_default():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
    )
    summary = _format_final_statistics(comparator)
    assert summary.startswith("CSV variance — mean")
    assert "Report match" not in summary


def test_format_final_statistics_includes_one_line_per_report():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
    )
    summary = _format_final_statistics(
        comparator,
        reference_counts=[[5, 10, 5], [0, 0, 10]],
        reference_labels=["Close Match", "Poor Match"],
    )
    lines = summary.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("CSV variance — mean")
    assert lines[1].startswith("Report match — Close Match:")
    assert lines[2].startswith("Report match — Poor Match:")


def test_format_final_statistics_defaults_report_labels():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5]],
    )
    summary = _format_final_statistics(comparator, reference_counts=[[5, 10, 5]])
    assert "Report match — Report 1:" in summary


def test_format_final_statistics_matching_report_has_low_divergence_high_overlap():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [5, 10, 5]],
    )
    summary = _format_final_statistics(
        comparator, reference_counts=[[5, 10, 5]], reference_labels=["Identical"]
    )
    line = next(l for l in summary.splitlines() if l.startswith("Report match"))
    js = float(line.split("JS div ")[1].split(",")[0])
    overlap = float(line.split("overlap ")[1])
    assert js == pytest.approx(0.0, abs=1e-3)
    assert overlap == pytest.approx(1.0, abs=1e-3)


def test_report_match_stats_returns_one_tuple_per_report():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
    )
    stats = _report_match_stats(
        comparator,
        reference_counts=[[5, 10, 5], [0, 0, 10]],
        reference_labels=["Close Match", "Poor Match"],
    )
    assert [label for label, _, _ in stats] == ["Close Match", "Poor Match"]
    close_js, close_overlap = stats[0][1], stats[0][2]
    poor_js, poor_overlap = stats[1][1], stats[1][2]
    assert close_js < poor_js
    assert close_overlap > poor_overlap


def test_report_match_stats_defaults_labels():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30], counts=[[5, 10, 5], [6, 11, 5]]
    )
    stats = _report_match_stats(comparator, reference_counts=[[5, 10, 5]])
    assert stats[0][0] == "Report 1"


def test_format_intuitive_summary_reports_pairs_within_threshold_when_all_pass():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
        labels=["d1", "d2", "d3"],
    )
    results = comparator.compare_all()
    summary = _format_intuitive_summary(
        "left", results, [], js_threshold=0.5, overlap_threshold=0.5, stable=True
    )
    assert "Pairs within threshold: 3/3" in summary
    assert "Verdict: STABLE" in summary
    assert "No --report given." in summary


def test_format_intuitive_summary_names_worst_pair_when_some_fail():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30, 40],
        counts=[[10, 0, 0, 0], [0, 0, 0, 10]],
        labels=["d1", "d2"],
    )
    results = comparator.compare_all()
    summary = _format_intuitive_summary(
        "left",
        results,
        [],
        js_threshold=0.1,
        overlap_threshold=0.85,
        stable=False,
    )
    assert "Pairs within threshold: 0/1" in summary
    assert "Worst pair: d1 vs d2" in summary
    assert "Verdict: NOT STABLE" in summary


def test_format_intuitive_summary_handles_single_row_with_no_pairs():
    summary = _format_intuitive_summary(
        "left", [], [], js_threshold=0.1, overlap_threshold=0.85, stable=True
    )
    assert "Pairs compared: 0" in summary


def test_format_intuitive_summary_labels_matching_and_diverging_reports():
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30], counts=[[5, 10, 5], [5, 10, 5]]
    )
    report_stats = _report_match_stats(
        comparator,
        reference_counts=[[5, 10, 5], [0, 0, 10]],
        reference_labels=["Close Match", "Poor Match"],
    )
    summary = _format_intuitive_summary(
        "left", [], report_stats, js_threshold=0.1, overlap_threshold=0.85, stable=True
    )
    close_line = next(l for l in summary.splitlines() if "Close Match" in l)
    poor_line = next(l for l in summary.splitlines() if "Poor Match" in l)
    assert close_line.strip().endswith("(PASS)")
    assert poor_line.strip().endswith("(FAIL)")


def test_plot_ridgeline_sets_axis_labels(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
        labels=["2026-01-01", "2026-01-02", "2026-01-03"],
    )
    captured = {}
    real_subplots = speed_distribution_comparator.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(speed_distribution_comparator.plt, "subplots", spy_subplots)

    plot_ridgeline(comparator, tmp_path / "ridgeline.png")

    assert captured["ax"].get_xlabel() == "Speed"
    assert captured["ax"].get_ylabel() == "Date"


def test_plot_ridgeline_highlight_last_n_adds_title_note_and_legend_entry(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6], [3, 8, 7], [2, 7, 8]],
        labels=["d1", "d2", "d3", "d4", "d5"],
    )
    captured = {}
    real_subplots = speed_distribution_comparator.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(speed_distribution_comparator.plt, "subplots", spy_subplots)

    plot_ridgeline(comparator, tmp_path / "ridgeline.png", highlight_last_n=2)

    assert "Last 2 Dates" in captured["ax"].get_title()
    legend_labels = [t.get_text() for t in captured["ax"].get_legend().get_texts()]
    assert "Last 2 Dates" in legend_labels
    assert "Car Counter" in legend_labels


def test_plot_ridgeline_no_last_n_omits_title_note_and_legend(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
        labels=["d1", "d2", "d3"],
    )
    captured = {}
    real_subplots = speed_distribution_comparator.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(speed_distribution_comparator.plt, "subplots", spy_subplots)

    plot_ridgeline(comparator, tmp_path / "ridgeline.png")

    assert "Last" not in captured["ax"].get_title()
    assert captured["ax"].get_legend() is None


def test_plot_ridgeline_draws_highlight_box_when_last_n_given(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6], [3, 8, 7], [2, 7, 8]],
        labels=["d1", "d2", "d3", "d4", "d5"],
    )
    calls = []
    real_rectangle = speed_distribution_comparator.PlotRectangle

    def spy_rectangle(*args, **kwargs):
        calls.append((args, kwargs))
        return real_rectangle(*args, **kwargs)

    monkeypatch.setattr(speed_distribution_comparator, "PlotRectangle", spy_rectangle)

    plot_ridgeline(comparator, tmp_path / "ridgeline.png", highlight_last_n=2)

    box_calls = [c for c in calls if c[1].get("edgecolor") == "black"]
    assert len(box_calls) == 1


def test_plot_ridgeline_no_highlight_box_by_default(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
        labels=["d1", "d2", "d3"],
    )
    calls = []
    monkeypatch.setattr(
        speed_distribution_comparator,
        "PlotRectangle",
        lambda *a, **k: calls.append((a, k)),
    )

    plot_ridgeline(comparator, tmp_path / "ridgeline.png")

    assert calls == []


def test_plot_ridgeline_highlight_last_n_clamped_to_row_count(tmp_path):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[5, 10, 5], [6, 11, 5], [4, 9, 6]],
        labels=["d1", "d2", "d3"],
    )
    output_path = tmp_path / "ridgeline.png"

    # highlight_last_n larger than the row count shouldn't raise.
    plot_ridgeline(comparator, output_path, highlight_last_n=100)

    assert output_path.exists()


def test_plot_ridgeline_highlight_survives_max_rows_subsampling(tmp_path, monkeypatch):
    comparator = SpeedDistributionComparator(
        speed_bins=[10, 20, 30],
        counts=[[i + 1, i + 2, i + 3] for i in range(50)],
        labels=[f"d{i}" for i in range(50)],
    )
    calls = []
    real_rectangle = speed_distribution_comparator.PlotRectangle

    def spy_rectangle(*args, **kwargs):
        calls.append((args, kwargs))
        return real_rectangle(*args, **kwargs)

    monkeypatch.setattr(speed_distribution_comparator, "PlotRectangle", spy_rectangle)

    # The last row is always kept by the even subsampling, so a highlight
    # box is still drawn even though most of the last-n rows get thinned out.
    plot_ridgeline(comparator, tmp_path / "ridgeline.png", max_rows=10, highlight_last_n=3)

    box_calls = [c for c in calls if c[1].get("edgecolor") == "black"]
    assert len(box_calls) == 1
