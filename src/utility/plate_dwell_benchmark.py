"""
Plate Dwell Time Benchmark

Compares src/processing/plate_dwellprofiler.py's dwell-time output when fed
ground-truth plate text (from plate_ground_truth_labeler.py) against its
output when fed the same pipeline's actual raw OCR readings - quantifying how
much OCR error propagates into the downstream dwell-time metric, not just
into plate-text accuracy (see plate_ocr_benchmark.py for that).
"""

import argparse
import base64
import io
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib

# Headless/non-interactive backend - this script generates chart images to a
# file or embeds them in HTML, it never shows a window, and the default
# backend would otherwise try (and fail) to open a display in a server/CI
# environment. Must be set before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.processing.plate_dwellprofiler import (
    PlateDetection,
    compute_average_dwell_time,
    compute_plate_dwell_times,
    extract_plate_detections,
    load_manifest,
)
from src.utility.htmlreport import html_heading, html_table, wrap_html_document
from src.utility.imgutils import get_timestamp
from src.utility.plate_ocr_benchmark import load_ground_truth

_STATUS_ORDER: Tuple[str, ...] = ("dropped", "split", "corrupted", "exact")


@dataclass
class CrossingComparison:
    """
    How one ground-truth crossing (a true plate with 2+ sightings) fared
    under the pipeline's actual OCR readings.
    """

    true_plate: str
    ground_truth_dwell_time: float
    ground_truth_num_sightings: int
    status: str
    ocr_dwell_time: Optional[float] = None
    ocr_num_sightings: Optional[int] = None
    ocr_texts_seen: List[str] = field(default_factory=list)

    @property
    def dwell_time_delta(self) -> Optional[float]:
        if self.ocr_dwell_time is None:
            return None
        return abs(self.ocr_dwell_time - self.ground_truth_dwell_time)


def _comparisons_from_json(crossings: List[Dict[str, Any]]) -> List[CrossingComparison]:
    """
    Reconstructs CrossingComparison objects from a summary_report's
    serialized "crossings" list, for re-deriving statistics (median,
    dwell accuracy) from a reloaded report without re-running OCR.
    """
    return [
        CrossingComparison(
            true_plate=c["true_plate"],
            ground_truth_dwell_time=c["ground_truth_dwell_time"],
            ground_truth_num_sightings=c["ground_truth_num_sightings"],
            status=c["status"],
            ocr_dwell_time=c["ocr_dwell_time"],
            ocr_num_sightings=c["ocr_num_sightings"],
            ocr_texts_seen=c["ocr_texts_seen"],
        )
        for c in crossings
    ]


def build_ground_truth_detections(
    ground_truth: Dict[str, str],
    input_folder: Path,
    manifest: Dict[Path, float],
) -> List[PlateDetection]:
    """
    Builds PlateDetections using each image's ground-truth plate text instead
    of an OCR reading, so the "known correct" dwell-time grouping can be
    computed and compared against what the pipeline's OCR actually produces.

    Args:
        ground_truth (Dict[str, str]): Relative image path -> ground-truth
            plate text, from plate_ocr_benchmark.load_ground_truth.
        input_folder (Path): Directory the relative paths are resolved against.
        manifest (Dict[Path, float]): Path -> timestamp, from
            plate_dwellprofiler.load_manifest.

    Returns:
        List[PlateDetection]: Detections sorted chronologically by timestamp.
    """
    detections = []
    for rel_path, true_text in ground_truth.items():
        img_path = input_folder / rel_path
        timestamp = manifest.get(img_path)
        if timestamp is None:
            timestamp = get_timestamp(img_path)
        detections.append(
            PlateDetection(plate_text=true_text, timestamp=timestamp, img_path=img_path)
        )
    detections.sort(key=lambda d: d.timestamp)
    return detections


def compare_crossings(
    ground_truth_matches: List[Dict[str, Any]],
    ground_truth_detections: List[PlateDetection],
    ocr_detections: List[PlateDetection],
    max_edit_distance: int = 0,
    max_time_gap: Optional[float] = None,
) -> List[CrossingComparison]:
    """
    Classifies every ground-truth crossing by how OCR-based grouping treated it.

    Ground truth is always grouped by exact text (it's the human-confirmed
    answer key - there's nothing to fuzzy-match). max_edit_distance/
    max_time_gap only apply to the OCR side, letting the pipeline's actual
    fuzzy-matching settings (see plate_dwellprofiler.compute_plate_dwell_times)
    be evaluated against ground truth before turning them on for real.

    Every image is looked up by which OCR-side CLUSTER (a match's index, or a
    single sighting's own slot) it landed in - not by its raw or consensus
    plate_text - since under fuzzy matching a cluster's plate_text (majority
    vote) doesn't necessarily equal every member's raw text, and two
    unrelated clusters could coincidentally share the same consensus text.

    Args:
        ground_truth_matches (List[Dict[str, Any]]): Dwell records from
            compute_plate_dwell_times(ground_truth_detections) - the "answer key".
        ground_truth_detections (List[PlateDetection]): All ground-truth
            detections (used to recover each true plate's full image set).
        ocr_detections (List[PlateDetection]): Detections built from the
            pipeline's actual OCR readings over the same images.
        max_edit_distance (int): Passed to compute_plate_dwell_times for the
            OCR-side grouping only. Defaults to 0 (exact match).
        max_time_gap (Optional[float]): Passed to compute_plate_dwell_times
            for the OCR-side grouping only. Defaults to None (no limit).

    Returns:
        List[CrossingComparison]: One entry per ground-truth crossing.
    """
    ocr_matches, ocr_singles = compute_plate_dwell_times(
        ocr_detections, max_edit_distance=max_edit_distance, max_time_gap=max_time_gap
    )

    # Cluster id -> (its match dict if it has one, its member paths, its
    # display text). Singles get their own ids continuing past the matches',
    # so a match and a single sighting can never collide.
    cluster_match: Dict[int, Optional[Dict[str, Any]]] = {}
    cluster_paths: Dict[int, Set[Path]] = {}
    cluster_text: Dict[int, str] = {}
    path_to_cluster: Dict[Path, int] = {}

    for cluster_id, m in enumerate(ocr_matches):
        paths = {Path(p) for p in m["images"]}
        cluster_match[cluster_id] = m
        cluster_paths[cluster_id] = paths
        cluster_text[cluster_id] = m["plate_text"]
        for p in paths:
            path_to_cluster[p] = cluster_id

    for offset, d in enumerate(ocr_singles, start=len(ocr_matches)):
        cluster_match[offset] = None
        cluster_paths[offset] = {d.img_path}
        cluster_text[offset] = d.plate_text
        path_to_cluster[d.img_path] = offset

    gt_paths_by_plate: Dict[str, Set[Path]] = defaultdict(set)
    for d in ground_truth_detections:
        gt_paths_by_plate[d.plate_text].add(d.img_path)

    comparisons: List[CrossingComparison] = []
    for match in ground_truth_matches:
        true_plate = match["plate_text"]
        true_paths = gt_paths_by_plate[true_plate]

        clusters_used = {path_to_cluster[p] for p in true_paths if p in path_to_cluster}

        if not clusters_used:
            comparisons.append(
                CrossingComparison(
                    true_plate,
                    match["dwell_time"],
                    match["num_sightings"],
                    status="dropped",
                )
            )
            continue

        if len(clusters_used) > 1:
            comparisons.append(
                CrossingComparison(
                    true_plate,
                    match["dwell_time"],
                    match["num_sightings"],
                    status="split",
                    ocr_texts_seen=sorted({cluster_text[c] for c in clusters_used}),
                )
            )
            continue

        cluster_id = next(iter(clusters_used))
        ocr_match = cluster_match[cluster_id]

        if ocr_match is None:
            # Only one of this plate's sightings survived OCR under a
            # consistent reading - a single sighting has no dwell time.
            comparisons.append(
                CrossingComparison(
                    true_plate,
                    match["dwell_time"],
                    match["num_sightings"],
                    status="corrupted",
                    ocr_num_sightings=len(cluster_paths[cluster_id]),
                    ocr_texts_seen=[cluster_text[cluster_id]],
                )
            )
            continue

        status = "exact" if cluster_paths[cluster_id] == true_paths else "corrupted"
        comparisons.append(
            CrossingComparison(
                true_plate,
                match["dwell_time"],
                match["num_sightings"],
                status=status,
                ocr_dwell_time=ocr_match["dwell_time"],
                ocr_num_sightings=ocr_match["num_sightings"],
                ocr_texts_seen=[cluster_text[cluster_id]],
            )
        )

    return comparisons


def split_fragment_texts(comparisons: List[CrossingComparison]) -> Set[str]:
    """
    Collects every OCR text that's a "split fragment" - part of a
    ground-truth crossing whose sightings were scattered across 2+ different
    misreads (status == "split").

    A fragment can still have 2+ sightings under its own misread and so form
    a real entry in compute_plate_dwell_times' matches - but it doesn't
    represent a genuine end-to-end crossing, just a wrong partial grouping,
    so it shouldn't count toward the OCR pipeline's average dwell time.

    Args:
        comparisons (List[CrossingComparison]): Per-crossing comparisons.

    Returns:
        Set[str]: Every OCR text seen on a "split" crossing.
    """
    fragment_texts: Set[str] = set()
    for c in comparisons:
        if c.status == "split":
            fragment_texts.update(c.ocr_texts_seen)
    return fragment_texts


def _status_summary(
    comparisons: List[CrossingComparison], min_dwell_time: float = 0.0
) -> Dict[str, Any]:
    """
    Aggregates a crossing-status breakdown shared by the console and HTML
    renderers, so both present the exact same numbers.

    Args:
        comparisons (List[CrossingComparison]): Per-crossing comparisons.
        min_dwell_time (float): Minimum ground-truth dwell time (in seconds)
            for a "corrupted" crossing to count toward mean_corrupted_delta -
            a near-zero true dwell is usually a few frames of one brief
            pass-by, so its delta against a bogus OCR-merged dwell is mostly
            noise, not a representative error. Defaults to 0.0 (no filtering).

    Returns:
        Dict[str, Any]: Total crossing count, a count per status, the mean
            absolute dwell-time delta among "corrupted" crossings (None if
            there are none left after filtering), and how many corrupted
            crossings were excluded from that mean for falling below
            min_dwell_time.
    """
    status_counts: Dict[str, int] = {status: 0 for status in _STATUS_ORDER}
    for c in comparisons:
        status_counts[c.status] += 1

    corrupted = [c for c in comparisons if c.status == "corrupted"]
    corrupted_deltas: List[float] = []
    for c in corrupted:
        delta = c.dwell_time_delta
        if delta is not None and c.ground_truth_dwell_time >= min_dwell_time:
            corrupted_deltas.append(delta)
    excluded_from_corrupted_delta = len(
        [
            c
            for c in corrupted
            if c.dwell_time_delta is not None
            and c.ground_truth_dwell_time < min_dwell_time
        ]
    )
    mean_corrupted_delta = (
        sum(corrupted_deltas) / len(corrupted_deltas) if corrupted_deltas else None
    )

    return {
        "total_crossings": len(comparisons),
        "status_counts": status_counts,
        "mean_corrupted_delta": mean_corrupted_delta,
        "excluded_from_corrupted_delta": excluded_from_corrupted_delta,
    }


def _counted_dwell_times(matches: List[Dict[str, Any]]) -> List[float]:
    """
    Extracts dwell_time from matches flagged counted_in_average - the shared
    "which numbers actually represent a real crossing" filter behind the
    average/median stats and the distribution histogram alike.

    A match missing the counted_in_average key entirely (a report saved
    before min_dwell_time/split-fragment exclusion existed) is treated as
    counted, so nothing goes missing.
    """
    return [m["dwell_time"] for m in matches if m.get("counted_in_average", True)]


def _mean_and_median_dwell(matches: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Computes the mean and median dwell time over matches flagged
    counted_in_average.

    Always derived fresh from the matches themselves rather than trusted
    from a stored statistic, so it works identically whether called right
    after a run or on a report reloaded from disk (see --from-report) - a
    report saved before median support existed still renders one correctly.

    Args:
        matches (List[Dict[str, Any]]): Dwell records, each with "dwell_time"
            and (usually) "counted_in_average".

    Returns:
        Tuple[float, float]: Mean and median dwell time in seconds, both 0.0
            if no matches are counted.
    """
    counted = _counted_dwell_times(matches)
    if not counted:
        return 0.0, 0.0
    return statistics.mean(counted), statistics.median(counted)


# Ground truth / OCR are always assigned these exact colors, everywhere they
# appear together (this chart's two series) - a fixed categorical assignment,
# not cycled, so the same series never repaints a different color.
_GT_COLOR = "#2a78d6"  # categorical slot 1 (blue)
_OCR_COLOR = "#e34948"  # categorical slot 6 (red) - the two are also the
# validated diverging pair, i.e. maximally distinct warm/cool poles, which is
# exactly the "read as opposite" contrast a two-series overlay wants.


def render_dwell_time_histogram(
    ground_truth_matches: List[Dict[str, Any]],
    ocr_matches: List[Dict[str, Any]],
) -> bytes:
    """
    Renders an overlaid histogram comparing the ground-truth and OCR dwell
    time distributions, as a PNG.

    Only matches flagged counted_in_average are plotted - the same
    min_dwell_time/split-fragment-filtered population the average/median
    stats are computed from (see _counted_dwell_times), so the chart is a
    visual explanation of those numbers rather than a different, inconsistent
    view of the data.

    Dwell times routinely span several orders of magnitude in this dataset (a
    fraction of a second up to a multi-hour crossing), so bins are log-spaced
    and the x-axis is log-scaled - a linear axis would crush nearly every bar
    into the first pixel and make the chart useless.

    Args:
        ground_truth_matches (List[Dict[str, Any]]): Ground-truth
            dwell_time_matches.
        ocr_matches (List[Dict[str, Any]]): OCR dwell_time_matches.

    Returns:
        bytes: PNG image data. Empty if neither side has a countable,
            positive dwell time to plot.
    """
    gt_values = [v for v in _counted_dwell_times(ground_truth_matches) if v > 0]
    ocr_values = [v for v in _counted_dwell_times(ocr_matches) if v > 0]

    if not gt_values and not ocr_values:
        return b""

    combined = gt_values + ocr_values
    low, high = min(combined), max(combined)
    bins = (
        np.logspace(np.log10(low * 0.9), np.log10(high * 1.1), 25)
        if low < high
        else np.array([low * 0.5, low * 1.5])
    )

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    if gt_values:
        ax.hist(
            gt_values,
            bins=bins,
            color=_GT_COLOR,
            alpha=0.55,
            label=f"Ground truth (n={len(gt_values)})",
            edgecolor=_GT_COLOR,
            linewidth=0.8,
        )
    if ocr_values:
        ax.hist(
            ocr_values,
            bins=bins,
            color=_OCR_COLOR,
            alpha=0.55,
            label=f"OCR (n={len(ocr_values)})",
            edgecolor=_OCR_COLOR,
            linewidth=0.8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Dwell time (seconds, log scale)", color="#52514e")
    ax.set_ylabel("Number of crossings", color="#52514e")
    ax.set_title(
        "Ground Truth vs OCR Dwell Time Distribution",
        color="#0b0b0b",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="#52514e")
    ax.grid(True, which="major", axis="both", color="#e3e2dd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right", labelcolor="#0b0b0b")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


_ACCURACY_BUCKETS: Tuple[str, ...] = ("<=5%", "<=20%", "<=50%", ">50%")


def compute_dwell_accuracy(
    comparisons: List[CrossingComparison],
    min_dwell_time: float = 0.0,
    tolerance: float = 0.2,
) -> Dict[str, Any]:
    """
    Measures how reasonable the dwell times the algorithm actually computed
    are - restricted to crossings where it produced a number at all
    ("exact" or "corrupted"; "split" and "dropped" never get an
    ocr_dwell_time, so there's nothing to judge for them).

    min_dwell_time excludes crossings with a near-zero (or zero) ground-truth
    dwell time from judgment, the same filter already applied to the plain
    averages and mean_corrupted_delta: dividing by a near-zero denominator
    turns a trivially small absolute difference into a huge, meaningless
    relative error, so those crossings would otherwise dominate the
    "unreasonable" bucket for reasons that have nothing to do with OCR
    quality.

    Args:
        comparisons (List[CrossingComparison]): Per-crossing comparisons.
        min_dwell_time (float): Minimum ground-truth dwell time (in seconds)
            for a crossing to be judged. Defaults to 0.0 (no filtering).
        tolerance (float): Maximum relative error
            (|ocr_dwell - gt_dwell| / gt_dwell) for a crossing to count as
            "reasonable". Defaults to 0.2 (20%).

    Returns:
        Dict[str, Any]: tolerance and min_dwell_time used, how many
            crossings were judged, how many were excluded for a near-zero
            ground-truth dwell, the reasonable count/rate, and a histogram
            of relative-error buckets (<=5%, <=20%, <=50%, >50%) among the
            judged crossings.
    """
    has_dwell = [c for c in comparisons if c.ocr_dwell_time is not None]
    judged = [
        c
        for c in has_dwell
        if c.ground_truth_dwell_time >= min_dwell_time and c.ground_truth_dwell_time > 0
    ]
    excluded_short_dwell = len(has_dwell) - len(judged)

    errors: List[float] = []
    for c in judged:
        ocr_dwell_time = c.ocr_dwell_time
        if ocr_dwell_time is not None:
            errors.append(
                abs(ocr_dwell_time - c.ground_truth_dwell_time)
                / c.ground_truth_dwell_time
            )

    counts: Dict[str, int] = {bucket: 0 for bucket in _ACCURACY_BUCKETS}
    for err in errors:
        if err <= 0.05:
            counts["<=5%"] += 1
        elif err <= 0.20:
            counts["<=20%"] += 1
        elif err <= 0.50:
            counts["<=50%"] += 1
        else:
            counts[">50%"] += 1

    total = len(judged)
    histogram = {
        bucket: {"count": count, "share": count / total if total else 0.0}
        for bucket, count in counts.items()
    }
    reasonable_count = sum(1 for err in errors if err <= tolerance)

    return {
        "tolerance": tolerance,
        "min_dwell_time": min_dwell_time,
        "total_judged": total,
        "excluded_short_dwell": excluded_short_dwell,
        "reasonable_count": reasonable_count,
        "reasonable_rate": reasonable_count / total if total else 0.0,
        "histogram": histogram,
    }


_BANNER_WIDTH = 46
_LABEL_WIDTH = 26


def _stat_line(label: str, value: str) -> str:
    return f"{label:<{_LABEL_WIDTH}}: {value}"


def _ordered_comparisons(
    comparisons: List[CrossingComparison],
) -> List[CrossingComparison]:
    return sorted(
        comparisons, key=lambda c: (_STATUS_ORDER.index(c.status), c.true_plate)
    )


def _print_dwell_accuracy(dwell_accuracy: Dict[str, Any]) -> None:
    if (
        not dwell_accuracy["total_judged"]
        and not dwell_accuracy["excluded_short_dwell"]
    ):
        return
    print()
    print("DWELL TIME ACCURACY".center(_BANNER_WIDTH, "-"))
    total = dwell_accuracy["total_judged"]
    tolerance_pct = dwell_accuracy["tolerance"] * 100
    if total:
        value = (
            f"{dwell_accuracy['reasonable_count']}/{total}"
            f"  ({dwell_accuracy['reasonable_rate'] * 100:.2f}%)"
        )
    else:
        value = "n/a (no judgeable crossings)"
    print(_stat_line(f"Reasonable (within {tolerance_pct:.0f}%)", value))
    if dwell_accuracy["excluded_short_dwell"]:
        print(
            _stat_line(
                "  - excluded (short/zero GT dwell)",
                str(dwell_accuracy["excluded_short_dwell"]),
            )
        )
    for bucket in _ACCURACY_BUCKETS:
        entry = dwell_accuracy["histogram"][bucket]
        print(
            _stat_line(
                f"  {bucket} error", f"{entry['count']}  ({entry['share'] * 100:.2f}%)"
            )
        )


def _print_summary(
    ground_truth_report: Dict[str, Any],
    ocr_report: Dict[str, Any],
    comparisons: List[CrossingComparison],
    status_summary: Dict[str, Any],
    dwell_accuracy: Dict[str, Any],
) -> None:
    total_crossings = status_summary["total_crossings"]

    print()
    print("=" * _BANNER_WIDTH)
    print("PLATE DWELL TIME BENCHMARK".center(_BANNER_WIDTH))
    print("=" * _BANNER_WIDTH)
    print(_stat_line("Ground truth crossings", str(total_crossings)))
    for status in _STATUS_ORDER:
        count = status_summary["status_counts"][status]
        share = f"{count / total_crossings * 100:.2f}%" if total_crossings else "-"
        print(_stat_line(f"  - {status}", f"{count}  ({share})"))
    print("-" * _BANNER_WIDTH)
    gt_stats = ground_truth_report["statistics"]
    gt_mean, gt_median = _mean_and_median_dwell(
        ground_truth_report["dwell_time_matches"]
    )
    print(
        _stat_line(
            "Ground truth avg dwell", f"{gt_mean:.2f}s  (median {gt_median:.2f}s)"
        )
    )
    if gt_stats.get("excluded_by_min_dwell_time"):
        print(
            _stat_line(
                "  - excluded (dwell < min)",
                str(gt_stats["excluded_by_min_dwell_time"]),
            )
        )

    ocr_stats = ocr_report["statistics"]
    ocr_mean, ocr_median = _mean_and_median_dwell(ocr_report["dwell_time_matches"])
    print(
        _stat_line(
            "OCR avg dwell (its matches)",
            f"{ocr_mean:.2f}s  (median {ocr_median:.2f}s)",
        )
    )
    if ocr_stats.get("excluded_by_min_dwell_time"):
        print(
            _stat_line(
                "  - excluded (dwell < min)",
                str(ocr_stats["excluded_by_min_dwell_time"]),
            )
        )
    if ocr_stats.get("excluded_by_split_fragment"):
        print(
            _stat_line(
                "  - excluded (split fragments)",
                str(ocr_stats["excluded_by_split_fragment"]),
            )
        )
    if status_summary["mean_corrupted_delta"] is not None:
        print(
            _stat_line(
                "Mean |delta| (corrupted only)",
                f"{status_summary['mean_corrupted_delta']:.2f}s",
            )
        )
        if status_summary.get("excluded_from_corrupted_delta"):
            print(
                _stat_line(
                    "  - excluded (short GT dwell)",
                    str(status_summary["excluded_from_corrupted_delta"]),
                )
            )
    print("=" * _BANNER_WIDTH)

    _print_dwell_accuracy(dwell_accuracy)

    if comparisons:
        print()
        print("CROSSINGS".center(_BANNER_WIDTH, "-"))
        plate_w = max(len("Plate"), max(len(c.true_plate) for c in comparisons))
        header = (
            f"{'Plate':<{plate_w}}  {'Status':<9}  {'GT Dwell':>9}  "
            f"{'OCR Dwell':>9}  {'Delta':>8}"
        )
        print(header)
        print("-" * len(header))
        for c in _ordered_comparisons(comparisons):
            ocr_dwell = (
                f"{c.ocr_dwell_time:.2f}s" if c.ocr_dwell_time is not None else "-"
            )
            delta = (
                f"{c.dwell_time_delta:.2f}s" if c.dwell_time_delta is not None else "-"
            )
            print(
                f"{c.true_plate:<{plate_w}}  {c.status:<9}  "
                f"{c.ground_truth_dwell_time:>8.2f}s  {ocr_dwell:>9}  {delta:>8}"
            )
    print()


def render_html_report(
    summary_report: Dict[str, Any],
    min_dwell_time: float = 0.0,
    tolerance: float = 0.2,
) -> str:
    """
    Renders a plate dwell benchmark summary_report as a standalone HTML page
    of tables, meant to be opened in a browser and copy-pasted (select-all,
    copy) directly into an Outlook email.

    Args:
        summary_report (Dict[str, Any]): The dict returned by
            run_plate_dwell_benchmark.
        min_dwell_time (float): Passed to compute_dwell_accuracy - see there.
            Defaults to 0.0 (no filtering).
        tolerance (float): Passed to compute_dwell_accuracy - see there.
            Defaults to 0.2 (20%).

    Returns:
        str: A complete HTML document.
    """
    ground_truth_stats = summary_report["ground_truth_run"]["statistics"]
    ocr_stats = summary_report["ocr_run"]["statistics"]
    status_summary = summary_report["status_summary"]
    crossings = summary_report["crossings"]

    dwell_accuracy = compute_dwell_accuracy(
        _comparisons_from_json(crossings),
        min_dwell_time=min_dwell_time,
        tolerance=tolerance,
    )

    gt_mean, gt_median = _mean_and_median_dwell(
        summary_report["ground_truth_run"]["dwell_time_matches"]
    )
    ocr_mean, ocr_median = _mean_and_median_dwell(
        summary_report["ocr_run"]["dwell_time_matches"]
    )

    overview_rows = [
        ["Ground truth crossings", status_summary["total_crossings"]],
    ]
    for status in _STATUS_ORDER:
        count = status_summary["status_counts"][status]
        total = status_summary["total_crossings"]
        share = f"{count / total * 100:.2f}%" if total else "-"
        overview_rows.append([f"  - {status}", f"{count}  ({share})"])
    overview_rows.append(
        ["Ground truth avg dwell", f"{gt_mean:.2f}s  (median {gt_median:.2f}s)"]
    )
    if ground_truth_stats.get("excluded_by_min_dwell_time"):
        overview_rows.append(
            [
                "  - excluded (dwell < min)",
                str(ground_truth_stats["excluded_by_min_dwell_time"]),
            ]
        )
    overview_rows.append(
        ["OCR avg dwell (its matches)", f"{ocr_mean:.2f}s  (median {ocr_median:.2f}s)"]
    )
    if ocr_stats.get("excluded_by_min_dwell_time"):
        overview_rows.append(
            [
                "  - excluded (dwell < min)",
                str(ocr_stats["excluded_by_min_dwell_time"]),
            ]
        )
    if ocr_stats.get("excluded_by_split_fragment"):
        overview_rows.append(
            [
                "  - excluded (split fragments)",
                str(ocr_stats["excluded_by_split_fragment"]),
            ]
        )
    if status_summary["mean_corrupted_delta"] is not None:
        overview_rows.append(
            [
                "Mean |delta| (corrupted only)",
                f"{status_summary['mean_corrupted_delta']:.2f}s",
            ]
        )
        if status_summary.get("excluded_from_corrupted_delta"):
            overview_rows.append(
                [
                    "  - excluded (short GT dwell)",
                    str(status_summary["excluded_from_corrupted_delta"]),
                ]
            )
    overview_table = html_table(["Metric", "Value"], overview_rows)

    sections = [
        html_heading("Plate Dwell Time Benchmark", level=2),
        overview_table,
    ]

    histogram_png = render_dwell_time_histogram(
        summary_report["ground_truth_run"]["dwell_time_matches"],
        summary_report["ocr_run"]["dwell_time_matches"],
    )
    if histogram_png:
        histogram_b64 = base64.b64encode(histogram_png).decode("ascii")
        sections.append(html_heading("Dwell Time Distribution", level=3))
        sections.append(
            f'<img src="data:image/png;base64,{histogram_b64}" '
            f'alt="Overlaid histogram of ground truth vs OCR dwell time distributions" '
            f'style="max-width:100%; height:auto;">'
        )

    if dwell_accuracy["total_judged"] or dwell_accuracy["excluded_short_dwell"]:
        total = dwell_accuracy["total_judged"]
        tolerance_pct = dwell_accuracy["tolerance"] * 100
        if total:
            reasonable_value = (
                f"{dwell_accuracy['reasonable_count']}/{total}"
                f"  ({dwell_accuracy['reasonable_rate'] * 100:.2f}%)"
            )
        else:
            reasonable_value = "n/a (no judgeable crossings)"
        accuracy_rows = [
            [f"Reasonable (within {tolerance_pct:.0f}%)", reasonable_value],
        ]
        if dwell_accuracy["excluded_short_dwell"]:
            accuracy_rows.append(
                [
                    "  - excluded (short/zero GT dwell)",
                    str(dwell_accuracy["excluded_short_dwell"]),
                ]
            )
        for bucket in _ACCURACY_BUCKETS:
            entry = dwell_accuracy["histogram"][bucket]
            accuracy_rows.append(
                [
                    f"  {bucket} error",
                    f"{entry['count']}  ({entry['share'] * 100:.2f}%)",
                ]
            )
        sections.append(html_heading("Dwell Time Accuracy", level=3))
        sections.append(html_table(["Metric", "Value"], accuracy_rows))

    if crossings:
        ordered = sorted(
            crossings,
            key=lambda c: (_STATUS_ORDER.index(c["status"]), c["true_plate"]),
        )
        crossings_table = html_table(
            ["Plate", "Status", "GT Dwell", "OCR Dwell", "Delta"],
            [
                [
                    c["true_plate"],
                    c["status"],
                    f"{c['ground_truth_dwell_time']:.2f}s",
                    (
                        f"{c['ocr_dwell_time']:.2f}s"
                        if c["ocr_dwell_time"] is not None
                        else "-"
                    ),
                    (
                        f"{c['dwell_time_delta']:.2f}s"
                        if c["dwell_time_delta"] is not None
                        else "-"
                    ),
                ]
                for c in ordered
            ],
        )
        sections.append(html_heading("Crossings", level=3))
        sections.append(crossings_table)

    return wrap_html_document(sections)


def run_plate_dwell_benchmark(
    ground_truth_path: Union[str, Path],
    input_dir: Optional[Union[str, Path]] = None,
    report: Optional[Union[str, Path]] = None,
    html_report: Optional[Union[str, Path]] = None,
    histogram: Optional[Union[str, Path]] = None,
    min_dwell_time: float = 0.0,
    tolerance: float = 0.2,
    max_edit_distance: int = 0,
    max_time_gap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Runs plate_dwellprofiler.py's dwell-time computation twice - once on
    ground-truth plate text, once on the pipeline's real OCR readings, both
    restricted to the same ground-truthed image set - and compares them.

    Args:
        ground_truth_path (Union[str, Path]): Path to the ground-truth JSON
            produced by plate_ground_truth_labeler.py.
        input_dir (Optional[Union[str, Path]]): Directory the ground truth's
            relative image paths (and its plate_manifest.json, if present)
            are resolved against. Defaults to the ground truth file's parent
            directory.
        report (Optional[Union[str, Path]]): Filepath to save a detailed
            JSON report. Defaults to None.
        html_report (Optional[Union[str, Path]]): Filepath to save an HTML
            version of the summary, with real <table> elements that survive
            copy-paste into Outlook (open in a browser, select all, copy,
            paste). The same overlaid histogram saved by histogram= is also
            embedded in this page. Defaults to None.
        histogram (Optional[Union[str, Path]]): Filepath to save a standalone
            PNG of the overlaid ground-truth-vs-OCR dwell time histogram -
            see render_dwell_time_histogram. Defaults to None.
        min_dwell_time (float): Minimum dwell time (in seconds) for a match
            to count toward the ground-truth average, the OCR average, the
            corrupted-crossings mean delta, and the dwell-time accuracy
            judgment - see compute_average_dwell_time, _status_summary, and
            compute_dwell_accuracy. Excluded values are still shown in full
            in the raw report, just left out of these aggregates. Defaults
            to 0.0 (no filtering).
        tolerance (float): Maximum relative error for a crossing's OCR dwell
            time to count as "reasonable" - see compute_dwell_accuracy.
            Defaults to 0.2 (20%).
        max_edit_distance (int): Passed to compute_plate_dwell_times for the
            OCR-side grouping only - ground truth always groups by exact
            text. Lets plate_dwellprofiler.py's fuzzy-matching setting be
            evaluated against ground truth before turning it on for real.
            Defaults to 0 (exact match, today's pipeline behavior).
        max_time_gap (Optional[float]): Passed to compute_plate_dwell_times
            for the OCR-side grouping only. Defaults to None (no limit).

    Returns:
        Dict[str, Any]: Summary dict containing both runs' dwell-time
            reports and the per-crossing comparison.
    """
    ground_truth_path = Path(ground_truth_path)
    input_folder = Path(input_dir) if input_dir else ground_truth_path.parent

    ground_truth = load_ground_truth(ground_truth_path)
    manifest = load_manifest(input_folder)

    ground_truth_detections = build_ground_truth_detections(
        ground_truth, input_folder, manifest
    )
    ocr_detections = extract_plate_detections(
        [input_folder / rel_path for rel_path in ground_truth], manifest=manifest
    )

    gt_matches, gt_singles = compute_plate_dwell_times(ground_truth_detections)
    ocr_matches, ocr_singles = compute_plate_dwell_times(
        ocr_detections, max_edit_distance=max_edit_distance, max_time_gap=max_time_gap
    )

    comparisons = compare_crossings(
        gt_matches,
        ground_truth_detections,
        ocr_detections,
        max_edit_distance=max_edit_distance,
        max_time_gap=max_time_gap,
    )
    status_summary = _status_summary(comparisons, min_dwell_time=min_dwell_time)
    dwell_accuracy = compute_dwell_accuracy(
        comparisons, min_dwell_time=min_dwell_time, tolerance=tolerance
    )
    fragment_texts = split_fragment_texts(comparisons)

    def _stats(
        matches, singles, total_images, excluded_texts: AbstractSet[str] = frozenset()
    ):
        # compute_average_dwell_time sets counted_in_average per-match based
        # on min_dwell_time; layer the split-fragment exclusion (OCR only) on
        # top, tracking each reason separately so the report doesn't mislabel
        # a short-dwell exclusion as a split fragment or vice versa.
        _, excluded_by_min_dwell_time = compute_average_dwell_time(
            matches, min_dwell_time=min_dwell_time
        )
        excluded_by_split_fragment = 0
        for m in matches:
            if m["counted_in_average"] and m["plate_text"] in excluded_texts:
                m["counted_in_average"] = False
                excluded_by_split_fragment += 1
        counted = [m for m in matches if m["counted_in_average"]]
        avg_dwell, median_dwell = _mean_and_median_dwell(matches)
        return {
            "input_images": total_images,
            "matched_crossings": len(matches),
            "single_sightings": len(singles),
            "min_dwell_time": min_dwell_time,
            "matches_excluded_from_average": len(matches) - len(counted),
            "excluded_by_min_dwell_time": excluded_by_min_dwell_time,
            "excluded_by_split_fragment": excluded_by_split_fragment,
            "average_dwell_time": avg_dwell,
            "median_dwell_time": median_dwell,
        }

    ground_truth_report: Dict[str, Any] = {
        "statistics": _stats(gt_matches, gt_singles, len(ground_truth_detections)),
        "dwell_time_matches": gt_matches,
    }
    ocr_report: Dict[str, Any] = {
        "statistics": _stats(
            ocr_matches,
            ocr_singles,
            len(ocr_detections),
            excluded_texts=fragment_texts,
        ),
        "dwell_time_matches": ocr_matches,
    }

    _print_summary(
        ground_truth_report, ocr_report, comparisons, status_summary, dwell_accuracy
    )

    summary_report: Dict[str, Any] = {
        "metadata": {
            "ground_truth_path": str(ground_truth_path),
            "input_dir": str(input_folder),
            "max_edit_distance": max_edit_distance,
            "max_time_gap": max_time_gap,
        },
        "ground_truth_run": ground_truth_report,
        "ocr_run": ocr_report,
        "status_summary": status_summary,
        "dwell_accuracy": dwell_accuracy,
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

    if report:
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    if histogram:
        with open(histogram, "wb") as f:
            f.write(
                render_dwell_time_histogram(
                    ground_truth_report["dwell_time_matches"],
                    ocr_report["dwell_time_matches"],
                )
            )

    if html_report:
        with open(html_report, "w") as f:
            f.write(
                render_html_report(
                    summary_report, min_dwell_time=min_dwell_time, tolerance=tolerance
                )
            )

    return summary_report


def render_saved_report(
    report_path: Union[str, Path],
    html_report: Optional[Union[str, Path]] = None,
    histogram: Optional[Union[str, Path]] = None,
    min_dwell_time: float = 0.0,
    tolerance: float = 0.2,
) -> Dict[str, Any]:
    """
    Reloads a JSON report previously saved via run_plate_dwell_benchmark's
    report= argument and re-renders its console summary (and, if given, an
    HTML page and/or histogram PNG) without re-running OCR.

    Median dwell time, the corrupted-crossings mean delta, and dwell-time
    accuracy are all recomputed fresh from the saved crossings using
    min_dwell_time/tolerance as given here - so a report can be re-examined
    under a different threshold without paying for OCR again. The plain
    average/median dwell time and their "excluded (dwell < min)" counts are
    the one exception: those depend on each match's counted_in_average flag,
    which was fixed at whatever min_dwell_time the original run used, and
    isn't recomputed here - re-run run_plate_dwell_benchmark for a fresh
    min_dwell_time to affect those too. The histogram is drawn from those same
    counted_in_average flags, so it reflects the original run's threshold too.

    Args:
        report_path (Union[str, Path]): Path to a JSON file previously
            written by run_plate_dwell_benchmark's report= argument.
        html_report (Optional[Union[str, Path]]): Filepath to save an HTML
            version of the summary. Defaults to None.
        histogram (Optional[Union[str, Path]]): Filepath to save a standalone
            PNG of the overlaid dwell time histogram. Defaults to None.
        min_dwell_time (float): Passed to _status_summary and
            compute_dwell_accuracy - see there. Defaults to 0.0 (no filtering).
        tolerance (float): Passed to compute_dwell_accuracy - see there.
            Defaults to 0.2 (20%).

    Returns:
        Dict[str, Any]: The reloaded summary report dict, unchanged.
    """
    with open(report_path, "r") as f:
        summary_report = json.load(f)

    ground_truth_report = summary_report["ground_truth_run"]
    ocr_report = summary_report["ocr_run"]
    comparisons = _comparisons_from_json(summary_report["crossings"])
    status_summary = _status_summary(comparisons, min_dwell_time=min_dwell_time)
    dwell_accuracy = compute_dwell_accuracy(
        comparisons, min_dwell_time=min_dwell_time, tolerance=tolerance
    )

    _print_summary(
        ground_truth_report, ocr_report, comparisons, status_summary, dwell_accuracy
    )

    if histogram:
        with open(histogram, "wb") as f:
            f.write(
                render_dwell_time_histogram(
                    ground_truth_report["dwell_time_matches"],
                    ocr_report["dwell_time_matches"],
                )
            )

    if html_report:
        with open(html_report, "w") as f:
            f.write(
                render_html_report(
                    summary_report, min_dwell_time=min_dwell_time, tolerance=tolerance
                )
            )

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the plate dwell time benchmark script.

    Raises:
        SystemExit: If the ground-truth file is missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Compare plate_dwellprofiler.py's dwell-time output on "
        "ground-truth plate text against its output on the pipeline's real "
        "OCR readings, to quantify how much OCR error propagates into the "
        "dwell-time metric."
    )
    parser.add_argument(
        "ground_truth",
        type=str,
        nargs="?",
        default=None,
        help="Path to the ground-truth JSON file. Omit if using --from-report.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Directory the ground truth's relative image paths are resolved "
        "against. Defaults to the ground truth file's parent directory.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save a detailed JSON report.",
    )
    parser.add_argument(
        "-w",
        "--html",
        type=str,
        default=None,
        help="Path to save an HTML version of the summary. Open it in a "
        "browser and copy/paste the tables directly into an Outlook email. "
        "Includes the same overlaid histogram as --histogram.",
    )
    parser.add_argument(
        "--histogram",
        type=str,
        default=None,
        help="Path to save a standalone PNG of the overlaid ground-truth-vs-"
        "OCR dwell time histogram (log-scaled, since dwell times span several "
        "orders of magnitude).",
    )
    parser.add_argument(
        "-m",
        "--min-dwell-time",
        type=float,
        default=0.0,
        help="Minimum ground-truth dwell time in seconds for a match/crossing "
        "to count toward the ground-truth average, the OCR average, the "
        "corrupted-crossings mean delta, and the dwell-time accuracy judgment "
        "- shorter ones are still listed in full, just excluded from these "
        "aggregates. Defaults to 0.0 (no filtering).",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.2,
        help="Maximum relative error for a crossing's OCR dwell time to count "
        'as "reasonable" in the dwell-time accuracy summary. Defaults to '
        "0.2 (20%%).",
    )
    parser.add_argument(
        "-e",
        "--max-edit-distance",
        type=int,
        default=0,
        help="Applies plate_dwellprofiler.py's fuzzy plate matching to the "
        "OCR side only (ground truth always groups by exact text), so its "
        "effect can be evaluated against ground truth before turning it on "
        "for real - see plate_dwellprofiler.py's --max-edit-distance. "
        "Defaults to 0 (exact match, today's pipeline behavior).",
    )
    parser.add_argument(
        "-g",
        "--max-time-gap",
        type=float,
        default=None,
        help="Applies plate_dwellprofiler.py's fuzzy-matching time-gap guard "
        "to the OCR side only. Only meaningful when --max-edit-distance > 0. "
        "Defaults to no limit.",
    )
    parser.add_argument(
        "--from-report",
        type=str,
        default=None,
        help="Path to a previously saved --report JSON to re-render (console, "
        "and --html if given) without re-running OCR. ground_truth/"
        "--input-dir/--max-edit-distance/--max-time-gap are ignored (the "
        "matching choice is baked into the saved crossings), but "
        "--min-dwell-time/--tolerance still apply, recomputed fresh.",
    )
    args = parser.parse_args()

    if args.from_report:
        from_report_path = Path(args.from_report)
        if not from_report_path.is_file():
            parser.error(f"{from_report_path} is not a file.")
        render_saved_report(
            from_report_path,
            html_report=args.html,
            histogram=args.histogram,
            min_dwell_time=args.min_dwell_time,
            tolerance=args.tolerance,
        )
        return

    if not args.ground_truth:
        parser.error("ground_truth is required unless --from-report is given.")

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.is_file():
        parser.error(f"{ground_truth_path} is not a file.")

    run_plate_dwell_benchmark(
        ground_truth_path=ground_truth_path,
        input_dir=args.input_dir,
        report=args.report,
        html_report=args.html,
        histogram=args.histogram,
        min_dwell_time=args.min_dwell_time,
        tolerance=args.tolerance,
        max_edit_distance=args.max_edit_distance,
        max_time_gap=args.max_time_gap,
    )


if __name__ == "__main__":
    main()
