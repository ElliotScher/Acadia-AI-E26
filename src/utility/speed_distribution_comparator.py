"""
Speed Distribution Comparator

Compares vehicle speed histograms sampled on different days to test whether
the underlying speed distribution is stable over time, rather than drifting
(e.g. because of a seasonal traffic pattern, a construction detour, or a
change in enforcement).

Input is a CSV shaped like:

    date,direction,10,15,20,25,30,35,40
    2026-01-01,left,3,12,48,61,20,4,1
    2026-01-01,right,1,9,40,58,22,5,0
    2026-01-02,left,2,15,45,58,24,3,0
    2026-01-02,right,4,11,42,55,20,4,0

The header row's first two cells are column labels (date, direction), and
the rest are the speed box (bin) values in mph shared by every histogram.
Every subsequent row is one day's, one direction's vehicle counts per bin -
each date appears once per direction. See load_directional_csv, which splits
the file by direction into its own SpeedDistributionComparator per
direction, so left- and right-traveling traffic are never compared against
each other or blended into the same histogram.

Two rows are compared with:
  - Jensen-Shannon divergence: treats each row as a probability distribution
    over speed bins (normalizing out sample size) and measures how much
    probability mass would have to move to turn one into the other. Bounded
    in [0, 1] bit (using log base 2), symmetric, and defined even when a bin
    is empty in one histogram but not the other - unlike KL divergence.
    Log-weighted (information-theoretic), so it doesn't treat every unit of
    misplaced mass as equally "costly".
  - Overlap coefficient: the shared probability mass between the two
    distributions (sum of the per-bin minimum), bounded in [0, 1] - 1.0 means
    identical distributions, 0.0 means disjoint support. Equivalent to
    1 - total variation distance: a simpler, purely geometric "what fraction
    of the mass do these two histograms have in common" measure, with no log
    weighting.
  - Pearson correlation: how linearly the bin-to-bin shape of one histogram's
    raw counts tracks the other's. Computed and reported for reference, but
    not part of the is_stable decision - it's a coarser, linear-trend notion
    of "shape" than the other two metrics' actual distributional distance,
    and can stay high even when meaningful probability mass has shifted
    between bins.

is_stable gates on both Jensen-Shannon divergence and overlap coefficient,
since they weight shape differences differently (log-weighted vs. linear) -
requiring both catches cases where one alone might read as "close enough".

The CLI runs everything above once per direction and keeps them entirely
separate: two comparison tables, two is_stable verdicts, and two ridgeline
plots (see plot_ridgeline) - the standard chart for "how does a distribution
change across many time-ordered samples", one filled curve per row stacked
with a slight overlap. A shifting peak, a widening spread, or a second hump
appearing partway up the stack is a drift you see directly in the shape,
not a number inferred from a summary metric. Passing one or more --report
flags additionally bins each report_recalibrator.py JSON report's
individual_entities onto the same speed_bins (see load_report_counts),
filtered to that direction's entities, and draws each as its own extra ridge
below the historical stack, in its own color, so specific reports can be
checked against the historical shape - and against each other - at a
glance. Each row's date is used as its y-axis tick label. --last-n is
applied per direction - "the last N rows" of a direction's own comparator,
i.e. the last N dates observed for that direction specifically - and each
plot keeps drawing that direction's full history, with the tested subset
outlined in a black box instead of hidden.

After every direction's tables/plots, the CLI prints a "Final Report" - one
section per direction (see _format_intuitive_summary) with the same numbers
already computed above, regrouped under two labeled headings instead of
left to infer from the raw per-pair table: "Ground truth stability
(date-to-date)" (from compare_all() - whether the ground truth's own speed
distribution held steady across the sampled dates, which is_stable and the
printed verdict are based on) and "Report match (vs. ground truth average)"
(from _report_match_stats - whether each --report's speeds match what the
ground truth usually looks like). These are separate numbers - a report can
diverge from the ground truth even when the ground truth itself is
perfectly STABLE, and vice versa.
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

# Headless/non-interactive backend - this module generates a chart image to a
# file, it never shows a window, and the default backend would otherwise try
# (and fail) to open a display in a server/CI environment. Must be set
# before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle as PlotRectangle
import numpy as np

logger = logging.getLogger("speed_distribution_comparator")

DEFAULT_JS_THRESHOLD = 0.1
DEFAULT_OVERLAP_THRESHOLD = 0.85

# Same chart palette established by report_recalibrator.py's speed
# histogram, direction_benchmark_common.py, and vehicle_speed_benchmark.py -
# the fixed "primary data" blue, on the same neutral surface/gridline/axis/
# text scheme used everywhere else in this project's charts.
RIDGELINE_LINE_COLOR = "#2a78d6"
RIDGELINE_GRIDLINE_COLOR = "#e1e0d9"
RIDGELINE_AXIS_COLOR = "#c3c2b7"
RIDGELINE_MUTED_TEXT_COLOR = "#898781"
RIDGELINE_PRIMARY_TEXT_COLOR = "#0b0b0b"
RIDGELINE_SURFACE_COLOR = "#fcfcfb"

# Reports loaded on top of the historical stack aren't more rows in the
# time-ordered sequence, so they don't take colors off the sequential ramp
# below - each gets its own slot off this project's validated 8-hue
# categorical palette instead (dataviz skill's references/palette.md),
# fixed order, never reassigned by data: slot 1 (blue) is already the
# historical ridges' identity, so reports start at slot 6 (red - the same
# color a single report used before, and the same "thing being checked
# against the data" identity direction_benchmark_common.PRED_COLOR /
# vehicle_speed_benchmark._PRED_COLOR use elsewhere in this codebase) and
# proceed through the remaining slots in a fixed order.
RIDGELINE_REPORT_PALETTE: List[str] = [
    "#e34948",  # slot 6 red
    "#eb6834",  # slot 8 orange
    "#e87ba4",  # slot 7 magenta
    "#eda100",  # slot 3 yellow
    "#1baf7a",  # slot 2 aqua
    "#4a3aa7",  # slot 5 violet
    "#008300",  # slot 4 green
]

# Sequential single-hue ramp (light tint -> the same primary data blue
# above) coloring each ridge by its row's position in time, so the stack
# reads earliest-to-latest at a glance in addition to top-to-bottom.
# Ridgeline plots don't have the order-dependent-compositing problem a
# single shared-axis overlay does - each ridge occupies its own vertical
# band, and one ridge partially covering the ridge behind it is the whole
# point (the "mountain range" look the chart is named for), not an
# artifact to work around.
_RIDGELINE_CMAP = LinearSegmentedColormap.from_list(
    "speed_distribution_ridgeline", ["#dce9f9", RIDGELINE_LINE_COLOR]
)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    # 0 * log(0/q) is defined as 0 in the standard KL/JS convention - mask
    # out bins with no probability mass in p instead of letting log2(0)
    # produce -inf/nan.
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))


def _overlap_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.minimum(p, q)))


@dataclass
class ComparisonResult:
    """
    Similarity of one pair of speed histograms.

    Args:
        label_a (str): Label (or row index, if the CSV had none) of the
            first histogram.
        label_b (str): Label (or row index, if the CSV had none) of the
            second histogram.
        pearson_correlation (float): Pearson correlation coefficient between
            the two rows' raw bin counts, in [-1.0, 1.0]. Reported for
            reference only - see is_stable.
        jensen_shannon_divergence (float): Jensen-Shannon divergence (base 2,
            so bounded in [0.0, 1.0]) between the two rows, each normalized
            into a probability distribution over bins.
        overlap_coefficient (float): Shared probability mass between the two
            rows' distributions, in [0.0, 1.0]. 1.0 means identical
            distributions, 0.0 means disjoint support.
    """

    label_a: str
    label_b: str
    pearson_correlation: float
    jensen_shannon_divergence: float
    overlap_coefficient: float

    def is_stable(
        self,
        js_threshold: float = DEFAULT_JS_THRESHOLD,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    ) -> bool:
        """
        Checks whether this pair is close enough to call the same distribution.

        Gates on Jensen-Shannon divergence and overlap coefficient -
        pearson_correlation is reported on the result but isn't part of this
        decision (see the module docstring for why).

        Args:
            js_threshold (float): Maximum acceptable Jensen-Shannon divergence.
                Defaults to DEFAULT_JS_THRESHOLD.
            overlap_threshold (float): Minimum acceptable overlap coefficient.
                Defaults to DEFAULT_OVERLAP_THRESHOLD.

        Returns:
            bool: True if jensen_shannon_divergence is within js_threshold
                and overlap_coefficient is within overlap_threshold.
        """
        return (
            self.jensen_shannon_divergence <= js_threshold
            and self.overlap_coefficient >= overlap_threshold
        )


class SpeedDistributionComparator:
    """
    Loads a CSV of per-day vehicle speed histograms sharing a common set of
    speed bins, and computes pairwise Jensen-Shannon divergence and overlap
    coefficient (plus Pearson correlation for reference) to test whether the
    speed distribution's shape held constant across samples.

    Args:
        speed_bins (Sequence[float]): The speed box values (mph) shared by
            every histogram, in the same order as each row's counts.
        counts (Sequence[Sequence[float]]): One row of raw vehicle counts
            per histogram, each the same length as speed_bins.
        labels (Optional[Sequence[str]]): One label per row (e.g. a date),
            for identifying rows in results. Defaults to the row's index
            (as a string) when not given.

    Raises:
        ValueError: If speed_bins is empty, no count rows are given, or any
            row's length doesn't match speed_bins.
    """

    def __init__(
        self,
        speed_bins: Sequence[float],
        counts: Sequence[Sequence[float]],
        labels: Optional[Sequence[str]] = None,
    ) -> None:
        if len(speed_bins) == 0:
            raise ValueError("speed_bins must contain at least one bin.")
        if len(counts) == 0:
            raise ValueError("At least one histogram row is required.")
        for i, row in enumerate(counts):
            if len(row) != len(speed_bins):
                raise ValueError(
                    f"Row {i} has {len(row)} values, expected {len(speed_bins)} "
                    "to match speed_bins."
                )

        self.speed_bins: np.ndarray = np.asarray(speed_bins, dtype=float)
        self.counts: np.ndarray = np.asarray(counts, dtype=float)
        self.labels: List[str] = (
            list(labels) if labels is not None else [str(i) for i in range(len(counts))]
        )
        if len(self.labels) != len(self.counts):
            raise ValueError("labels must have one entry per histogram row.")

    def tail(self, n: int) -> "SpeedDistributionComparator":
        """
        Restricts this comparator to just its last n rows (e.g. the most
        recent n days), for checking whether the distribution has settled
        into a stable regime even when the full history hasn't.

        Args:
            n (int): Number of most recent rows to keep. If greater than the
                number of rows available, every row is kept.

        Returns:
            SpeedDistributionComparator: New comparator over just the last n
                rows (and their labels), sharing this comparator's speed_bins.

        Raises:
            ValueError: If n is less than 1.
        """
        if n < 1:
            raise ValueError("n must be at least 1.")
        return SpeedDistributionComparator(
            self.speed_bins.tolist(), self.counts[-n:].tolist(), labels=self.labels[-n:]
        )

    def distribution(self, i: int) -> np.ndarray:
        """
        Returns one row's speed counts normalized into a probability
        distribution over speed_bins.

        Args:
            i (int): Index of the histogram row.

        Returns:
            np.ndarray: Row i's counts divided by their total, summing to 1.

        Raises:
            ValueError: If the row sums to zero (no vehicles observed, so it
                can't be normalized into a distribution).
        """
        return self._as_distribution(self.counts[i])

    def distributions(self) -> np.ndarray:
        """
        Returns every row's speed counts normalized into probability
        distributions over speed_bins.

        Returns:
            np.ndarray: Array of shape (num_rows, num_bins), each row
                summing to 1, in the same order as counts/labels.

        Raises:
            ValueError: If any row sums to zero (no vehicles observed).
        """
        return np.array([self.distribution(i) for i in range(len(self.counts))])

    def variance(self, i: int) -> float:
        """
        Computes the variance of one row's speed distribution.

        Treats the row as a probability distribution over speed_bins (see
        distribution) and returns that distribution's variance - how spread
        out the vehicle speeds were that day, independent of how many
        vehicles were observed.

        Args:
            i (int): Index of the histogram row.

        Returns:
            float: Variance of the row's speed distribution, in squared
                units of speed_bins (e.g. mph^2 if speed_bins is in mph).

        Raises:
            ValueError: If the row sums to zero (no vehicles observed, so it
                can't be normalized into a distribution).
        """
        p = self.distribution(i)
        mean = float(np.sum(p * self.speed_bins))
        return float(np.sum(p * (self.speed_bins - mean) ** 2))

    def variances(self) -> List[float]:
        """
        Computes the variance of every row's speed distribution.

        Returns:
            List[float]: One variance value per row, in the same order as
                counts/labels.

        Raises:
            ValueError: If any row sums to zero (no vehicles observed).
        """
        return [self.variance(i) for i in range(len(self.counts))]

    def pearson_correlation(self, i: int, j: int) -> float:
        """
        Computes the Pearson correlation coefficient between two rows' counts.

        Args:
            i (int): Index of the first histogram row.
            j (int): Index of the second histogram row.

        Returns:
            float: Pearson correlation coefficient in [-1.0, 1.0]. 1.0 if
                both rows are constant (zero variance) and identical, 0.0 if
                either row is constant but they aren't.
        """
        a, b = self.counts[i], self.counts[j]
        if np.std(a) == 0 or np.std(b) == 0:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def jensen_shannon_divergence(self, i: int, j: int) -> float:
        """
        Computes the Jensen-Shannon divergence between two rows' counts.

        Each row is normalized into a probability distribution over speed
        bins before comparing, so this is invariant to how many vehicles
        were observed on each day.

        Args:
            i (int): Index of the first histogram row.
            j (int): Index of the second histogram row.

        Returns:
            float: Jensen-Shannon divergence in bits, bounded in [0.0, 1.0].
                0.0 means identical distributions; 1.0 means disjoint
                support (no bin has mass in both).

        Raises:
            ValueError: If either row sums to zero (no vehicles observed,
                so it can't be normalized into a distribution).
        """
        return _jensen_shannon_divergence(self.distribution(i), self.distribution(j))

    def overlap_coefficient(self, i: int, j: int) -> float:
        """
        Computes the overlap coefficient between two rows' counts.

        Each row is normalized into a probability distribution over speed
        bins, then the shared mass (the per-bin minimum, summed) is
        returned - equivalent to 1 minus the total variation distance
        between the two distributions.

        Args:
            i (int): Index of the first histogram row.
            j (int): Index of the second histogram row.

        Returns:
            float: Overlap coefficient in [0.0, 1.0]. 1.0 means identical
                distributions; 0.0 means disjoint support (no bin has mass
                in both).

        Raises:
            ValueError: If either row sums to zero (no vehicles observed,
                so it can't be normalized into a distribution).
        """
        return _overlap_coefficient(self.distribution(i), self.distribution(j))

    @staticmethod
    def _as_distribution(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        if total <= 0:
            raise ValueError("Cannot normalize a histogram row with zero total count.")
        return counts / total

    def compare(self, i: int, j: int) -> ComparisonResult:
        """
        Computes every similarity metric for one pair of histogram rows.

        Args:
            i (int): Index of the first histogram row.
            j (int): Index of the second histogram row.

        Returns:
            ComparisonResult: Pearson correlation, Jensen-Shannon divergence,
                and overlap coefficient for the (i, j) pair.
        """
        return ComparisonResult(
            label_a=self.labels[i],
            label_b=self.labels[j],
            pearson_correlation=self.pearson_correlation(i, j),
            jensen_shannon_divergence=self.jensen_shannon_divergence(i, j),
            overlap_coefficient=self.overlap_coefficient(i, j),
        )

    def compare_all(self) -> List[ComparisonResult]:
        """
        Computes every similarity metric for every pair of histogram rows.

        Returns:
            List[ComparisonResult]: One result per unordered pair (i, j)
                with i < j, in row order.
        """
        n = len(self.counts)
        return [self.compare(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_stable(
        self,
        js_threshold: float = DEFAULT_JS_THRESHOLD,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    ) -> bool:
        """
        Checks whether every pair of histograms is close enough to call the
        speed distribution constant across all sampled days.

        Args:
            js_threshold (float): Maximum acceptable pairwise
                Jensen-Shannon divergence. Defaults to DEFAULT_JS_THRESHOLD.
            overlap_threshold (float): Minimum acceptable pairwise overlap
                coefficient. Defaults to DEFAULT_OVERLAP_THRESHOLD.

        Returns:
            bool: True if every pair passes ComparisonResult.is_stable.
        """
        return all(
            result.is_stable(js_threshold, overlap_threshold)
            for result in self.compare_all()
        )


def load_directional_csv(path: "str | Path") -> Dict[str, SpeedDistributionComparator]:
    """
    Loads a CSV of per-day, per-direction vehicle speed histograms, splitting
    it into one SpeedDistributionComparator per direction.

    Expected shape:

        date,direction,10,15,20,25,30,35,40
        2026-01-01,left,3,12,48,61,20,4,1
        2026-01-01,right,1,9,40,58,22,5,0
        2026-01-02,left,2,15,45,58,24,3,0
        2026-01-02,right,4,11,42,55,20,4,0

    Every date appears once per direction, interleaved in any order - rows
    are grouped by their direction column (second column) rather than by
    position. Each resulting comparator keeps only that direction's rows, in
    the order they appeared in the file, so SpeedDistributionComparator.tail(n)
    on one of them means "the last n dates observed for this direction"
    rather than "the last n rows of the file" (which would mix both
    directions and only cover about n/2 dates per direction).

    Args:
        path (str | Path): Path to the CSV. Its first row is
            [<date column label>, <direction column label>, *speed_bins] -
            the first two header cells are themselves ignored (they're
            column labels, not data), and every subsequent row is
            [date, direction, *counts].

    Returns:
        Dict[str, SpeedDistributionComparator]: One comparator per distinct
            value found in the direction column (e.g. "left" and "right"),
            keyed by that value exactly as it appears in the CSV, each
            sharing the same speed_bins.

    Raises:
        ValueError: If the file has fewer than two rows (a header plus at
            least one data row), the header has fewer than 3 columns, a data
            row has fewer than 3 columns, or a bin value can't be parsed as
            a number.
    """
    with open(path, "r", newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    if len(rows) < 2:
        raise ValueError(
            f"CSV '{path}' must have a header row plus at least one data row."
        )

    header, *data_rows = rows
    if len(header) < 3:
        raise ValueError(
            f"CSV '{path}' header must have at least 3 columns: date, "
            "direction, and one or more speed bins."
        )

    try:
        speed_bins = [float(v) for v in header[2:]]
    except ValueError as e:
        raise ValueError(f"Could not parse speed bin values from header: {e}") from e

    dates_by_direction: Dict[str, List[str]] = defaultdict(list)
    counts_by_direction: Dict[str, List[List[float]]] = defaultdict(list)
    for row in data_rows:
        if len(row) < 3:
            raise ValueError(
                f"Row {row!r} must have at least 3 columns: date, direction, "
                "and counts."
            )
        date, direction, *values = row
        try:
            counts = [float(v) for v in values]
        except ValueError as e:
            raise ValueError(f"Could not parse row {row!r} as numbers: {e}") from e
        dates_by_direction[direction].append(date)
        counts_by_direction[direction].append(counts)

    return {
        direction: SpeedDistributionComparator(
            speed_bins, counts_by_direction[direction], labels=dates
        )
        for direction, dates in dates_by_direction.items()
    }


def _format_results_table(results: Sequence[ComparisonResult]) -> str:
    header = f"{'A':<20}{'B':<20}{'pearson':>10}{'js_div':>10}{'overlap':>10}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.label_a:<20}{r.label_b:<20}{r.pearson_correlation:>10.4f}"
            f"{r.jensen_shannon_divergence:>10.4f}{r.overlap_coefficient:>10.4f}"
        )
    return "\n".join(lines)


def _report_match_stats(
    comparator: SpeedDistributionComparator,
    reference_counts: Sequence[Sequence[float]],
    reference_labels: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float, float]]:
    """
    Computes each report's Jensen-Shannon divergence and overlap coefficient
    against comparator's mean distribution (the average of every row's own
    distribution) - the "how well does this report match the ground truth
    as a whole" comparison shared by _format_final_statistics and
    _format_intuitive_summary.

    Args:
        comparator (SpeedDistributionComparator): Comparator whose mean
            distribution every report is compared against.
        reference_counts (Sequence[Sequence[float]]): One count array per
            report (e.g. from load_report_counts), matched by position to
            reference_labels.
        reference_labels (Optional[Sequence[str]]): One label per report,
            matched by position to reference_counts. Defaults to "Report 1",
            "Report 2", etc. when not given.

    Returns:
        List[Tuple[str, float, float]]: One (label, jensen_shannon_divergence,
            overlap_coefficient) tuple per report, in reference_counts' order.

    Raises:
        ValueError: If any comparator row sums to zero (no vehicles
            observed, so it can't be normalized into a distribution).
    """
    labels = (
        list(reference_labels)
        if reference_labels is not None
        else [f"Report {i + 1}" for i in range(len(reference_counts))]
    )
    csv_mean_distribution = comparator.distributions().mean(axis=0)
    stats = []
    for label, counts in zip(labels, reference_counts):
        distribution = SpeedDistributionComparator._as_distribution(np.asarray(counts, dtype=float))
        js = _jensen_shannon_divergence(distribution, csv_mean_distribution)
        overlap = _overlap_coefficient(distribution, csv_mean_distribution)
        stats.append((label, js, overlap))
    return stats


def _format_final_statistics(
    comparator: SpeedDistributionComparator,
    reference_counts: Optional[Sequence[Sequence[float]]] = None,
    reference_labels: Optional[Sequence[str]] = None,
) -> str:
    """
    Formats a final summary: comparator's variance (mean and range across
    every row), plus - if any reports were loaded - how well each matches
    comparator's data as a whole (see _report_match_stats).

    Args:
        comparator (SpeedDistributionComparator): Comparator to summarize.
        reference_counts (Optional[Sequence[Sequence[float]]]): One count
            array per report (e.g. from load_report_counts), matched by
            position to reference_labels. Defaults to None (no report
            match lines).
        reference_labels (Optional[Sequence[str]]): One label per report,
            matched by position to reference_counts. Defaults to "Report 1",
            "Report 2", etc. when reference_counts is given without labels.

    Returns:
        str: Multi-line summary, one line for the CSV variance and one
            additional line per report.

    Raises:
        ValueError: If any comparator row sums to zero (no vehicles
            observed, so it can't be normalized into a distribution).
    """
    variances = comparator.variances()
    lines = [
        f"CSV variance — mean {np.mean(variances):.2f}, "
        f"range [{min(variances):.2f}, {max(variances):.2f}]"
    ]

    if reference_counts:
        for label, js, overlap in _report_match_stats(comparator, reference_counts, reference_labels):
            lines.append(f"Report match — {label}: JS div {js:.3f}, overlap {overlap:.3f}")

    return "\n".join(lines)


def _format_intuitive_summary(
    direction: str,
    results: Sequence[ComparisonResult],
    report_stats: Sequence[Tuple[str, float, float]],
    js_threshold: float,
    overlap_threshold: float,
    stable: bool,
) -> str:
    """
    Formats one direction's numbers under two clearly labeled headings, so
    the two distinct things this tool's Jensen-Shannon divergence measures
    aren't left to infer from unlabeled numbers alone:

    1. "Ground truth stability" - results (from compare_all()) compares
       every pair of sampled dates in the CSV against each other, i.e.
       whether the ground truth's own speed distribution held steady over
       time. is_stable and the printed verdict are based on this alone.
    2. "Report match" - report_stats (from _report_match_stats) compares
       each --report against the CSV's average distribution, i.e. whether
       that specific report looks like what the ground truth usually looks
       like.

    Args:
        direction (str): Direction this section covers (e.g. "left").
        results (Sequence[ComparisonResult]): Every pairwise ground truth
            comparison, as returned by
            SpeedDistributionComparator.compare_all().
        report_stats (Sequence[Tuple[str, float, float]]): One
            (label, jensen_shannon_divergence, overlap_coefficient) tuple
            per report - see _report_match_stats. Empty if no --report was
            given.
        js_threshold (float): The same Jensen-Shannon divergence cutoff
            is_stable uses, reused here to label each pair/report
            pass/fail consistently with the printed verdict.
        overlap_threshold (float): The same overlap cutoff is_stable uses.
        stable (bool): This direction's overall is_stable verdict (ground
            truth date-to-date consistency only - see above).

    Returns:
        str: A multi-line, labeled-numbers summary for this direction.
    """
    lines = [f"[{direction}]"]

    lines.append("  Ground truth stability (date-to-date):")
    if not results:
        lines.append("    Pairs compared: 0 (only one date sampled)")
    else:
        unstable_pairs = [r for r in results if not r.is_stable(js_threshold, overlap_threshold)]
        worst = max(results, key=lambda r: r.jensen_shannon_divergence)
        lines.append(
            f"    Pairs within threshold: {len(results) - len(unstable_pairs)}/{len(results)}"
        )
        lines.append(
            f"    Worst pair: {worst.label_a} vs {worst.label_b} "
            f"(JS divergence {worst.jensen_shannon_divergence:.3f}, "
            f"overlap {worst.overlap_coefficient:.3f})"
        )
    lines.append(f"    Verdict: {'STABLE' if stable else 'NOT STABLE'}")

    lines.append("  Report match (vs. ground truth average):")
    if not report_stats:
        lines.append("    No --report given.")
    else:
        for label, js, overlap in report_stats:
            match = "PASS" if js <= js_threshold and overlap >= overlap_threshold else "FAIL"
            lines.append(f"    {label}: JS divergence {js:.3f}, overlap {overlap:.3f} ({match})")

    return "\n".join(lines)


DEFAULT_RIDGELINE_MAX_ROWS = 40
_RIDGELINE_X_RESOLUTION = 240
_RIDGELINE_ROW_STEP = 1.0
_RIDGELINE_OVERLAP = 3.0


def load_report_counts(
    report_path: Union[str, Path],
    speed_bins: Sequence[float],
    entity_type: Optional[int] = None,
    direction: Optional[str] = None,
) -> np.ndarray:
    """
    Builds raw per-bin vehicle counts from a report_recalibrator.py (or
    video_entityprofiler.py --report) JSON report, so a single report can be
    compared against a SpeedDistributionComparator's historical rows on the
    same ridgeline plot.

    Unlike the comparator's CSV rows, which are already pre-binned counts, a
    report's "individual_entities" list carries one continuous speed value
    per tracked vehicle. If absolute_speed is populated (the report went
    through calibrate_absolute_speeds against a known reference vehicle),
    those values are already in the same real-world unit as speed_bins and
    are used directly. If no entity has an absolute_speed - the report is
    from before calibration - relative_speed is used instead: a unit-less
    0-1 pixel-displacement ratio with no fixed relationship to speed_bins on
    its own (see video_entityprofiler.compute_relative_speeds), so it's
    linearly rescaled (its own min/max mapped onto speed_bins' min/max)
    before binning. That trades away an absolute-value comparison for a
    shape-only one - an uncalibrated report can't say "this vehicle was
    doing 30mph", but rescaled onto the same axis it can still say "this
    report's *shape* looks like the fast end (or slow end) of the historical
    distribution."

    Either way, each resulting value is then trimmed to speed_bins' own
    range - any value below min(speed_bins) or above max(speed_bins) is
    dropped rather than piled into the nearest edge bin, since a value
    outside the CSV's own boxes isn't really "closest" to the edge box, it's
    just outside what the CSV measured at all. What's left is assigned to
    whichever value in speed_bins is numerically closest to it (nearest-bin,
    not a strict [lo, hi) range) - speed_bins are plain box labels (e.g. 20,
    25, 30, ...), not bin edges, so "closest label" is the only assignment
    rule that doesn't assume anything about their spacing.

    Args:
        report_path (Union[str, Path]): Path to the JSON report.
        speed_bins (Sequence[float]): The speed box values (mph) to bin
            against - pass a SpeedDistributionComparator's speed_bins so the
            result lines up with its rows.
        entity_type (Optional[int]): If given, only entities whose
            "entity_type" (a COCO class id - see src/detection/classes.py's
            CLASS_ID_MAPPING) equals this value are counted. Defaults to
            None (every entity).
        direction (Optional[str]): If given, only entities whose "direction"
            (video_entityprofiler.py's "left"/"right" travel direction,
            always written lowercase) equals this value are counted, matched
            case-insensitively so a CSV direction column typed in any case
            still lines up - pass the same direction as the
            SpeedDistributionComparator being compared against, since a
            report's left-traveling entities have nothing to say about a
            right-traveling historical distribution. Defaults to None
            (every entity, regardless of direction).

    Returns:
        np.ndarray: Raw counts, one per speed_bins value, of how many
            entities' (rescaled, if uncalibrated) speed fell nearest to that
            bin, after trimming anything outside speed_bins' range.

    Raises:
        ValueError: If the report has no "individual_entities", none (after
            any entity_type/direction filtering) have a non-null
            absolute_speed or relative_speed, or none fall within
            speed_bins' range.
    """
    with open(report_path, "r") as f:
        report = json.load(f)

    entities = report.get("individual_entities")
    if not entities:
        raise ValueError(f"Report '{report_path}' has no 'individual_entities'.")

    if entity_type is not None:
        entities = [e for e in entities if e.get("entity_type") == entity_type]
    if direction is not None:
        # video_entityprofiler.py always writes "left"/"right" in lowercase,
        # but the CSV's own direction column (whatever case it was typed in)
        # is what callers pass through here - normalize so a CSV direction
        # of e.g. "Left" still matches the report's lowercase "left".
        direction_lower = direction.lower()
        entities = [e for e in entities if str(e.get("direction")).lower() == direction_lower]

    bins_array = np.asarray(speed_bins, dtype=float)
    bins_min, bins_max = bins_array.min(), bins_array.max()
    filter_notes = []
    if entity_type is not None:
        filter_notes.append(f"entity_type {entity_type}")
    if direction is not None:
        filter_notes.append(f"direction '{direction}'")
    filter_note = f" of {' and '.join(filter_notes)}" if filter_notes else ""

    absolute_speeds = [
        e["absolute_speed"] for e in entities if e.get("absolute_speed") is not None
    ]
    if absolute_speeds:
        speeds = np.asarray(absolute_speeds, dtype=float)
    else:
        relative_speeds = [
            e["relative_speed"] for e in entities if e.get("relative_speed") is not None
        ]
        if not relative_speeds:
            raise ValueError(
                f"Report '{report_path}' has no entities with an absolute_speed "
                f"or relative_speed{filter_note}."
            )
        logger.debug(
            "Report '%s' has no calibrated absolute_speed%s - falling back to "
            "relative_speed, rescaled onto speed_bins' range.",
            report_path,
            filter_note,
        )
        speeds = _minmax_scale(np.asarray(relative_speeds, dtype=float), bins_min, bins_max)

    in_range = (speeds >= bins_min) & (speeds <= bins_max)
    trimmed = int((~in_range).sum())
    if trimmed:
        logger.debug(
            "Report '%s' had %d entit%s outside speed_bins range [%g, %g] - trimmed.",
            report_path,
            trimmed,
            "y" if trimmed == 1 else "ies",
            bins_min,
            bins_max,
        )
    speeds = speeds[in_range]
    if len(speeds) == 0:
        raise ValueError(
            f"Report '{report_path}' has no entities{filter_note} within "
            f"speed_bins' range [{bins_min}, {bins_max}]."
        )

    counts = np.zeros(len(bins_array), dtype=float)
    for speed in speeds:
        nearest_bin = int(np.abs(bins_array - speed).argmin())
        counts[nearest_bin] += 1
    return counts


def _minmax_scale(values: np.ndarray, target_min: float, target_max: float) -> np.ndarray:
    source_min, source_max = values.min(), values.max()
    if source_max == source_min:
        return np.full_like(values, (target_min + target_max) / 2)
    return target_min + (values - source_min) * (target_max - target_min) / (
        source_max - source_min
    )


def plot_ridgeline(
    comparator: SpeedDistributionComparator,
    output_path: Union[str, Path],
    max_rows: int = DEFAULT_RIDGELINE_MAX_ROWS,
    reference_counts: Optional[Sequence[Sequence[float]]] = None,
    reference_labels: Optional[Sequence[str]] = None,
    highlight_last_n: Optional[int] = None,
) -> None:
    """
    Saves a ridgeline plot (joyplot) of every row's speed distribution to a
    PNG - the standard chart for "how does a distribution change across many
    time-ordered samples". Each row's normalized distribution is its own
    filled curve, stacked with a slight vertical overlap, earliest row at
    the top and latest at the bottom, colored on a light-to-dark ramp by
    time so the stack reads earliest-to-latest by color as well as
    position. A shifting peak, a widening spread, or a second hump appearing
    partway down the stack is a drift you see directly in the shape.

    Args:
        comparator (SpeedDistributionComparator): Comparator whose rows to
            plot - pass the full, untruncated comparator (not a
            comparator.tail(n) subset) so highlight_last_n has the whole
            history to draw against.
        output_path (Union[str, Path]): Filepath to save the chart PNG to.
        max_rows (int): Maximum number of ridges to draw. If comparator has
            more rows than this, an evenly spaced subset (always including
            the first and last row) is drawn instead of every row, so the
            chart stays legible instead of drowning in hundreds of
            overlapping ridges. Defaults to DEFAULT_RIDGELINE_MAX_ROWS.
        reference_counts (Optional[Sequence[Sequence[float]]]): If given,
            one or more extra distributions (e.g. from load_report_counts)
            drawn as their own ridges below the historical stack, each in
            its own fixed color off RIDGELINE_REPORT_PALETTE instead of a
            slot on the time ramp - since none of them are historical rows
            being compared for drift, they're things being checked against
            them. Each must have one value per comparator.speed_bins.
            Defaults to None (no reference ridges).
        reference_labels (Optional[Sequence[str]]): Y-axis label and legend
            entry for each reference ridge, matched by position to
            reference_counts. Defaults to "Report 1", "Report 2", etc. when
            reference_counts is given without labels.
        highlight_last_n (Optional[int]): If given (e.g. from --last-n),
            draws a black box around the most recent highlight_last_n rows'
            ridges - the subset of comparator's full history that's actually
            being tested - so it reads at a glance against the rest of the
            historical stack instead of only ever being plotted in
            isolation. Also noted in the title and given its own legend
            entry ("Last N Dates"), which - together with the reference
            ridges' colors, if any - forces the legend to appear even when
            no --report was given. Clamped to comparator's row count.
            Defaults to None (no box, no legend, no title note).

    Raises:
        ValueError: If any drawn row sums to zero (no vehicles observed, so
            it can't be normalized into a distribution), any reference_counts
            entry's length doesn't match comparator.speed_bins, or
            reference_labels is given with a different length than
            reference_counts.
    """
    n = len(comparator.counts)
    if n > max_rows:
        indices = sorted(set(np.linspace(0, n - 1, max_rows).round().astype(int).tolist()))
    else:
        indices = list(range(n))

    bins = comparator.speed_bins
    x_fine = np.linspace(bins.min(), bins.max(), _RIDGELINE_X_RESOLUTION)
    curves = [np.interp(x_fine, bins, comparator.distribution(i)) for i in indices]
    peak = max(curve.max() for curve in curves)

    reference_curves: List[np.ndarray] = []
    if reference_counts:
        for i, counts in enumerate(reference_counts):
            counts_array = np.asarray(counts, dtype=float)
            if len(counts_array) != len(bins):
                raise ValueError(
                    f"reference_counts[{i}] has {len(counts_array)} values, "
                    f"expected {len(bins)} to match comparator.speed_bins."
                )
            distribution = SpeedDistributionComparator._as_distribution(counts_array)
            curve = np.interp(x_fine, bins, distribution)
            reference_curves.append(curve)
            peak = max(peak, curve.max())

        if reference_labels is not None:
            if len(reference_labels) != len(reference_curves):
                raise ValueError(
                    f"reference_labels has {len(reference_labels)} entries, "
                    f"expected {len(reference_curves)} to match reference_counts."
                )
            reference_labels = list(reference_labels)
        else:
            reference_labels = [f"Report {i + 1}" for i in range(len(reference_curves))]

    scale = (_RIDGELINE_ROW_STEP * _RIDGELINE_OVERLAP) / peak

    num_ridges = len(indices)
    num_references = len(reference_curves)
    fig_height = max(4.0, 0.18 * (num_ridges + num_references) + 2.0)
    fig, ax = plt.subplots(figsize=(9, fig_height), facecolor=RIDGELINE_SURFACE_COLOR)
    ax.set_facecolor(RIDGELINE_SURFACE_COLOR)

    # Reserve one row_step per reference ridge at the very bottom by
    # shifting every historical baseline up by that much.
    baseline_shift = num_references * _RIDGELINE_ROW_STEP

    # Baselines run high-to-low as rank increases (earliest row highest, so
    # it reads top-to-bottom as chronological), and are drawn in that same
    # order so each later, lower ridge is drawn on top of and partially
    # occludes the one behind it - the deliberate "mountain range" layering
    # a ridgeline plot is named for.
    baselines = [
        (num_ridges - 1 - rank) * _RIDGELINE_ROW_STEP + baseline_shift
        for rank in range(num_ridges)
    ]
    for rank, i in enumerate(indices):
        baseline = baselines[rank]
        y = baseline + curves[rank] * scale
        color = _RIDGELINE_CMAP(i / max(n - 1, 1))
        ax.fill_between(x_fine, baseline, y, color=color, alpha=0.92, linewidth=0, zorder=rank)
        ax.plot(
            x_fine,
            y,
            color=RIDGELINE_SURFACE_COLOR,
            linewidth=1.2,
            zorder=rank + 0.5,
        )

    tick_positions = list(baselines)
    tick_labels = [comparator.labels[i] for i in indices]
    label_step = max(1, num_ridges // 25)
    # Rows inside the highlighted (--last-n) subset always keep their date
    # label even when thinned out below - the whole point of the box is to
    # show exactly which dates it covers.
    highlighted: set = set()
    if highlight_last_n is not None:
        highlight_last_n = min(highlight_last_n, n)
        highlighted = {i for i in indices if i >= n - highlight_last_n}
    thinned_positions, thinned_labels = [], []
    for rank, (position, label) in enumerate(zip(tick_positions, tick_labels)):
        if rank % label_step == 0 or indices[rank] in highlighted:
            thinned_positions.append(position)
            thinned_labels.append(label)
    tick_positions, tick_labels = thinned_positions, thinned_labels

    if reference_curves:
        # Reference index 0 sits just below the historical stack; the last
        # one sits at the very bottom (baseline 0), drawn last so it's in
        # front of everything else - consistent with "later/more-in-front"
        # already meaning "lower and higher zorder" for the historical
        # ridges above. Each gets its own fixed palette color instead of a
        # spot on the time ramp, since none of them are one more historical
        # sample - they're things being checked against the stack.
        for i, curve in enumerate(reference_curves):
            baseline = (num_references - 1 - i) * _RIDGELINE_ROW_STEP
            y = baseline + curve * scale
            color = RIDGELINE_REPORT_PALETTE[i % len(RIDGELINE_REPORT_PALETTE)]
            label = reference_labels[i]
            ax.fill_between(
                x_fine,
                baseline,
                y,
                color=color,
                alpha=0.55,
                linewidth=0,
                zorder=num_ridges + i,
                label=label,
            )
            ax.plot(x_fine, y, color=color, linewidth=1.8, zorder=num_ridges + i + 0.5)
            tick_positions.append(baseline)
            tick_labels.append(label)

    if highlighted:
        # highlighted is filtered from indices, so every member has a rank -
        # this is never empty here.
        drawn_ranks = [rank for rank, i in enumerate(indices) if i in highlighted]
        top_baseline = baselines[min(drawn_ranks)]
        bottom_baseline = baselines[max(drawn_ranks)]
        box_bottom = bottom_baseline - _RIDGELINE_ROW_STEP * 0.5
        box_top = top_baseline + _RIDGELINE_ROW_STEP * (_RIDGELINE_OVERLAP + 0.3)
        ax.add_patch(
            PlotRectangle(
                (x_fine[0], box_bottom),
                x_fine[-1] - x_fine[0],
                box_top - box_bottom,
                fill=False,
                edgecolor="black",
                linewidth=2,
                zorder=num_ridges + num_references + 10,
                clip_on=False,
                label=f"Last {highlight_last_n} Dates",
            )
        )

    if reference_curves or highlighted:
        # The historical ridges themselves aren't individually labeled (they're
        # colored on the light-to-dark time ramp, not one fixed color, and
        # labeling every single one would be unreadable) - add one proxy patch
        # in the ramp's primary blue so the legend still identifies "the
        # historical stack" as a group, distinct from each report's color and
        # from the --last-n highlight box.
        historical_patch = PlotRectangle(
            (0, 0), 0, 0, facecolor=RIDGELINE_LINE_COLOR, edgecolor="none", label="Car Counter"
        )
        ax.add_patch(historical_patch)
        ax.legend(loc="upper right", frameon=False, labelcolor=RIDGELINE_MUTED_TEXT_COLOR)

    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    subset_note = "" if num_ridges == n else f" of {n}"
    title = f"Speed Distribution Ridgeline (n={num_ridges}{subset_note})"
    if highlighted:
        title += f" — Last {highlight_last_n} Dates"
    ax.set_title(
        title,
        color=RIDGELINE_PRIMARY_TEXT_COLOR,
        fontsize=14,
    )
    ax.set_xlabel("Speed", color=RIDGELINE_MUTED_TEXT_COLOR)
    ax.set_ylabel("Date", color=RIDGELINE_MUTED_TEXT_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(RIDGELINE_AXIS_COLOR)
    ax.tick_params(colors=RIDGELINE_MUTED_TEXT_COLOR, length=0)
    ax.set_xlim(x_fine[0], x_fine[-1])
    ax.set_ylim(bottom=-_RIDGELINE_ROW_STEP * 0.5)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=RIDGELINE_SURFACE_COLOR)
    plt.close(fig)


def _ridgeline_plot_path(args: argparse.Namespace, direction: str) -> Path:
    """
    Resolves the output path for one direction's ridgeline plot, inserting
    the direction before the file extension so multiple directions never
    collide on the same file.

    Args:
        args (argparse.Namespace): Parsed CLI args - uses ridgeline_plot if
            given, else csv_path to build the default name.
        direction (str): Direction this plot is for (e.g. "left").

    Returns:
        Path: '<ridgeline_plot stem>_<direction><ridgeline_plot suffix>' if
            --ridgeline-plot was given, else
            '<csv_path stem>_<direction>_ridgeline.png'.
    """
    if args.ridgeline_plot:
        base = Path(args.ridgeline_plot)
        return base.with_name(f"{base.stem}_{direction}{base.suffix}")
    return Path(args.csv_path).with_name(f"{Path(args.csv_path).stem}_{direction}_ridgeline.png")


def main() -> None:
    """
    Main CLI entry point for the speed distribution comparator.
    """
    parser = argparse.ArgumentParser(
        description="Compare per-day vehicle speed histograms (Jensen-Shannon "
        "divergence and overlap coefficient, with Pearson correlation "
        "reported for reference) to test whether the speed distribution is "
        "constant over time."
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to a CSV whose header row is [date, direction, *speed "
        "bins (mph)] and whose subsequent rows are [date, direction, "
        "*counts] - one row per date per direction (e.g. 'left'/'right'). "
        "Every direction found is compared and plotted entirely separately.",
    )
    parser.add_argument(
        "--js-threshold",
        type=float,
        default=DEFAULT_JS_THRESHOLD,
        help=f"Maximum acceptable pairwise Jensen-Shannon divergence. "
        f"Defaults to {DEFAULT_JS_THRESHOLD}.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help=f"Minimum acceptable pairwise overlap coefficient. Defaults to "
        f"{DEFAULT_OVERLAP_THRESHOLD}.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Only compare each direction's last N dates, instead of its "
        "full history - applied independently per direction, so this is "
        "the last N dates observed for that direction specifically, not the "
        "last N rows of the CSV (which would mix both directions). Useful "
        "for checking whether the distribution has settled into a stable "
        "regime even if the full history hasn't. Each direction's ridgeline "
        "plot still draws its full history - this subset is outlined with a "
        "black box instead of hiding the rest. Defaults to None (use every "
        "date).",
    )
    parser.add_argument(
        "--ridgeline-plot",
        type=str,
        default=None,
        help="Path to save each direction's ridgeline plot to - the "
        "direction is inserted before the file extension (e.g. "
        "'out_left.png', 'out_right.png' for 'out.png'). Defaults to "
        "'<csv_path>_<direction>_ridgeline.png'.",
    )
    parser.add_argument(
        "--no-ridgeline-plot",
        action="store_true",
        help="Skip generating the ridgeline plot.",
    )
    parser.add_argument(
        "--ridgeline-max-rows",
        type=int,
        default=DEFAULT_RIDGELINE_MAX_ROWS,
        help="Maximum number of ridges to draw before subsampling evenly "
        f"across rows. Defaults to {DEFAULT_RIDGELINE_MAX_ROWS}.",
    )
    parser.add_argument(
        "--report",
        type=str,
        action="append",
        default=None,
        help="Path to a report_recalibrator.py (or video_entityprofiler.py "
        "--report) JSON report to bin against the CSV's speed_bins and draw "
        "as its own ridge, in a distinct color, below the historical stack. "
        "Repeat to plot multiple reports, each in its own color.",
    )
    parser.add_argument(
        "--report-entity-type",
        type=int,
        default=None,
        help="If given (see src/detection/classes.py's CLASS_ID_MAPPING), "
        "only entities of this COCO class id are counted from every "
        "--report. Defaults to None (every entity).",
    )
    parser.add_argument(
        "--report-label",
        type=str,
        action="append",
        default=None,
        help="Y-axis label and legend entry for the corresponding --report, "
        "matched by position (the first --report-label goes with the first "
        "--report, and so on). Defaults to that report's file name for any "
        "--report given without one.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    try:
        comparators = load_directional_csv(args.csv_path)
    except ValueError as e:
        logger.error("Could not load '%s': %s", args.csv_path, e)
        sys.exit(1)

    if args.last_n is not None and args.last_n < 2:
        logger.error("--last-n must be at least 2 to compare rows against each other.")
        sys.exit(1)

    overall_stable = True
    summary_sections = []
    for direction in sorted(comparators):
        comparator = comparators[direction]
        print(f"\n=== Direction: {direction} ===")

        # analysis_comparator is what's actually compared/tested against
        # --last-n; comparator itself is kept at full history so the
        # ridgeline plot can still show every row, with a box around this
        # subset.
        analysis_comparator = comparator.tail(args.last_n) if args.last_n else comparator

        results = analysis_comparator.compare_all()
        print(_format_results_table(results))

        reference_counts = None
        reference_labels = None
        if args.report:
            report_labels = args.report_label or []
            reference_counts = []
            reference_labels = []
            for i, report_path in enumerate(args.report):
                try:
                    reference_counts.append(
                        load_report_counts(
                            report_path,
                            analysis_comparator.speed_bins,
                            entity_type=args.report_entity_type,
                            direction=direction,
                        )
                    )
                except ValueError as e:
                    logger.error(
                        "Could not load report '%s' for direction '%s': %s",
                        report_path,
                        direction,
                        e,
                    )
                    sys.exit(1)
                label = report_labels[i] if i < len(report_labels) else Path(report_path).stem
                reference_labels.append(label)

        try:
            print(
                _format_final_statistics(analysis_comparator, reference_counts, reference_labels)
            )
        except ValueError as e:
            logger.error(
                "Could not compute final statistics for direction '%s': %s", direction, e
            )

        if not args.no_ridgeline_plot:
            ridgeline_plot_path = _ridgeline_plot_path(args, direction)
            try:
                plot_ridgeline(
                    comparator,
                    ridgeline_plot_path,
                    max_rows=args.ridgeline_max_rows,
                    reference_counts=reference_counts,
                    reference_labels=reference_labels,
                    highlight_last_n=args.last_n,
                )
                logger.info(
                    "Saved '%s' ridgeline plot to %s.", direction, ridgeline_plot_path
                )
            except ValueError as e:
                logger.error(
                    "Could not generate ridgeline plot for direction '%s': %s", direction, e
                )

        stable = analysis_comparator.is_stable(args.js_threshold, args.overlap_threshold)
        verdict = "STABLE" if stable else "NOT STABLE"
        logger.info(
            "Verdict (%s): %s (js_divergence <= %.4f and overlap >= %.4f required for "
            "every pair)",
            direction,
            verdict,
            args.js_threshold,
            args.overlap_threshold,
        )
        overall_stable = overall_stable and stable

        report_stats = (
            _report_match_stats(analysis_comparator, reference_counts, reference_labels)
            if reference_counts
            else []
        )
        summary_sections.append(
            _format_intuitive_summary(
                direction,
                results,
                report_stats,
                args.js_threshold,
                args.overlap_threshold,
                stable,
            )
        )

    print("\n" + "=" * 72)
    print("Final Report")
    print("=" * 72)
    print("\n\n".join(summary_sections))

    if not overall_stable:
        sys.exit(1)


if __name__ == "__main__":
    main()
