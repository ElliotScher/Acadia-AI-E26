"""
Direction Benchmark Common

Shared, subject-agnostic scoring and reporting infrastructure used by both
pedestrian_direction_benchmark.py and vehicle_direction_benchmark.py. None of
the ground-truth-from-filename convention, the confusion matrix/precision/
recall math, or the bar chart/HTML report rendering has anything to do with
which detector produced the predictions, so it lives here once instead of
being duplicated per subject type. What IS specific to each subject - which
labels count as valid ground truth, how to run its detector and pick a
primary result, its own banner/heading title - stays in that subject's own
benchmark module.
"""

import base64
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib

# Headless/non-interactive backend - this module generates chart images to a
# file or embeds them in HTML, it never shows a window, and the default
# backend would otherwise try (and fail) to open a display in a server/CI
# environment. Must be set before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utility.htmlreport import html_heading, html_table, wrap_html_document

IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

# A prediction outcome, not a ground-truth label - the detector found no
# subject to classify at all, which is a meaningfully different failure mode
# than confidently predicting "unknown".
NO_DETECTION = "no_detection"

# Splits a filename stem into delimiter-separated tokens (on any run of
# non-alphanumeric characters: underscore, dash, space, dot, ...) so a
# direction label can be matched as a whole token anywhere in the filename
# without a raw substring search spuriously matching inside an unrelated
# word - e.g. "leftover_notes.jpg" or "background_check.jpg" should not be
# mistaken for a "left"/"back" label.
_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Ground truth / predicted are always assigned these exact colors, everywhere
# they appear together across every direction benchmark - the same two
# series identity reused everywhere, not cycled. Matches the fixed
# ground-truth-vs-pipeline-output color pair already established by
# plate_dwell_benchmark.py's histogram.
GT_COLOR = "#2a78d6"  # categorical slot 1 (blue)
PRED_COLOR = "#e34948"  # categorical slot 6 (red)

BANNER_WIDTH = 46
LABEL_WIDTH = 26


def stat_line(label: str, value: str) -> str:
    return f"{label:<{LABEL_WIDTH}}: {value}"


def box_area(box: Tuple[int, int, int, int]) -> int:
    """
    Computes the pixel area of an (x1, y1, x2, y2) box - used to pick the
    largest/primary detection when more than one subject appears in a
    labeled benchmark image.
    """
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def label_from_filename(
    path: Union[str, Path], valid_labels: Tuple[str, ...]
) -> Optional[str]:
    """
    Recovers an image's ground-truth direction from a direction-label token
    appearing anywhere in its filename (prefix, middle, or suffix) - e.g.
    "car_017_left.jpg", "left_car_017.jpg", and "cam2_left_017.jpg" are all
    recognized as "left".

    Matching is done on whole delimiter-separated tokens (see
    _TOKEN_SPLIT_RE), not a raw substring search, so a word like "leftover"
    or "background" doesn't spuriously match "left"/"back".

    Args:
        path (Union[str, Path]): Image path or filename.
        valid_labels (Tuple[str, ...]): The direction labels this subject
            type's ground truth may encode (e.g. left/right/front/back/
            unknown for pedestrians, left/right only for vehicles).

    Returns:
        Optional[str]: The recognized direction label if exactly one appears
            as a token in the filename. None if none do, or if more than one
            distinct label does (ambiguous - e.g. a filename with both
            "left" and "right" tokens) - either way, there's no single
            known-correct label to score a prediction against, so such files
            are excluded from the benchmark entirely.
    """
    stem = Path(path).stem.lower()
    tokens = set(_TOKEN_SPLIT_RE.split(stem))
    found = [label for label in valid_labels if label in tokens]
    return found[0] if len(found) == 1 else None


def find_labeled_images(
    input_dir: Path, valid_labels: Tuple[str, ...]
) -> Dict[Path, str]:
    """
    Recursively finds every image under input_dir whose filename encodes a
    recognized ground-truth direction.

    Args:
        input_dir (Path): Directory to search (recursively).
        valid_labels (Tuple[str, ...]): Passed through to label_from_filename.

    Returns:
        Dict[Path, str]: Image path -> ground-truth direction label, for
            every image file with an unambiguous, recognized direction token
            in its filename (see label_from_filename).
    """
    labeled: Dict[Path, str] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = label_from_filename(path, valid_labels)
        if label is not None:
            labeled[path] = label
    return labeled


def build_confusion_matrix(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """
    Tallies a ground-truth-label -> predicted-label -> count confusion matrix.

    Args:
        results (List[Dict[str, Any]]): Per-image results, each with
            "ground_truth" and "predicted" keys.

    Returns:
        Dict[str, Dict[str, int]]: Confusion matrix, keyed first by
            ground-truth label (only labels actually present in the data),
            then by predicted label (including NO_DETECTION).
    """
    matrix: Dict[str, Dict[str, int]] = {}
    for result in results:
        gt = result["ground_truth"]
        pred = result["predicted"]
        row = matrix.setdefault(gt, {})
        row[pred] = row.get(pred, 0) + 1
    return matrix


def compute_label_metrics(
    confusion_matrix: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, Union[int, float]]]:
    """
    Computes per-label precision, recall, and F1 from a confusion matrix.

    Args:
        confusion_matrix (Dict[str, Dict[str, int]]): From build_confusion_matrix.

    Returns:
        Dict[str, Dict[str, Union[int, float]]]: Per ground-truth label:
            support (row total), true positives, precision, recall, and F1.
            Labels that only ever appear as a prediction (never as ground
            truth) - which can only be NO_DETECTION - are not included, since
            there is no support to compute recall against.
    """
    column_totals: Dict[str, int] = {}
    for row in confusion_matrix.values():
        for pred_label, count in row.items():
            column_totals[pred_label] = column_totals.get(pred_label, 0) + count

    metrics: Dict[str, Dict[str, Union[int, float]]] = {}
    for gt_label, row in confusion_matrix.items():
        support = sum(row.values())
        true_positives = row.get(gt_label, 0)
        predicted_as_label = column_totals.get(gt_label, 0)

        precision = true_positives / predicted_as_label if predicted_as_label else 0.0
        recall = true_positives / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        metrics[gt_label] = {
            "support": support,
            "true_positives": true_positives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return metrics


def compute_label_counts(
    results: List[Dict[str, Any]], direction_order: Tuple[str, ...]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Tallies how many images fall into each direction bucket, once for the
    ground-truth labels and once for the predicted labels.

    Args:
        results (List[Dict[str, Any]]): Per-image results, each with
            "ground_truth" and "predicted" keys.
        direction_order (Tuple[str, ...]): Every bucket that should appear
            (even with a 0 count), in display order.

    Returns:
        Tuple[Dict[str, int], Dict[str, int]]: (ground-truth counts,
            predicted counts), each keyed by every bucket in direction_order
            (0 if unseen) so the two dicts always share the same keys and can
            be plotted side by side.
    """
    gt_counts: Dict[str, int] = {label: 0 for label in direction_order}
    pred_counts: Dict[str, int] = {label: 0 for label in direction_order}
    for result in results:
        gt_counts[result["ground_truth"]] += 1
        pred_counts[result["predicted"]] += 1
    return gt_counts, pred_counts


def render_direction_bar_chart(
    label_metrics: Dict[str, Dict[str, Union[int, float]]],
    direction_order: Tuple[str, ...],
    overall_total: int,
    overall_correct: int,
    title: str = "Ground Truth vs Correct Detections",
) -> bytes:
    """
    Renders a bar chart comparing each ground-truth direction's total count
    against how many of those were correctly detected, plus a trailing
    "Overall" group summarizing every evaluated image - each bar pair is
    annotated with its accuracy percentage (correct / ground truth) directly
    above it, so the per-direction and overall numbers are readable off the
    chart without cross-referencing the metrics table.

    This deliberately shows only the true-positive count per ground-truth
    label (see compute_label_metrics), not the raw count of every prediction
    that happened to land in that bucket - the latter would also include
    incorrect predictions borrowed from other ground-truth labels (false
    positives), which isn't the comparison this chart is for. Prediction-only
    outcomes (NO_DETECTION, and "unknown" where that isn't valid ground
    truth) are dropped entirely - the x-axis is exactly the labels being
    scored, plus "Overall".

    Args:
        label_metrics (Dict[str, Dict[str, Union[int, float]]]): From
            compute_label_metrics.
        direction_order (Tuple[str, ...]): Ground-truth labels to include, in
            display order. Labels with 0 support are skipped.
        overall_total (int): Total images evaluated, for the "Overall" group.
        overall_correct (int): Total exact matches, for the "Overall" group.
        title (str): Chart title. Defaults to
            "Ground Truth vs Correct Detections".

    Returns:
        bytes: PNG image data. Empty if there's nothing to plot.
    """
    entries = [
        (label, label_metrics[label]["support"], label_metrics[label]["true_positives"])
        for label in direction_order
        if label in label_metrics and label_metrics[label]["support"] > 0
    ]
    if overall_total:
        entries.append(("Overall", overall_total, overall_correct))

    if not entries:
        return b""

    labels = [entry[0] for entry in entries]
    gt_values = [entry[1] for entry in entries]
    correct_values = [entry[2] for entry in entries]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    if labels[-1] == "Overall" and len(labels) > 1:
        # Recessive separator marking "Overall" as an aggregate, not a peer
        # ground-truth category - same visual language as the grid, not a
        # third color, since it's still the same two series (ground truth /
        # correct), just summed across all of them.
        ax.axvline(x[-1] - 0.5, color="#c3c2b7", linewidth=1, linestyle="--", zorder=1)

    ax.bar(
        x - width / 2,
        gt_values,
        width,
        color=GT_COLOR,
        alpha=0.85,
        label="Ground truth",
        edgecolor=GT_COLOR,
        linewidth=0.8,
    )
    ax.bar(
        x + width / 2,
        correct_values,
        width,
        color=PRED_COLOR,
        alpha=0.85,
        label="Correct detections",
        edgecolor=PRED_COLOR,
        linewidth=0.8,
    )

    y_max = max(gt_values + correct_values)
    pad = y_max * 0.03 if y_max else 1
    for i, (_, gt, correct) in enumerate(entries):
        pct = correct / gt * 100 if gt else 0.0
        ax.text(
            x[i],
            max(gt, correct) + pad,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#0b0b0b",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ").title() for label in labels])
    ax.set_ylabel("Number of images", color="#52514e")
    ax.set_title(
        title,
        color="#0b0b0b",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(top=y_max + pad * 5)
    ax.tick_params(colors="#52514e")
    ax.grid(True, which="major", axis="y", color="#e3e2dd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    # Legend sits outside the plot area, not "upper right" inside it - the
    # percentage label above the tallest bar (often the trailing "Overall"
    # group) would otherwise collide with an in-plot legend regardless of
    # which corner it's pinned to.
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        labelcolor="#0b0b0b",
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def print_direction_benchmark_summary(
    title: str,
    total_images_scanned: int,
    skipped_unlabeled: int,
    total: int,
    exact_matches: int,
    accuracy: float,
    confusion_matrix: Dict[str, Dict[str, int]],
    label_metrics: Dict[str, Dict[str, Union[int, float]]],
    prediction_labels: List[str],
) -> None:
    print()
    print("=" * BANNER_WIDTH)
    print(title.center(BANNER_WIDTH))
    print("=" * BANNER_WIDTH)
    print(stat_line("Images scanned", str(total_images_scanned)))
    print(stat_line("Skipped (no direction token)", str(skipped_unlabeled)))
    print(stat_line("Images evaluated", str(total)))
    print("-" * BANNER_WIDTH)
    print(
        stat_line("Exact direction match", f"{accuracy * 100:.2f}% ({exact_matches}/{total})")
    )
    print("=" * BANNER_WIDTH)

    if label_metrics:
        print()
        print("PER-DIRECTION METRICS".center(BANNER_WIDTH, "-"))
        label_w = max(len("Direction"), max(len(l) for l in label_metrics))
        header = (
            f"{'Direction':<{label_w}}  {'Support':>7}  "
            f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}"
        )
        print(header)
        print("-" * len(header))
        for gt_label in sorted(label_metrics):
            m = label_metrics[gt_label]
            print(
                f"{gt_label:<{label_w}}  {m['support']:>7}  "
                f"{m['precision'] * 100:>8.2f}%  {m['recall'] * 100:>8.2f}%  "
                f"{m['f1'] * 100:>8.2f}%"
            )

    if confusion_matrix:
        print()
        print(
            "CONFUSION MATRIX (rows=truth, cols=predicted)".center(BANNER_WIDTH, "-")
        )
        gt_labels = sorted(confusion_matrix)
        col_w = max(len("truth"), max(len(l) for l in prediction_labels))
        header = f"{'':<{col_w}}  " + "  ".join(
            f"{l:>{col_w}}" for l in prediction_labels
        )
        print(header)
        for gt_label in gt_labels:
            row = confusion_matrix[gt_label]
            print(
                f"{gt_label:<{col_w}}  "
                + "  ".join(f"{row.get(l, 0):>{col_w}}" for l in prediction_labels)
            )
    print()


def render_direction_html_report(
    summary_report: Dict[str, Any],
    title: str,
    bar_chart_direction_order: Tuple[str, ...],
) -> str:
    """
    Renders a direction benchmark summary_report as a standalone HTML page of
    tables, meant to be opened in a browser and copy-pasted (select-all,
    copy) directly into an Outlook email.

    Args:
        summary_report (Dict[str, Any]): The dict returned by a subject's
            run_..._benchmark function - must contain "statistics",
            "confusion_matrix", "label_metrics", "prediction_labels", and
            "results" keys.
        title (str): Heading/report title, e.g.
            "Pedestrian/Bicycle Direction Benchmark".
        bar_chart_direction_order (Tuple[str, ...]): Passed through to
            render_direction_bar_chart.

    Returns:
        str: A complete HTML document.
    """
    stats = summary_report["statistics"]
    confusion_matrix = summary_report["confusion_matrix"]
    label_metrics = summary_report["label_metrics"]
    prediction_labels = summary_report["prediction_labels"]

    overview_table = html_table(
        ["Metric", "Value"],
        [
            ["Images scanned", stats["total_images_scanned"]],
            ["Skipped (no direction token)", stats["skipped_unlabeled"]],
            ["Images evaluated", stats["images_evaluated"]],
            [
                "Exact direction match",
                f"{stats['accuracy'] * 100:.2f}% "
                f"({stats['exact_matches']}/{stats['images_evaluated']})",
            ],
        ],
    )

    metrics_table = html_table(
        ["Direction", "Support", "Precision", "Recall", "F1"],
        [
            [
                gt_label,
                m["support"],
                f"{m['precision'] * 100:.2f}%",
                f"{m['recall'] * 100:.2f}%",
                f"{m['f1'] * 100:.2f}%",
            ]
            for gt_label, m in sorted(label_metrics.items())
        ],
    )

    confusion_table = html_table(
        ["Truth \\ Predicted"] + prediction_labels,
        [
            [gt_label] + [confusion_matrix[gt_label].get(l, 0) for l in prediction_labels]
            for gt_label in sorted(confusion_matrix)
        ],
    )

    sections = [
        html_heading(title, level=2),
        overview_table,
        html_heading("Per-Direction Metrics", level=3),
        metrics_table,
        html_heading("Confusion Matrix (rows=truth, cols=predicted)", level=3),
        confusion_table,
    ]

    bar_chart_png = render_direction_bar_chart(
        summary_report["label_metrics"],
        bar_chart_direction_order,
        stats["images_evaluated"],
        stats["exact_matches"],
    )
    if bar_chart_png:
        bar_chart_b64 = base64.b64encode(bar_chart_png).decode("ascii")
        sections.append(html_heading("Ground Truth vs Correct Detections", level=3))
        sections.append(
            f'<img src="data:image/png;base64,{bar_chart_b64}" '
            f'alt="Bar chart of ground truth counts vs correct detections per direction, with accuracy percentages" '
            f'style="max-width:100%; height:auto;">'
        )

    return wrap_html_document(sections)
