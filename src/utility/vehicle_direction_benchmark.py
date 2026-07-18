"""
Vehicle Direction Benchmark

Benchmarks src/detection/direction/vehicle_direction.py's left/right car
classification against a labeled image dataset - one with no separate
ground-truth file, since the ground truth is encoded directly in each
filename as a "_<direction>_"-delimited token that can appear anywhere in
the name, not just at the end (e.g. entity_1_right_car.jpg and
left_entity_2.jpg are both recognized). Runs vehicle direction detection on
every recognized image, scores the vehicle with the largest bounding box
(the one the filename's label most plausibly refers to) against that label,
and reports the overall exact-match accuracy plus a full confusion matrix
and per-direction precision/recall/F1.

Unlike pedestrian_direction_benchmark.py, valid ground truth here is only
"left"/"right" - cars have no meaningful "front"/"back" analog the way a
person's facing direction does (see vehicle_direction.py's module
docstring), so a filename token of "front"/"back"/"unknown" is treated as
unrecognized rather than accepted.

The generic scoring/reporting machinery (confusion matrix, precision/recall,
bar chart, HTML report) is shared with pedestrian_direction_benchmark.py via
direction_benchmark_common.py - see that module for why.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from detection.image_yolo import load_model
from ultralytics import YOLO

from src.detection.direction.vehicle_direction import (
    DEFAULT_BOX_CONF,
    MIN_POINTS_PER_SIDE,
    detect_vehicle_pose,
    parse_vehicle_directions,
)
from src.utility.direction_benchmark_common import (
    IMAGE_SUFFIXES,
    NO_DETECTION,
    box_area,
    build_confusion_matrix,
    compute_label_metrics,
)
from src.utility.direction_benchmark_common import compute_label_counts as _compute_label_counts
from src.utility.direction_benchmark_common import find_labeled_images as _find_labeled_images
from src.utility.direction_benchmark_common import label_from_filename
from src.utility.direction_benchmark_common import (
    print_direction_benchmark_summary,
)
from src.utility.direction_benchmark_common import (
    render_direction_bar_chart as _render_direction_bar_chart,
)
from src.utility.direction_benchmark_common import (
    render_direction_html_report,
)
from src.utility.fetch_vehicle_pose_weights import fetch_vehicle_pose_weights

# Cars only need a left/right verdict - see the module docstring.
VALID_LABELS: Tuple[str, ...] = ("left", "right")

# Prediction-only outcome, distinct from NO_DETECTION: vehicle_direction.py
# found a car but couldn't get enough confident keypoints on one or both
# sides to call a direction - see parse_vehicle_directions. Unlike
# pedestrian_direction.py, "unknown" is never valid ground truth for a
# vehicle (not part of VALID_LABELS above), so it has to be added to the
# bucket list separately rather than inherited from VALID_LABELS.
_UNKNOWN = "unknown"

# Fixed display order for every direction bucket - ground truth labels plus
# the two prediction-only outcomes.
_DIRECTION_ORDER: Tuple[str, ...] = (*VALID_LABELS, NO_DETECTION, _UNKNOWN)

_TITLE = "Vehicle Direction Benchmark"
_BANNER_TITLE = "VEHICLE DIRECTION BENCHMARK"


def ground_truth_label_from_filename(path: Union[str, Path]) -> Optional[str]:
    """
    Recovers an image's ground-truth direction from a direction-label token
    appearing anywhere in its filename - see
    direction_benchmark_common.label_from_filename.
    """
    return label_from_filename(path, VALID_LABELS)


def find_labeled_images(input_dir: Path) -> Dict[Path, str]:
    """
    Recursively finds every image under input_dir whose filename encodes a
    recognized ground-truth direction - see
    direction_benchmark_common.find_labeled_images.
    """
    return _find_labeled_images(input_dir, VALID_LABELS)


def compute_label_counts(
    results: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Tallies ground-truth vs predicted counts per direction bucket - see
    direction_benchmark_common.compute_label_counts.
    """
    return _compute_label_counts(results, _DIRECTION_ORDER)


def render_direction_bar_chart(results: List[Dict[str, Any]]) -> bytes:
    """
    Renders a bar chart comparing each ground-truth direction's count
    against how many were correctly detected, plus an overall accuracy
    figure - see direction_benchmark_common.render_direction_bar_chart.
    """
    confusion_matrix = build_confusion_matrix(results)
    label_metrics = compute_label_metrics(confusion_matrix)
    total = len(results)
    exact_matches = sum(m["true_positives"] for m in label_metrics.values())
    return _render_direction_bar_chart(label_metrics, VALID_LABELS, total, exact_matches, "Vehicle Ground Truth vs Correct Detections")


def predict_direction(
    model: YOLO,
    img_path: Path,
    conf: float = DEFAULT_BOX_CONF,
    min_points: int = MIN_POINTS_PER_SIDE,
) -> str:
    """
    Runs vehicle_direction.py's detector on a single image and returns the
    predicted direction for its primary vehicle.

    Always runs on the full image - vehicle_direction.py's detector requires
    it, see detect_vehicle_pose's docstring. When more than one vehicle is
    detected, the one with the largest bounding box is treated as the
    primary subject, matching pedestrian_direction_benchmark.py's convention
    for a labeled benchmark image expected to depict one subject of interest.

    Args:
        model (YOLO): Loaded vehicle-pose model, from
            detection.image_yolo.load_model.
        img_path (Path): Path to the image to classify.
        conf (float): Minimum detection confidence, passed through to
            detect_vehicle_pose. Defaults to DEFAULT_BOX_CONF.
        min_points (int): Minimum keypoints required per side before that
            side is trusted, passed through to parse_vehicle_directions.
            Defaults to MIN_POINTS_PER_SIDE.

    Returns:
        str: "left", "right", or "unknown" (see vehicle_direction.py's
            labels), or NO_DETECTION if no vehicle was found.
    """
    raw_results = detect_vehicle_pose(model, img_path, conf=conf)
    directions = parse_vehicle_directions(raw_results, img_path, min_points=min_points)
    if not directions:
        return NO_DETECTION
    primary = max(directions, key=lambda d: box_area(d.box))
    return primary.label


def render_html_report(summary_report: Dict[str, Any]) -> str:
    """
    Renders a benchmark summary_report as a standalone HTML page of tables,
    meant to be opened in a browser and copy-pasted (select-all, copy)
    directly into an Outlook email.

    Args:
        summary_report (Dict[str, Any]): The dict returned by
            run_vehicle_direction_benchmark.

    Returns:
        str: A complete HTML document.
    """
    return render_direction_html_report(summary_report, _TITLE, VALID_LABELS)


def run_vehicle_direction_benchmark(
    input_dir: Union[str, Path],
    model_name: Optional[str] = None,
    model: Optional[YOLO] = None,
    conf: float = DEFAULT_BOX_CONF,
    min_points: int = MIN_POINTS_PER_SIDE,
    report: Optional[Union[str, Path]] = None,
    html_report: Optional[Union[str, Path]] = None,
    bar_chart: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs vehicle direction detection against every image under input_dir
    whose filename encodes a ground-truth direction, and scores accuracy.

    Args:
        input_dir (Union[str, Path]): Directory to search recursively for
            labeled images (see find_labeled_images).
        model_name (Optional[str]): Vehicle-pose model weights path to load
            if model isn't given. Defaults to None, which auto-downloads the
            default checkpoint via fetch_vehicle_pose_weights - unlike
            pedestrian_direction_benchmark.py's yolo26s-pose.pt, this
            checkpoint isn't on any public registry Ultralytics can resolve
            by name.
        model (Optional[YOLO]): Pre-loaded vehicle-pose model. Defaults to
            None, which loads model_name (or the auto-fetched default) via
            detection.image_yolo.load_model.
        conf (float): Minimum detection confidence. Defaults to
            DEFAULT_BOX_CONF - deliberately low, see vehicle_direction.py's
            module docstring.
        min_points (int): Minimum keypoints required per side before that
            side is trusted, passed through to parse_vehicle_directions.
            Defaults to MIN_POINTS_PER_SIDE.
        report (Optional[Union[str, Path]]): Filepath to save a detailed
            per-image JSON report. Defaults to None.
        html_report (Optional[Union[str, Path]]): Filepath to save an HTML
            version of the summary, with real <table> elements that survive
            copy-paste into Outlook (open in a browser, select all, copy,
            paste). The same bar chart saved by bar_chart= is also embedded
            in this page. Defaults to None.
        bar_chart (Optional[Union[str, Path]]): Filepath to save a standalone
            PNG of the ground-truth-vs-predicted direction bar chart - see
            render_direction_bar_chart. Defaults to None.

    Returns:
        Dict[str, Any]: Summary report dict containing overall accuracy,
            a confusion matrix, per-direction precision/recall/F1, and
            per-image detail.
    """
    input_folder = Path(input_dir)

    all_images = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    labeled_images = find_labeled_images(input_folder)
    skipped_unlabeled = len(all_images) - len(labeled_images)

    if model is None:
        if model_name is None:
            model_name = str(fetch_vehicle_pose_weights())
        model = load_model(model_name)

    results: List[Dict[str, Any]] = []
    exact_matches = 0

    for img_path in sorted(labeled_images):
        ground_truth = labeled_images[img_path]
        predicted = predict_direction(model, img_path, conf=conf, min_points=min_points)
        is_match = predicted == ground_truth
        exact_matches += int(is_match)

        results.append(
            {
                "image": str(img_path.relative_to(input_folder)),
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": is_match,
            }
        )

    total = len(results)
    accuracy = exact_matches / total if total else 0.0

    confusion_matrix = build_confusion_matrix(results)
    label_metrics = compute_label_metrics(confusion_matrix)

    prediction_labels = sorted(
        {pred for row in confusion_matrix.values() for pred in row}
    )

    summary_report: Dict[str, Any] = {
        "metadata": {
            "input_dir": str(input_folder),
            "model_name": model_name,
            "conf": conf,
            "min_points": min_points,
        },
        "statistics": {
            "total_images_scanned": len(all_images),
            "skipped_unlabeled": skipped_unlabeled,
            "images_evaluated": total,
            "exact_matches": exact_matches,
            "accuracy": accuracy,
        },
        "confusion_matrix": confusion_matrix,
        "label_metrics": label_metrics,
        "prediction_labels": prediction_labels,
        "results": results,
    }

    print_direction_benchmark_summary(
        title=_BANNER_TITLE,
        total_images_scanned=len(all_images),
        skipped_unlabeled=skipped_unlabeled,
        total=total,
        exact_matches=exact_matches,
        accuracy=accuracy,
        confusion_matrix=confusion_matrix,
        label_metrics=label_metrics,
        prediction_labels=prediction_labels,
    )

    if report:
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    if bar_chart:
        with open(bar_chart, "wb") as f:
            f.write(render_direction_bar_chart(results))

    if html_report:
        with open(html_report, "w") as f:
            f.write(render_html_report(summary_report))

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the vehicle direction benchmark script.

    Raises:
        SystemExit: If the input directory is missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark src/detection/direction/vehicle_direction.py's "
        "left/right classification against a labeled image dataset, where "
        "each image's ground-truth direction is encoded as a "
        '"_<direction>_"-delimited token anywhere in its filename (e.g. '
        'entity_1_right_car.jpg). Reports the exact match rate, a confusion '
        "matrix, and per-direction precision/recall/F1."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory to search recursively for labeled images.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to the vehicle-pose model weights. Defaults to "
        "auto-downloading the default checkpoint via "
        "fetch_vehicle_pose_weights if not given.",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=DEFAULT_BOX_CONF,
        help=f"Minimum detection confidence (default: {DEFAULT_BOX_CONF}).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS_PER_SIDE,
        help="Minimum keypoints required per side before that side is "
        f"trusted (default: {MIN_POINTS_PER_SIDE}).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save a detailed per-image JSON report.",
    )
    parser.add_argument(
        "-w",
        "--html",
        type=str,
        default=None,
        help="Path to save an HTML version of the summary. Open it in a "
        "browser and copy/paste the tables directly into an Outlook email. "
        "Includes the same bar chart as --bar-chart.",
    )
    parser.add_argument(
        "--bar-chart",
        type=str,
        default=None,
        help="Path to save a standalone PNG bar chart comparing ground-truth "
        "vs predicted direction counts.",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_dir)
    if not input_folder.is_dir():
        parser.error(f"{input_folder} is not a directory.")

    run_vehicle_direction_benchmark(
        input_dir=input_folder,
        model_name=args.model,
        conf=args.conf,
        min_points=args.min_points,
        report=args.report,
        html_report=args.html,
        bar_chart=args.bar_chart,
    )


if __name__ == "__main__":
    main()
