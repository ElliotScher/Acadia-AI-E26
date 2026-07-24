"""
Pedestrian/Bicycle Direction Benchmark

Benchmarks src/detection/direction/pedestrian_direction.py's left/right/
front/back subject classification against a labeled image dataset - one with
no separate ground-truth file, since the ground truth is encoded directly in
each filename as a "_<direction>_"-delimited token that can appear anywhere
in the name, not just at the end (e.g. car_017_left.jpg, left_car_017.jpg,
and cam2_left_017.jpg are all recognized). Runs pedestrian/bicycle direction
detection on every recognized image, scores the subject with the largest
bounding box (the one the filename's label most plausibly refers to) against
that label, and reports the overall exact-match accuracy plus a full
confusion matrix and per-direction precision/recall/F1 - so a systematic
left/right or front/back mixup doesn't hide behind a single aggregate number.

The generic scoring/reporting machinery (confusion matrix, precision/recall,
bar chart, HTML report) is shared with vehicle_direction_benchmark.py via
direction_benchmark_common.py - see that module for why.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from detection.image_yolo import load_model
from ultralytics import YOLO

from src.detection.direction.pedestrian_direction import Direction, detect_pose, parse_poses
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

# The ground-truth labels a filename token may encode. "unknown" is included
# since pedestrian_direction.py can itself predict it, even though a human
# labeler would rarely deliberately tag an image that way.
VALID_LABELS: Tuple[str, ...] = ("left", "right", "front", "back", "unknown")

# Fixed display order for every direction bucket - ground truth labels plus
# the NO_DETECTION prediction-only outcome - so the bar chart's x-axis (and
# any other listing of "all buckets") is stable across runs instead of
# reshuffling based on whichever labels happen to appear in a given dataset.
_DIRECTION_ORDER: Tuple[str, ...] = (*VALID_LABELS, NO_DETECTION)

_TITLE = "Pedestrian/Bicycle Direction Benchmark"
_BANNER_TITLE = "PEDESTRIAN/BICYCLE DIRECTION BENCHMARK"


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
    return _render_direction_bar_chart(label_metrics, VALID_LABELS, total, exact_matches, "Pedestrian Ground Truth vs Correct Detections")


def predict_direction(
    model: YOLO,
    img_path: Path,
    conf: float = 0.25,
    min_points: int = 2,
) -> str:
    """
    Runs pedestrian_direction.py's detector on a single image and returns the
    predicted direction for its primary subject.

    When more than one person is detected, the one with the largest
    bounding box is treated as the primary subject - a labeled benchmark
    image is expected to depict one subject of interest, and the largest
    box is the one nearest the camera, matching what a human labeler would
    most likely have been looking at when naming the file.

    Args:
        model (YOLO): Loaded pose model, from detection.image_yolo.load_model.
        img_path (Path): Path to the image to classify.
        conf (float): Minimum detection/keypoint confidence, passed through
            to detect_pose and parse_poses. Defaults to 0.25.
        min_points (int): Minimum keypoints required per side before that
            side is trusted, passed through to parse_poses. Defaults to 2.

    Returns:
        str: One of pedestrian_direction.py's labels ("left", "right", "front",
            "back", "unknown"), or NO_DETECTION if no person was found.
    """
    raw_results = detect_pose(model, img_path, conf=conf)
    directions = parse_poses(
        raw_results, img_path, min_conf=conf, min_points=min_points
    )
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
            run_pedestrian_direction_benchmark.

    Returns:
        str: A complete HTML document.
    """
    return render_direction_html_report(summary_report, _TITLE, VALID_LABELS)


def run_pedestrian_direction_benchmark(
    input_dir: Union[str, Path],
    model_name: str = "yolo26s-pose.pt",
    model: Optional[YOLO] = None,
    conf: float = 0.25,
    min_points: int = 2,
    report: Optional[Union[str, Path]] = None,
    html_report: Optional[Union[str, Path]] = None,
    bar_chart: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs pedestrian/bicycle direction detection against every image under input_dir whose
    filename encodes a ground-truth direction, and scores accuracy.

    Args:
        input_dir (Union[str, Path]): Directory to search recursively for
            labeled images (see find_labeled_images).
        model_name (str): YOLO pose model weights to load if model isn't
            given. Defaults to "yolo26s-pose.pt", matching pedestrian_direction.py.
        model (Optional[YOLO]): Pre-loaded pose model. Defaults to None,
            which loads model_name via detection.image_yolo.load_model.
        conf (float): Minimum detection/keypoint confidence. Defaults to 0.25.
        min_points (int): Minimum keypoints required per side before that
            side is trusted, passed through to parse_poses. Defaults to 2.
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
    Main CLI entry point for the pedestrian/bicycle direction benchmark script.

    Raises:
        SystemExit: If the input directory is missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark src/detection/direction/pedestrian_direction.py's left/right/"
        "front/back classification against a labeled image dataset, where "
        "each image's ground-truth direction is encoded as a "
        '"_<direction>_"-delimited token anywhere in its filename - prefix, '
        "middle, or suffix (e.g. car_017_left.jpg, left_car_017.jpg, and "
        "cam2_left_017.jpg are all recognized as \"left\"). Reports the "
        "exact match rate, a confusion matrix, and per-direction "
        "precision/recall/F1."
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
        default="yolo26s-pose.pt",
        help="YOLO pose model weights to use (default: yolo26s-pose.pt).",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection/keypoint confidence (default: 0.25).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum keypoints required per side before that side is "
        "trusted (default: 2).",
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

    run_pedestrian_direction_benchmark(
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
