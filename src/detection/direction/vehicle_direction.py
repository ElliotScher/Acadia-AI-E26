"""
Vehicle Direction Detection

Determines whether a car is traveling left or right across the frame from a
single still image. Unlike pedestrian_direction.py, there's no COCO-style
body-keypoint model to lean on here - vehicles have no body pose - so this
uses a different pretrained checkpoint entirely: a YOLOv8-pose model
fine-tuned on the CarFusion dataset to output 14 keypoints per car
(https://github.com/Habib0905/Vehicle-Pose-Estimation - unlicensed research
code, no LICENSE file or declared license; a deliberate call to depend on it
anyway, not an oversight).

That repo never documents what its 14 keypoints actually are. The
FRONT_KEYPOINTS/REAR_KEYPOINTS split below was reverse-engineered by
visually inspecting labeled examples (plotting each keypoint index on cars
with a known ground-truth direction) and confirmed against a 187-image
labeled benchmark: predicting "right" when the front cluster's average x
exceeds the rear cluster's, "left" otherwise, scored 94% accuracy with 100%
precision on every image it was confident enough to commit to (the misses
were all "couldn't decide", never a wrong left/right call). Indices 8 and 13
were consistently near-zero confidence across every example checked -
almost certainly occluded far-side points - and are excluded entirely.

Also unlike pedestrian_direction.py, this checkpoint must always be run on
the FULL, uncropped frame. Pre-cropping tightly to the vehicle - even with
generous padding, even replicating this repo's own preprocessing - was
empirically found to collapse its detection confidence and accuracy (94%
full-frame vs. 9% on a tight crop in testing). It appears to need
surrounding scene context to localize the vehicle at all, so
detect_vehicle_pose deliberately takes no crop/box parameter. If a
vehicle's location is already known from an upstream detector, run
detect_vehicle_pose on the full image regardless and use select_best_match
to pick the matching detection afterward by IoU - don't crop beforehand.

Relatedly, this checkpoint's raw box confidence is poorly calibrated - it
rarely exceeds ~0.3 even on a correct detection. DEFAULT_BOX_CONF is
intentionally set far lower than a typical YOLO confidence threshold; the
low scores just aren't meaningfully scaled, but which candidate ranks
highest is still reliable (precision held at 100% all the way down to a
0.01 threshold in testing).

Cars only need a left/right verdict, not front/back - "left"/"right"
describes which way the vehicle's front is pointed, matching the direction
convention already used by src/processing/video_entityprofiler.py's
motion-based track-level direction (first vs. last frame center x). That
tracker only works across a whole video track; this module exists for the
case it doesn't cover: a single still image with no motion history.
"""

import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import List, Optional, Tuple, Union

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from detection.image_yolo import load_model
from utility.fetch_vehicle_pose_weights import (
    DEFAULT_WEIGHTS_PATH,
    fetch_vehicle_pose_weights,
)
from utility.geometryutils import Rectangle

# Empirically reverse-engineered from the model's undocumented 14-keypoint
# CarFusion-style schema (see module docstring) - not sourced from any spec.
FRONT_KEYPOINTS: Tuple[int, ...] = (0, 1, 4, 5, 9, 10)  # near-side front wheel, headlight/fender, front roof
REAR_KEYPOINTS: Tuple[int, ...] = (2, 3, 6, 7, 11, 12)  # near-side rear wheel, taillight/bumper, rear roof
# Indices 8 and 13 are deliberately absent from both clusters above - see
# module docstring.

MIN_KEYPOINT_CONF = 0.5
MIN_POINTS_PER_SIDE = 2
DEFAULT_BOX_CONF = 0.01

VEHICLE_CLASSES: List[int] = [0]  # this checkpoint's only class is "car"

# Global object counter for CLI summaries and tracking
total_counts: dict[str, int] = {
    "left": 0,
    "right": 0,
    "unknown": 0,
}
total_counts_lock = threading.Lock()


@dataclass
class VehicleDirection:
    """
    Struct representing the left/right travel direction of a detected
    vehicle. Holds the image coordinates of the bounding box, direction
    label, an informational confidence figure, and the associated image path.
    """

    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates
    label: str  # "left", "right", or "unknown"
    confidence: float  # fraction of front/rear keypoints that cleared MIN_KEYPOINT_CONF - informational, not a probability
    image_path: Path  # Path to the associated image


def detect_vehicle_pose(
    model: YOLO,
    img_path: Union[Path, str],
    conf: float = DEFAULT_BOX_CONF,
    classes: List[int] = VEHICLE_CLASSES,
) -> List[Results]:
    """
    Runs the vehicle-pose model on the full image and returns raw results.

    Deliberately takes no crop/box parameter - see the module docstring for
    why pre-cropping this specific checkpoint backfires.
    """
    img = Image.open(img_path)
    results = model.predict(source=img, conf=conf, classes=classes, verbose=False)
    return results


def parse_vehicle_directions(
    data: List[Results],
    image_path: Union[Path, str],
    min_conf: float = MIN_KEYPOINT_CONF,
    min_points: int = MIN_POINTS_PER_SIDE,
) -> List[VehicleDirection]:
    """
    Parses raw vehicle-pose results into a normalized list of
    VehicleDirection objects, one per detected vehicle.
    """
    results: List[VehicleDirection] = []
    path = image_path if isinstance(image_path, Path) else Path(image_path)

    for datum in data:
        if datum.keypoints is None:
            continue

        boxes = datum.boxes
        for i, kpts_xy in enumerate(datum.keypoints.xy):
            kpts_conf = datum.keypoints.conf[i]

            front_xs = [
                kpts_xy[idx][0].item()
                for idx in FRONT_KEYPOINTS
                if kpts_conf[idx].item() >= min_conf
            ]
            rear_xs = [
                kpts_xy[idx][0].item()
                for idx in REAR_KEYPOINTS
                if kpts_conf[idx].item() >= min_conf
            ]

            total_kpts = len(FRONT_KEYPOINTS) + len(REAR_KEYPOINTS)
            confidence = (len(front_xs) + len(rear_xs)) / total_kpts

            if len(front_xs) < min_points or len(rear_xs) < min_points:
                label = "unknown"
            else:
                front_avg = sum(front_xs) / len(front_xs)
                rear_avg = sum(rear_xs) / len(rear_xs)
                if front_avg == rear_avg:
                    label = "unknown"
                else:
                    label = "right" if front_avg > rear_avg else "left"

            if boxes is not None and i < len(boxes):
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                box = (int(x1), int(y1), int(x2), int(y2))
            else:
                box = (0, 0, 0, 0)

            results.append(VehicleDirection(box, label, confidence, path))

    return results


def select_best_match(
    directions: List[VehicleDirection], reference_box: Tuple[int, int, int, int]
) -> Optional[VehicleDirection]:
    """
    Picks whichever detected VehicleDirection best overlaps a known
    reference box (e.g. an upstream object detector's tracked bounding box),
    by IoU. This is how to combine an already-known vehicle location with
    this module without cropping the input image - see detect_vehicle_pose's
    docstring for why cropping isn't supported here.

    Args:
        directions (List[VehicleDirection]): Detections from
            parse_vehicle_directions, from a full-frame prediction.
        reference_box (Tuple[int, int, int, int]): (x1, y1, x2, y2) of the
            known vehicle location.

    Returns:
        Optional[VehicleDirection]: The detection with the highest IoU
            against reference_box, or None if directions is empty.
    """
    if not directions:
        return None

    rx1, ry1, rx2, ry2 = reference_box
    ref_rect = Rectangle(x=rx1, y=ry1, w=rx2 - rx1, h=ry2 - ry1)

    def iou(direction: VehicleDirection) -> float:
        x1, y1, x2, y2 = direction.box
        rect = Rectangle(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        return Rectangle.compute_iou(rect, ref_rect)

    return max(directions, key=iou)


def save_annotated_results(
    img_path: Path,
    directions: List[VehicleDirection],
    input_folder: Path,
    output_folder: Path,
) -> None:
    """
    Saves the annotated copies of the image (or the original if no detections)
    to the output folder, preserving the input directory structure.
    """
    image = cv2.imread(str(img_path))
    if image is None:
        return

    out_path_base = output_folder / img_path.relative_to(input_folder)
    out_path_base.parent.mkdir(parents=True, exist_ok=True)

    if not directions:
        cv2.imwrite(str(out_path_base), image)
    else:
        for i, direction in enumerate(directions):
            x1, y1, x2, y2 = direction.box
            label = direction.label
            image_copy = image.copy()
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

            out_path = out_path_base.with_name(
                f"{out_path_base.stem}-{i}-{label}{out_path_base.suffix}"
            )
            cv2.imwrite(str(out_path), image_copy)


def process_single_image(
    model: YOLO,
    img_path: Path,
    input_folder: Path,
    output_folder: Path,
    save_images: bool = True,
    conf: float = DEFAULT_BOX_CONF,
    min_points: int = MIN_POINTS_PER_SIDE,
) -> List[VehicleDirection]:
    """
    Processes a single image: runs detection, parses results, and optionally
    saves output. Always runs on the full image - see detect_vehicle_pose.
    """
    raw_results = detect_vehicle_pose(model, img_path, conf)
    directions = parse_vehicle_directions(raw_results, img_path, conf, min_points)

    if save_images:
        save_annotated_results(img_path, directions, input_folder, output_folder)

    return directions


def process_image_worker(
    img_paths: List[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool,
    conf: float,
    progress_bar: Optional[tqdm],
    min_points: int = MIN_POINTS_PER_SIDE,
) -> None:
    """
    Worker function executed by threads in batch mode.
    """
    model = load_model(model_name)

    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            if progress_bar:
                progress_bar.update(1)
            continue

        directions = process_single_image(
            model=model,
            img_path=img_path,
            input_folder=input_folder,
            output_folder=output_folder,
            save_images=save_images,
            conf=conf,
            min_points=min_points,
        )

        with total_counts_lock:
            for direction in directions:
                total_counts[direction.label] = total_counts.get(direction.label, 0) + 1

        if progress_bar:
            progress_bar.update(1)


def batch_detect_and_process(
    img_paths: List[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool = True,
    conf: float = DEFAULT_BOX_CONF,
    num_threads: int = 1,
    show_progress: bool = True,
    min_points: int = MIN_POINTS_PER_SIDE,
) -> None:
    """
    Performs multi-threaded batch detection and processing on a list of images.
    """
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(
            total=len(img_paths), desc="Processing Detections", unit="image"
        )

    chunk_size = max(1, len(img_paths) // num_threads)
    threads: List[Thread] = []

    for i in range(num_threads):
        start = i * chunk_size
        end = None if i == num_threads - 1 else (i + 1) * chunk_size
        imgs = img_paths[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=process_image_worker,
            args=(
                imgs,
                input_folder,
                output_folder,
                model_name,
                save_images,
                conf,
                progress_bar,
                min_points,
            ),
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    if progress_bar:
        progress_bar.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process images in an input directory using the vehicle-"
        "pose model and save annotated results to an output directory. "
        "Always runs on full frames, never pre-cropped - see the module "
        "docstring in src/detection/direction/vehicle_direction.py."
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Path to the vehicle-pose model weights. Defaults to "
        f"{DEFAULT_WEIGHTS_PATH}, auto-downloading it there first via "
        "fetch_vehicle_pose_weights if it isn't already cached - unlike "
        "pedestrian_direction.py's yolo26s-pose.pt, this checkpoint isn't on "
        "any public model registry Ultralytics can auto-resolve by name.",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate to detections (default: 1).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images to the output directory.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_BOX_CONF,
        help=f"Detection confidence threshold (default: {DEFAULT_BOX_CONF} - "
        "deliberately low, see the module docstring).",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_POINTS_PER_SIDE,
        help="Minimum number of confident keypoints required on each of the "
        "front/rear clusters before trusting that side (default: "
        f"{MIN_POINTS_PER_SIDE}).",
    )
    args = parser.parse_args()

    input_folder = Path(args.input_dir).resolve()
    output_folder = Path(args.output_dir).resolve()

    if not input_folder.is_dir():
        print(
            f"Error: Input directory '{input_folder}' does not exist or is not a directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    if args.cores <= 0:
        print(
            "Error: The number of allocated CPU cores must be at least 1.",
            file=sys.stderr,
        )
        sys.exit(1)
    thread_count = args.cores

    all_images = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not all_images:
        print("No matching images found in the input directory.")
        return

    if args.model:
        model_path = Path(args.model)
    else:
        print(f"No --model given, fetching default weights to {DEFAULT_WEIGHTS_PATH}...")
        model_path = fetch_vehicle_pose_weights()

    # Configure PyTorch CPU thread count to respect our core allocation globally
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"Allocating {thread_count} CPU core(s) to vehicle-pose detections...")

    with total_counts_lock:
        total_counts.clear()
        total_counts["left"] = 0
        total_counts["right"] = 0
        total_counts["unknown"] = 0

    batch_detect_and_process(
        img_paths=all_images,
        input_folder=input_folder,
        output_folder=output_folder,
        model_name=str(model_path),
        save_images=not args.no_save,
        conf=args.conf,
        num_threads=thread_count,
        show_progress=True,
        min_points=args.min_points,
    )

    print("\n" + "=" * 30)
    print("      DETECTION SUMMARY")
    print("=" * 30)
    for label, count in sorted(total_counts.items()):
        print(f"Total {label}:".ljust(15) + f"{count}")
    print("=" * 30)


if __name__ == "__main__":
    main()
