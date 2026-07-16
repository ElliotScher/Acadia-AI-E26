import argparse
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Union

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from detection.classes import TARGET_CLASSES
from detection.image_yolo import load_model

left_points = (1, 3, 5, 7, 9, 11, 13, 15)
right_points = (2, 4, 6, 8, 10, 12, 14, 16)
front_points = (1, 2, 3, 9, 10)
back_points = (5, 6, 11, 12)

# Global object counter for CLI summaries and tracking
total_counts: dict[str, int] = {
    "left": 0,
    "right": 0,
    "front": 0,
    "back": 0,
    "unknown": 0,
}
total_counts_lock = threading.Lock()


@dataclass
class Direction:
    """
    Struct representing the direction of a detected object.
    Holds the image coordinates of the bounding box, direction,
    an informational label, and the associated image path.
    """

    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates
    left_right: int  # -1 is left, 0 is unknown, 1 is right
    front_back: int  # -1 is back, 0 is unknown, 1 is front
    image_path: Path  # Path to the associated image
    label: str  # Informational label


def detect_pose(
    model: YOLO,
    img_path: Union[Path, str],
    conf: float = 0.25,
    classes: list[int] = TARGET_CLASSES,
    box: tuple[int, int, int, int] | None = None,
) -> list[Results]:
    """
    Runs YOLO model prediction and returns raw results.
    """
    img = Image.open(img_path)
    if box:
        img = img.crop(box)
    results = model.predict(source=img, conf=conf, classes=classes, verbose=False)
    return results


def parse_poses(
    data: list[Results],
    image_path: Union[Path, str],
    min_conf: float = 0.25,
    box: tuple[int, int, int, int] | None = None,
    min_points: int = 2,
) -> list[Direction]:
    """
    Parses raw YOLO results into a normalized list of Direction objects.
    """
    results: list[Direction] = []
    for datum in data:
        if datum.keypoints is None:
            continue

        for person, kps in enumerate(datum.keypoints.xy):
            left_total = 0
            left_count = 0
            right_total = 0
            right_count = 0
            front_total = 0
            front_count = 0
            back_total = 0
            back_count = 0

            x1 = 99999
            y1 = 99999
            x2 = 0
            y2 = 0

            for i, (x, y) in enumerate(kps):
                if datum.keypoints.conf[person][i].item() >= min_conf:  # type: ignore
                    if i in left_points and x > 0 and y > 0:
                        left_total += x
                        left_count += 1
                    elif i in right_points and x > 0 and y > 0:
                        right_total += x
                        right_count += 1

                    if i in front_points and x > 0 and y > 0:
                        front_total += x
                        front_count += 1
                    elif i in back_points and x > 0 and y > 0:
                        back_total += x
                        back_count += 1

                    if x < x1:
                        x1 = int(x)
                    if x > x2:
                        x2 = int(x)
                    if y < y1:
                        y1 = int(y)
                    if y > y2:
                        y2 = int(y)

            left_average = left_total / (left_count or 1)
            right_average = right_total / (right_count or 1)
            front_average = front_total / (front_count or 1)
            back_average = back_total / (back_count or 1)

            if (
                front_average == back_average
                or front_count < min_points
                or back_count < min_points
            ):
                lr = 0
            elif front_average < back_average:
                lr = -1
            else:
                lr = 1

            if (
                left_average == right_average
                or left_count < min_points
                or right_count < min_points
            ):
                fb = 0
            elif left_average < right_average:
                fb = -1
            else:
                fb = 1

            if lr == 0 and fb == 0:
                label = "unknown"
            elif abs(left_average - right_average) > abs(front_average - back_average):
                label = "right" if lr == 1 else "left"
            else:
                label = "front" if fb == 1 else "back"

            if not box:
                box = (x1, y1, x2, y2)

            results.append(
                Direction(
                    box,
                    lr,
                    fb,
                    image_path if isinstance(image_path, Path) else Path(image_path),
                    label,
                )
            )

    return results


def save_annotated_results(
    img_path: Path, directions: list[Direction], input_folder: Path, output_folder: Path
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
        for i, dir in enumerate(directions):
            x1, y1, x2, y2 = dir.box
            label = dir.label
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
    conf: float = 0.25,
    classes: list[int] = TARGET_CLASSES,
    box: tuple[int, int, int, int] | None = None,
    min_points: int = 2,
) -> list[Direction]:
    """
    Processes a single image: runs detection, parses results, and optionally saves output.
    """
    raw_results = detect_pose(model, img_path, conf, classes, box)
    directions = parse_poses(raw_results, img_path, conf, box, min_points)

    if save_images:
        save_annotated_results(img_path, directions, input_folder, output_folder)

    return directions


def process_image_worker(
    img_paths: list[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool,
    conf: float,
    progress_bar: tqdm | None,
    classes: list[int] = TARGET_CLASSES,
    min_points: int = 2,
) -> None:
    """
    Worker function executed by threads in batch mode.
    """
    model = load_model(model_name)

    for i in range(len(img_paths)):
        img_path = img_paths[i]
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
            classes=classes,
            min_points=min_points,
        )

        # Update global counts
        for dir in directions:
            label = dir.label
            with total_counts_lock:
                total_counts[label] = total_counts.get(label, 0) + 1

        if progress_bar:
            progress_bar.update(1)


def batch_detect_and_process(
    img_paths: list[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool = True,
    conf: float = 0.25,
    num_threads: int = 1,
    show_progress: bool = True,
    classes: list[int] = TARGET_CLASSES,
    min_points: int = 1,
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
    threads: list[Thread] = []

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
                classes,
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
        description="Process images in an input directory using YOLO and save results to an output directory."
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo26s-pose.pt",
        help="YOLO model weights to use (default: yolo26s-pose.pt).",
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate to YOLO detections (default: 1).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images to the output directory.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=2,
        help="Minimum number of points that need to be detected in a particular direction to consider that direction",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names or class IDs to detect (e.g. 0 2 or person car). Default: person, bike, car.",
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

    # Configure PyTorch CPU thread count to respect our core allocation globally
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Load model to resolve classes
    model = load_model(args.model)

    classes_of_interest = []
    if args.classes is not None:
        # Create a reverse mapping: lowercase_name -> id
        name_to_id = {v.lower(): k for k, v in model.names.items()}
        for c in args.classes:
            if c.isdigit():
                classes_of_interest.append(int(c))
            else:
                c_low = c.lower()
                if c_low in name_to_id:
                    classes_of_interest.append(name_to_id[c_low])
                else:
                    print(
                        f"Warning: Class name '{c}' not found in model classes. Ignoring.",
                        file=sys.stderr,
                    )

        # Deduplicate
        classes_of_interest = list(dict.fromkeys(classes_of_interest))

        if not classes_of_interest:
            print(
                "Error: No valid classes resolved from the provided --classes argument.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        classes_of_interest = TARGET_CLASSES

    print(f"Allocating {thread_count} CPU core(s) to YOLO detections...")

    # Recurse over all subdirectories in the input directory
    all_images = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not all_images:
        print("No matching images found in the input directory.")
        return

    # Reset and initialize total counts for each direction
    with total_counts_lock:
        total_counts.clear()
        total_counts["left"] = 0
        total_counts["right"] = 0
        total_counts["front"] = 0
        total_counts["back"] = 0
        total_counts["unknown"] = 0

    batch_detect_and_process(
        img_paths=all_images,
        input_folder=input_folder,
        output_folder=output_folder,
        model_name=args.model,
        save_images=not args.no_save,
        num_threads=thread_count,
        show_progress=True,
        classes=classes_of_interest,
        min_points=args.min_points,
    )

    print("\n" + "=" * 30)
    print("      DETECTION SUMMARY")
    print("=" * 30)
    for label, count in sorted(total_counts.items()):
        if label == "person":
            display_label = "people"
        elif label in ("car", "bike"):
            display_label = f"{label}s"
        else:
            display_label = label
        print(f"Total {display_label}:".ljust(15) + f"{count}")
    print("=" * 30)


if __name__ == "__main__":
    main()
