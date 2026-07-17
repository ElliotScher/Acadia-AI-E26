"""
YOLO Image Object Detection Utility

Processes images in an input directory using YOLO, maps target categories (e.g., bus/truck to car).
"""

import argparse
import datetime
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from detection.classes import CLASS_ID_MAPPING

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from utility.geometryutils import Rectangle

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
DEFAULT_TARGET_CLASSES = [0, 2, 5, 7]

# Initialize Logger
logger = logging.getLogger("yolo_detection")


@dataclass
class DetectionResult:
    """
    Holds the YOLO detection results for a single image.

    Args:
        image_path (Path): Path to the source image.
        image (np.ndarray): BGR image representation.
        boxes (List[Tuple[Rectangle, int, float]]): List of (rectangle, id, confidence) detections.
    """

    image_path: Path
    image: np.ndarray | None  # BGR image representation
    boxes: List[Tuple[Rectangle, int, float]]  # List of (rectangle, id, confidence)

def load_model(model_name: str) -> YOLO:
    """
    Loads and returns a YOLO model from the given path or name.

    Args:
        model_name (str): YOLO weights name.
    """
    return YOLO(model_name)


def process_images(
    img_paths: List[Path],
    model_name: str,
    progress_bar: Optional[tqdm] = None,
    inclusion_region: Optional[Rectangle] = None,
    conf_threshold: float = 0.25,
    target_classes: Optional[List[int]] = None,
) -> List[DetectionResult]:
    """
    Processes a list of image paths using YOLO and extracts detection boxes and images.
    Does not write anything to disk.

    Args:
        img_paths (List[Path]): List of image paths to process.
        model_name (str): YOLO weights name or path.
        progress_bar (tqdm): Thread-safe progress bar instance.
        inclusion_region (Optional[Rectangle]): Optional spatial filter region.
        conf_threshold (float): Minimum confidence threshold for detections.
        target_classes (Optional[List[int]]): List of COCO class IDs to filter.

    Returns:
        List[DetectionResult]: List of detection results per image.
    """
    classes_list = (
        target_classes if target_classes is not None else DEFAULT_TARGET_CLASSES
    )
    results_list: List[DetectionResult] = []

    try:
        thread_model = YOLO(model_name)
    except Exception as e:
        logger.error("Failed to load YOLO model '%s': %s", model_name, e)
        if progress_bar:
            progress_bar.update(len(img_paths))
        return []

    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            if progress_bar:
                progress_bar.update(1)
            continue

        try:
            results = thread_model.predict(
                source=str(img_path),
                conf=conf_threshold,
                classes=classes_list,
                verbose=False,
            )

            image = cv2.imread(str(img_path))
            if image is None:
                if progress_bar:
                    progress_bar.update(1)
                continue

            boxes_found: List[Tuple[Rectangle, int, float]] = []
            for r in results:
                for box in r.boxes: # type:ignore
                    cls = int(box.cls[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1

                    # Apply crop region filter if provided
                    if inclusion_region is not None:
                        box_rect = Rectangle(x1, y1, w, h)
                        if not Rectangle.bounding_box_intersects(
                            box_rect, inclusion_region
                        ):
                            continue

                    conf = float(box.conf[0])
                    rect = Rectangle(x=x1, y=y1, w=w, h=h)
                    boxes_found.append((rect, cls, conf))

            results_list.append(
                DetectionResult(image_path=img_path, image=image, boxes=boxes_found)
            )

        except Exception as e:
            logger.error("Error processing image %s: %s", img_path, e)
        finally:
            if progress_bar:
                progress_bar.update(1)

    return results_list


def save_annotated_images(
    results: List[DetectionResult],
    input_folder: Path,
    output_folder: Path,
) -> None:
    """
    Saves detection results to disk with annotations.

    Args:
        results (List[DetectionResult]): YOLO detection outputs.
        input_folder (Path): Source directory for relative path resolution.
        output_folder (Path): Destination directory for saved images.
    """
    for res in results:
        img_path = res.image_path
        image = res.image
        boxes = res.boxes

        out_path_base = output_folder / img_path.relative_to(input_folder)
        out_path_base.parent.mkdir(parents=True, exist_ok=True)

        if image is not None:
            if not boxes:
                cv2.imwrite(str(out_path_base), image)
                logger.debug("Saved original image with no detections: %s", out_path_base)
            else:
                for i, (rect, label, conf) in enumerate(boxes):
                    image_copy = image.copy()
                    cv2.rectangle(
                        image_copy,
                        (rect.x, rect.y),
                        (rect.x + rect.w, rect.y + rect.h),
                        (0, 255, 0),
                        thickness=5,
                    )
                    out_path = out_path_base.with_name(
                        f"{out_path_base.stem}-{i}-{label}{out_path_base.suffix}"
                    )
                    cv2.imwrite(str(out_path), image_copy)
                    logger.debug("Saved single-annotated detection copy: %s", out_path)


def thread_worker(
    img_paths: List[Path],
    model_name: str,
    progress_bar: tqdm,
    results_list: List[DetectionResult],
    lock: threading.Lock,
    inclusion_region: Optional[Rectangle] = None,
    conf_threshold: float = 0.25,
    target_classes: Optional[List[int]] = None,
) -> None:
    """
    Thread worker for parallel YOLO inference.

    Args:
        img_paths (List[Path]): Images assigned to this thread.
        model_name (str): YOLO model weights or path.
        progress_bar (tqdm): Shared progress bar instance.
        results_list (List[DetectionResult]): Shared results accumulator.
        lock (threading.Lock): Thread lock for safe writes.
        inclusion_region (Optional[Rectangle]): Optional spatial filter.
        conf_threshold (float): Detection confidence threshold.
        target_classes (Optional[List[int]]): COCO class filter list.
    """
    thread_results = process_images(
        img_paths=img_paths,
        model_name=model_name,
        progress_bar=progress_bar,
        inclusion_region=inclusion_region,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
    )
    with lock:
        results_list.extend(thread_results)


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
        default="yolo26s.pt",
        help="YOLO model weights to use (default: yolo26s.pt).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads to allocate to YOLO detections (default: 1).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions (default: 0.25).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of COCO category IDs to detect (e.g., '0,2,5,7').",
    )
    parser.add_argument(
        "--inclusion-region",
        type=str,
        default=None,
        help="Inclusion region as 'x,y,w,h' in pixels (default: None).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images.",
    )
    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)
    assert input_folder is not None and output_folder is not None

    output_folder.mkdir(parents=True, exist_ok=True)

    if args.threads <= 0:
        logger.error("The number of allocated CPU threads must be at least 1.")
        sys.exit(1)
    thread_count = args.threads

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    logger.info("Allocating %d CPU thread(s) to YOLO detections...", thread_count)

    target_classes = DEFAULT_TARGET_CLASSES
    if args.classes:
        try:
            target_classes = [int(x.strip()) for x in args.classes.split(",")]
        except ValueError:
            logger.error(
                "Invalid class list format: '%s'. Expected comma-separated integers.",
                args.classes,
            )
            sys.exit(1)

    inclusion_region = None
    if args.inclusion_region:
        try:
            parts = [int(x.strip()) for x in args.inclusion_region.split(",")]
            if len(parts) == 4:
                inclusion_region = Rectangle(parts[0], parts[1], parts[2], parts[3])
            else:
                logger.error(
                    "Invalid inclusion region format: '%s'. Expected 'x,y,w,h'.",
                    args.inclusion_region,
                )
                sys.exit(1)
        except ValueError:
            logger.error(
                "Invalid inclusion region coordinates in '%s'. Expected integers.",
                args.inclusion_region,
            )
            sys.exit(1)

    total_counts: Dict[str, int] = {}
    try:
        model = YOLO(args.model)
        for cls in target_classes:
            if cls in model.names:
                label = model.names[cls]
                if label in ["car", "bus", "truck"]:
                    label = "car"
                total_counts[label] = 0
            else:
                logger.warning("Class ID %d not found in model names.", cls)
    except Exception as e:
        logger.error("Error loading model '%s': %s", args.model, e)
        sys.exit(1)

    all_images = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not all_images:
        logger.warning("No matching images found in the input directory.")
        return

    progress_bar = tqdm(total=len(all_images), desc="Progress", unit="image")

    chunk_size = max(1, len(all_images) // thread_count)
    threads = []
    all_results: List[DetectionResult] = []
    lock = threading.Lock()

    for i in range(thread_count):
        start = i * chunk_size
        end = None if i == thread_count - 1 else (i + 1) * chunk_size
        imgs = all_images[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=thread_worker,
            args=(
                imgs,
                args.model,
                progress_bar,
                all_results,
                lock,
            ),
            kwargs={
                "inclusion_region": inclusion_region,
                "conf_threshold": args.conf,
                "target_classes": target_classes,
            },
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()

    progress_bar.close()

    if not args.no_save:
        logger.info("Saving annotated images to output directory...")
        save_annotated_images(all_results, input_folder, output_folder)
    else:
        logger.info("Skipping saving annotated images as requested.")

    detection_details: Dict[str, List[Dict[str, Union[List[int], float]]]] = {}
    for res in all_results:
        relative_key = str(res.image_path.relative_to(input_folder))
        file_detections = []
        for rect, coco_id, conf in res.boxes:
            label = CLASS_ID_MAPPING[coco_id]
            if label not in total_counts:
                total_counts[label] = 0
            total_counts[label] += 1

            file_detections.append(
                {
                    "box": [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
                    "label": coco_id,
                    "confidence": conf,
                }
            )
        detection_details[relative_key] = file_detections

    print("\n--- Summary ---")
    for category, count in total_counts.items():
        print(f"Total {category} detected: {count}")
    print("YOLO processing complete!\n")

    if args.report:
        logger.info("Generating detection report at %s...", args.report)
        report_data = {
            "metadata": {
                "input_dir": str(input_folder),
                "output_dir": str(output_folder),
                "model_weights": args.model,
                "confidence_threshold": args.conf,
                "target_classes": target_classes,
                "total_images_processed": len(all_images),
                "generated_at": datetime.datetime.now().isoformat(),
            },
            "statistics": total_counts,
            "detections": detection_details,
        }

        try:
            with open(args.report, "w") as f:
                json.dump(report_data, f, indent=4)
        except Exception as e:
            logger.error("Failed to save report to %s: %s", args.report, e)


if __name__ == "__main__":
    main()
