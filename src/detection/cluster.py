from pathlib import Path
import cv2
from ultralytics import YOLO
import argparse
import sys
from dataclasses import dataclass
from tqdm import tqdm
import threading
from threading import Thread
import torch
import logging
from typing import Optional, List

from detection.classes import CLASS_ID_MAPPING, TARGET_CLASSES
from detection.image_yolo import (
    DetectionResult,
    load_model,
)
from utility.geometryutils import Rectangle
from utility.parallel import ProgressTracker

# Initialize Logger
logger = logging.getLogger("cluster")

@dataclass
class Cluster:
    """
    Struct representing a cluster detected in an image.
    Holds the image coordinates of the bounding box, the number of contained
    instances of each COCO class, and a list of Detection structs
    contained in the cluster.
    """

    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates,
    counts: dict[int, int]  # number of instances of each COCO class
    detections: DetectionResult  # DetectionResults in the cluster

Detection = tuple[Rectangle, int, float]

# Checks whether two boxes are within distance pixels of each other.
def _boxes_close(a: Detection, b: Detection, distance: int):
    ax1, ay1, ax2, ay2 = a[0].x, a[0].y, a[0].x + a[0].w, a[0].y + a[0].y
    bx1, by1, bx2, by2 = b[0].x, b[0].y, b[0].x + b[0].w, b[0].y + b[0].y

    ax1e, ay1e = ax1 - distance, ay1 - distance
    ax2e, ay2e = ax2 + distance, ay2 + distance

    noOverlap = bx2 < ax1e or bx1 > ax2e or by2 < ay1e or by1 > ay2e
    return not noOverlap


# Checks whether two boxes are similar enough in SIZE to be considered
# at roughly the same distance from the camera. Uses box area; a person
# far away has a much smaller box area than a person standing close up.
def _similar_size(a: Detection, b: Detection, maxRatio: float):
    aArea = a[0].w * a[0].h
    bArea = b[0].w * b[0].h

    if aArea <= 0 or bArea <= 0:
        return False

    ratio = max(aArea, bArea) / min(aArea, bArea)
    return ratio <= maxRatio


def process_clusters(
    detectionResults: DetectionResult, maxDistance: int, maxSizeRatio: float
) -> list[Cluster]:
    detections = detectionResults.boxes
    n: int = len(detections)
    parent: list[int] = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # boxes only join the same cluster if they're both close together
    # AND roughly the same size -- proximity alone isn't enough
    for i in range(n):
        for j in range(i + 1, n):
            if not _boxes_close(detections[i], detections[j], maxDistance):
                continue
            if not _similar_size(detections[i], detections[j], maxSizeRatio):
                continue
            union(i, j)

    group: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        group.setdefault(root, []).append(i)

    clusters: list[Cluster] = []
    for idxs in group.values():
        x1 = min(detections[i][0].x for i in idxs)
        y1 = min(detections[i][0].y for i in idxs)
        x2 = max((detections[i][0].x + detections[i][0].w) for i in idxs)
        y2 = max((detections[i][0].y + detections[i][0].h) for i in idxs)

        classCounts: dict[int, int] = {}
        for i in idxs:
            cls_id = detections[i][1]
            classCounts[cls_id] = classCounts.get(cls_id, 0) + 1

        clusters.append(
            Cluster((x1, y1, x2, y2), classCounts, DetectionResult(
                detectionResults.image_path,
                list(detections[i] for i in idxs)
            ))
        )
    return clusters


def save_annotated_results(
    img_path: Path, clusters: list[Cluster], input_folder: Path, output_folder: Path
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

    if len(clusters) == 0:
        cv2.imwrite(str(out_path_base), image)
    else:
        for i, cluster in enumerate(clusters):
            x1, y1, x2, y2 = cluster.box
            image_copy = image.copy()
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
            text = ", ".join(
                f"{CLASS_ID_MAPPING[cls_id]} x{count}"
                for cls_id, count in cluster.counts.items()
            )
            cv2.putText(
                image_copy,
                text,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            filename_text = "-".join(
                f"{CLASS_ID_MAPPING[cls_id]}-x{count}"
                for cls_id, count in cluster.counts.items()
            )
            out_path = out_path_base.with_name(
                f"{out_path_base.stem}-{i}-{filename_text}{out_path_base.suffix}"
            )
            cv2.imwrite(str(out_path), image_copy)

def process_images(
    img_paths: list[Path],
    model_name: str,
    progress_bar: Optional[tqdm | ProgressTracker] = None,
    inclusion_region: Optional[Rectangle] = None,
    conf_threshold: float = 0.25,
    target_classes: Optional[List[int]] = None,
    max_distance: int = 60,
    max_size_ratio: float = 2.5,
) -> list[Cluster]:
    """
    Processes a list of image paths using YOLO and extracts detection boxes and images.
    Does not write anything to disk.

    Args:
        img_paths (list[Path]): List of image paths to process.
        model_name (str): YOLO weights name or path.
        progress_bar (Optional[tqdm | ProgressTracker]): Thread-safe progress bar instance.
        inclusion_region (Optional[Rectangle]): Optional spatial filter region.
        conf_threshold (float): Minimum confidence threshold for detections.
        target_classes (Optional[list[int]]): List of COCO class IDs to filter.

    Returns:
        list[DetectionResult]: List of detection results per image.
    """
    classes_list = (
        target_classes if target_classes is not None else TARGET_CLASSES
    )
    results_list: list[Cluster] = []

    try:
        thread_model = load_model(model_name)
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

            boxes_found: list[tuple[Rectangle, int, float]] = []
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

            results_list.extend(
                process_clusters(
                    DetectionResult(image_path=img_path, boxes=boxes_found),
                    max_distance,
                    max_size_ratio
                )
            )

        except Exception as e:
            logger.error("Error processing image %s: %s", img_path, e)
        finally:
            if progress_bar:
                progress_bar.update(1)

    return results_list


def save_annotated_images(
    results: list[Cluster],
    input_folder: Path,
    output_folder: Path,
) -> None:
    """
    Saves cluster results to disk with annotations.

    Args:
        results (list[Cluster]): YOLO cluster outputs.
        input_folder (Path): Source directory for relative path resolution.
        output_folder (Path): Destination directory for saved images.
    """
    paths: list[Path] = []

    for res in results:
        img_path = res.detections.image_path
        paths.append(img_path)
        image = cv2.imread(str(res.detections.image_path))

        out_path_base = output_folder / img_path.relative_to(input_folder)
        out_path_base.parent.mkdir(parents=True, exist_ok=True)

        if image is not None:
            image_copy = image.copy()
            cv2.rectangle(
                image_copy,
                (res.box[0], res.box[1]),
                (res.box[2], res.box[3]),
                (0, 255, 0),
                thickness=5,
            )
            for i, (rect, label, conf) in enumerate(res.detections.boxes):
                cv2.rectangle(
                    image_copy,
                    (rect.x, rect.y),
                    (rect.x + rect.w, rect.y + rect.h),
                    (0, 0, 255),
                    thickness=3,
                )
            out_path = out_path_base.with_name(
                f"{out_path_base.stem}-{len(res.detections.boxes)}x-{paths.count(img_path)}{out_path_base.suffix}"
            )
            cv2.imwrite(str(out_path), image_copy)
            logger.debug("Saved single-annotated detection copy: %s", out_path)


def thread_worker(
    img_paths: list[Path],
    model_name: str,
    progress_bar: Optional[tqdm | ProgressTracker],
    results_list: list[Cluster],
    lock: threading.Lock,
    inclusion_region: Optional[Rectangle] = None,
    conf_threshold: float = 0.25,
    target_classes: Optional[list[int]] = None,
    max_distance: int = 60,
    max_size_ratio: float = 2.5,
) -> None:
    """
    Thread worker for parallel YOLO inference.

    Args:
        img_paths (list[Path]): Images assigned to this thread.
        model_name (str): YOLO model weights or path.
        progress_bar (Optional[tqdm | ProgressTracker]): Shared progress bar instance.
        results_list (list[DetectionResult]): Shared results accumulator.
        lock (threading.Lock): Thread lock for safe writes.
        inclusion_region (Optional[Rectangle]): Optional spatial filter.
        conf_threshold (float): Detection confidence threshold.
        target_classes (Optional[list[int]]): COCO class filter list.
    """
    thread_results = process_images(
        img_paths=img_paths,
        model_name=model_name,
        progress_bar=progress_bar,
        inclusion_region=inclusion_region,
        conf_threshold=conf_threshold,
        target_classes=target_classes,
        max_distance=max_distance,
        max_size_ratio=max_size_ratio
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
        "--no-save",
        action="store_true",
        help="Do not save annotated images to the output directory.",
    )
    parser.add_argument(
        "--inclusion-region",
        type=str,
        default=None,
        help="Inclusion region as 'x,y,w,h' in pixels (default: None).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=60,
        help="Maximum distance in pixels between two instances to consider them part of the same cluster.",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=2.5,
        help="Maximum ratio between the sizes of two instances to consider them part of the same cluster.",
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

    from utility.loggingutils import setup_logging_and_paths

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

    target_classes = TARGET_CLASSES
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

    total_counts: dict[str, int] = {}
    try:
        model = load_model(args.model)
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
    all_results: list[Cluster] = []
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
                "max_distance": args.max_distance,
                "max_size_ratio": args.max_ratio
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


if __name__ == "__main__":
    main()
