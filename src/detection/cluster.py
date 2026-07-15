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

from detection.yolo import (
    Detection,
    CLASS_ID_MAPPING,
    TARGET_CLASSES,
    detect_objects,
    parse_detections,
    load_model,
)


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
    detections: list[Detection]  # Detection structs in the cluster


# Checks whether two boxes are within distance pixels of each other.
def _boxes_close(a: Detection, b: Detection, distance: int):
    ax1, ay1, ax2, ay2 = a.box[0], a.box[1], a.box[2], a.box[3]
    bx1, by1, bx2, by2 = b.box[0], b.box[1], b.box[2], b.box[3]

    ax1e, ay1e = ax1 - distance, ay1 - distance
    ax2e, ay2e = ax2 + distance, ay2 + distance

    noOverlap = bx2 < ax1e or bx1 > ax2e or by2 < ay1e or by1 > ay2e
    return not noOverlap


# Checks whether two boxes are similar enough in SIZE to be considered
# at roughly the same distance from the camera. Uses box area; a person
# far away has a much smaller box area than a person standing close up.
def _similar_size(a: Detection, b: Detection, maxRatio: float):
    aArea = (a.box[2] - a.box[0]) * (a.box[3] - a.box[1])
    bArea = (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])

    if aArea <= 0 or bArea <= 0:
        return False

    ratio = max(aArea, bArea) / min(aArea, bArea)
    return ratio <= maxRatio


def process_clusters(
    detections: list[Detection], maxDistance: int, maxSizeRatio: float
) -> list[Cluster]:
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
        x1 = min(detections[i].box[0] for i in idxs)
        y1 = min(detections[i].box[1] for i in idxs)
        x2 = max(detections[i].box[2] for i in idxs)
        y2 = max(detections[i].box[3] for i in idxs)

        classCounts: dict[int, int] = {}
        for i in idxs:
            cls_id = detections[i].cls_id
            classCounts[cls_id] = classCounts.get(cls_id, 0) + 1

        clusters.append(
            Cluster((x1, y1, x2, y2), classCounts, list(detections[i] for i in idxs))
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


def process_single_image(
    model: YOLO,
    img_path: Path,
    input_folder: Path,
    output_folder: Path,
    save_images: bool = True,
    conf: float = 0.25,
    classes: list[int] = TARGET_CLASSES,
    maxDistance: int = 60,
    maxSizeRatio: float = 2.5,
) -> list[Detection]:
    """
    Processes a single image: runs detection, parses results, and optionally saves output.
    """
    raw_results = detect_objects(model, img_path, conf, classes)
    detections = parse_detections(raw_results, model.names, img_path)
    clusters = process_clusters(detections, maxDistance, maxSizeRatio)

    if save_images:
        save_annotated_results(img_path, clusters, input_folder, output_folder)

    return detections


def process_image_worker(
    img_paths: list[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool,
    conf: float,
    progress_bar: tqdm | None,
    classes: list[int] = TARGET_CLASSES,
    maxDistance: int = 60,
    maxSizeRatio: float = 2.5,
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

        detections = process_single_image(
            model=model,
            img_path=img_path,
            input_folder=input_folder,
            output_folder=output_folder,
            save_images=save_images,
            conf=conf,
            classes=classes,
            maxDistance=maxDistance,
            maxSizeRatio=maxSizeRatio,
        )

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
    max_distance: int = 60,
    max_size_ratio: float = 2.5,
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
        thread = Thread(
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
                max_distance,
                max_size_ratio,
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
        default="yolo26s.pt",
        help="YOLO model weights to use (default: yolo26s.pt).",
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

    batch_detect_and_process(
        img_paths=all_images,
        input_folder=input_folder,
        output_folder=output_folder,
        model_name=args.model,
        save_images=not args.no_save,
        num_threads=thread_count,
        show_progress=True,
        classes=classes_of_interest,
        max_distance=args.max_distance,
        max_size_ratio=args.max_ratio,
    )


if __name__ == "__main__":
    main()
