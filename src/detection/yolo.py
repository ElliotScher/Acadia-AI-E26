from typing import List, Dict, Any, Tuple, Union
import os
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
import cv2
import threading
from threading import Thread
import torch
from ultralytics import YOLO
from tqdm import tqdm

# COCO classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
TARGET_CLASSES: List[int] = [0, 1, 2, 3, 5, 7]

CLASS_MAPPING: Dict[str, str] = {
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "bike",
    "car": "car",
    "bus": "car",
    "truck": "car"
}

CLASS_ID_MAPPING: Dict[int, str] = {
    0: "person",
    1: "bike",
    3: "bike",
    2: "car",
    5: "car",
    7: "car"
}

# Global object counter for CLI summaries and tracking
total_counts: Dict[str, int] = {"car": 0, "person": 0, "bike": 0}
total_counts_lock = threading.Lock()


@dataclass
class Detection:
    """
    Struct representing a detected object.
    Holds the image coordinates of the bounding box, the category/label, the associated image path, and detection confidence.
    """
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image coordinates
    label: str                      # Target category ('car', 'person', or 'bike')
    image_path: Path                # Path to the associated image
    conf: float = 0.0          # Confidence score


def load_model(model_name: str) -> YOLO:
    """
    Loads and returns a YOLO model from the given path or name.
    """
    return YOLO(model_name)


def detect_objects(
    model: YOLO,
    img_path: Union[Path, str],
    conf: float = 0.25,
    classes: List[int] = TARGET_CLASSES
) -> List[Any]:
    """
    Runs YOLO model prediction and returns raw results.
    """
    results = model.predict(
        source=str(img_path),
        conf=conf,
        classes=classes,
        verbose=False
    )
    return results


def map_class(cls_id: int, cls_name: str) -> str:
    """
    Maps class ID or name to target categories: 'car', 'person', 'bike'.
    If no mapping is found, returns the raw cls_name.
    """
    # 1. Try mapping by class ID first
    if cls_id in CLASS_ID_MAPPING:
        return CLASS_ID_MAPPING[cls_id]
    
    # 2. Fall back to mapping by label name (case-insensitive)
    name_lower = cls_name.lower()
    if name_lower in CLASS_MAPPING:
        return CLASS_MAPPING[name_lower]
    if "car" in name_lower or "bus" in name_lower or "truck" in name_lower:
        return "car"
    if "bike" in name_lower or "bicycle" in name_lower or "motorcycle" in name_lower:
        return "bike"
    if "person" in name_lower:
        return "person"
    return cls_name


def parse_detections(results: List[Any], model_names: Dict[int, str], img_path: Path) -> List[Detection]:
    """
    Parses raw YOLO results into a normalized list of Detection objects.
    """
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_label = model_names.get(cls_id, "")
            normalized_label = map_class(cls_id, raw_label)
            
            if normalized_label:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(Detection(
                    box=(x1, y1, x2, y2),
                    label=normalized_label,
                    image_path=img_path,
                    conf=conf
                ))
    return detections


def save_annotated_results(
    img_path: Path,
    detections: List[Detection],
    input_folder: Path,
    output_folder: Path
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

    if not detections:
        cv2.imwrite(str(out_path_base), image)
    else:
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.box
            label = det.label
            image_copy = image.copy()
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)
            
            out_path = out_path_base.with_name(f"{out_path_base.stem}-{i}-{label}{out_path_base.suffix}")
            cv2.imwrite(str(out_path), image_copy)


def process_single_image(
    model: YOLO,
    img_path: Path,
    input_folder: Path,
    output_folder: Path,
    save_images: bool = True,
    conf: float = 0.25,
    classes: List[int] = TARGET_CLASSES
) -> List[Detection]:
    """
    Processes a single image: runs detection, parses results, and optionally saves output.
    """
    raw_results = detect_objects(model, img_path, conf, classes)
    detections = parse_detections(raw_results, model.names, img_path)
    
    if save_images:
        save_annotated_results(img_path, detections, input_folder, output_folder)
        
    return detections


def process_image_worker(
    img_paths: List[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool,
    conf: float,
    progress_bar: tqdm | None,
    classes: List[int] = TARGET_CLASSES
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
            classes=classes
        )

        # Update global counts
        for det in detections:
            label = det.label
            with total_counts_lock:
                total_counts[label] = total_counts.get(label, 0) + 1

        if progress_bar:
            progress_bar.update(1)


def batch_detect_and_process(
    img_paths: List[Path],
    input_folder: Path,
    output_folder: Path,
    model_name: str,
    save_images: bool = True,
    conf: float = 0.25,
    num_threads: int = 1,
    show_progress: bool = True,
    classes: List[int] = TARGET_CLASSES
) -> None:
    """
    Performs multi-threaded batch detection and processing on a list of images.
    """
    progress_bar = None
    if show_progress:
        progress_bar = tqdm(total=len(img_paths), desc="Processing Detections", unit="image")
        
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
            args=(imgs, input_folder, output_folder, model_name, save_images, conf, progress_bar, classes)
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
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights to use (default: yolo26s.pt)."
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate to YOLO detections (default: 1)."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images to the output directory."
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="List of class names or class IDs to detect (e.g. 0 2 or person car). Default: person, bike, car."
    )
    args = parser.parse_args()

    input_folder = Path(args.input_dir).resolve()
    output_folder = Path(args.output_dir).resolve()

    if not input_folder.is_dir():
        print(f"Error: Input directory '{input_folder}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    if args.cores <= 0:
        print("Error: The number of allocated CPU cores must be at least 1.", file=sys.stderr)
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
                    print(f"Warning: Class name '{c}' not found in model classes. Ignoring.", file=sys.stderr)
        
        # Deduplicate
        classes_of_interest = list(dict.fromkeys(classes_of_interest))
        
        if not classes_of_interest:
            print("Error: No valid classes resolved from the provided --classes argument.", file=sys.stderr)
            sys.exit(1)
    else:
        classes_of_interest = TARGET_CLASSES

    print(f"Allocating {thread_count} CPU core(s) to YOLO detections...")

    # Recurse over all subdirectories in the input directory
    all_images = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not all_images:
        print("No matching images found in the input directory.")
        return

    # Reset and initialize total counts for each class of interest
    with total_counts_lock:
        total_counts.clear()
        for cls_id in classes_of_interest:
            raw_label = model.names.get(cls_id, str(cls_id))
            normalized_label = map_class(cls_id, raw_label)
            total_counts[normalized_label] = 0

    batch_detect_and_process(
        img_paths=all_images,
        input_folder=input_folder,
        output_folder=output_folder,
        model_name=args.model,
        save_images=not args.no_save,
        num_threads=thread_count,
        show_progress=True,
        classes=classes_of_interest
    )
    
    print("\n" + "="*30)
    print("      DETECTION SUMMARY")
    print("="*30)
    for label, count in sorted(total_counts.items()):
        if label == "person":
            display_label = "people"
        elif label in ("car", "bike"):
            display_label = f"{label}s"
        else:
            display_label = label
        print(f"Total {display_label}:".ljust(15) + f"{count}")
    print("="*30)


if __name__ == "__main__":
    main()