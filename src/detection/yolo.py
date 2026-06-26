from typing import List
import os
import argparse
import sys
from pathlib import Path
import cv2
import threading
from threading import Thread
import torch
from ultralytics import YOLO
from tqdm import tqdm

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 5, 7]

total_counts = {}
total_counts_lock = threading.Lock()


def process_images(img_paths: List[Path], input_folder: Path, output_folder: Path, model_name: str, progress_bar: tqdm):
    thread_model = YOLO(model_name)

    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            progress_bar.update(1)
            continue

        results = thread_model.predict(
            source=str(img_path),
            conf=0.25,
            classes=TARGET_CLASSES,
            verbose=False
        )

        image = cv2.imread(str(img_path))

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = thread_model.names[cls]
                
                with total_counts_lock:
                    total_counts[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=25)
                cv2.putText(
                    image, 
                    f"{label} {conf:.2f}", 
                    (x1, y1 - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    5,
                    (0, 255, 0), 
                    5
                )

        # Save result, preserving directory structure
        out_path = output_folder / img_path.relative_to(input_folder)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
        progress_bar.update(1)


def main():
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
        default=None,
        help="Number of CPU cores to allocate to YOLO detections (default: use all available cores)."
    )
    args = parser.parse_args()

    input_folder = Path(args.input_dir).resolve()
    output_folder = Path(args.output_dir).resolve()

    if not input_folder.is_dir():
        print(f"Error: Input directory '{input_folder}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

    # Determine CPU cores allocation
    max_cores = os.cpu_count() or 4
    if args.cores is not None:
        if args.cores <= 0:
            print("Error: The number of allocated CPU cores must be at least 1.", file=sys.stderr)
            sys.exit(1)
        thread_count = args.cores
    else:
        thread_count = max_cores

    # Configure PyTorch CPU thread count to respect our core allocation globally
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    print(f"Allocating {thread_count} CPU core(s) to YOLO detections...")

    # Initialize total_counts with class names from the selected model
    try:
        model = YOLO(args.model)
        for cls in TARGET_CLASSES:
            total_counts[model.names[cls]] = 0
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # Recurse over all subdirectories in the input directory
    all_images = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    if not all_images:
        print("No matching images found in the input directory.")
        return

    progress_bar = tqdm(total=len(all_images), desc="Progress", unit="image")

    chunk_size = max(1, len(all_images) // thread_count)
    threads: List[Thread] = []

    for i in range(thread_count):
        start = i * chunk_size
        end = None if i == thread_count - 1 else (i + 1) * chunk_size
        imgs = all_images[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=process_images,
            args=(imgs, input_folder, output_folder, args.model, progress_bar)
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()
        
    progress_bar.close()
    print("YIPPEEKIYAY MOTHERFUCKER")


if __name__ == "__main__":
    main()

