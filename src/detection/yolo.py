from typing import List
import os
import argparse
import sys

from ultralytics import YOLO
from pathlib import Path
import cv2
import threading
from threading import Thread

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 5, 7]

total_counts = {}
total_counts_lock = threading.Lock()


def process_images(img_paths: List[Path], input_folder: Path, output_folder: Path, model_name: str):
    thread_model = YOLO(model_name)

    for img_path in img_paths:
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
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
        print(f"Processed: {img_path.relative_to(input_folder)}")


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
    args = parser.parse_args()

    input_folder = Path(args.input_dir).resolve()
    output_folder = Path(args.output_dir).resolve()

    if not input_folder.is_dir():
        print(f"Error: Input directory '{input_folder}' does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_folder.mkdir(parents=True, exist_ok=True)

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

    thread_count = os.cpu_count() or 4
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
            args=(imgs, input_folder, output_folder, args.model)
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()
        
    print("YIPPEEKIYAY MOTHERFUCKER")


if __name__ == "__main__":
    main()
