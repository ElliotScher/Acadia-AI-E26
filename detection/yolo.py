from typing import List
import os

from ultralytics import YOLO
from pathlib import Path
import cv2
import threading
from threading import Thread

# Load model
model = YOLO("yolov8n.pt")

# Folder setup
input_folder = Path("DATADUMP2")
output_folder = Path("output_images")
output_folder.mkdir(exist_ok=True)

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
TARGET_CLASSES = [0, 2, 5, 7]

total_counts = {model.names[cls]: 0 for cls in TARGET_CLASSES}

def process_images(img_paths: List[Path]):
    thread_model = YOLO("yolo26s.pt")

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

        # Save result
        cv2.imwrite(str(output_folder / img_path.name), image)
        print(f"Processed: {img_path.name}")

thread_count = os.cpu_count() or 4
threads: List[Thread] = []

chunk_size = max(1, len(os.listdir(input_folder)) // thread_count)

all_images = [
    p for p in input_folder.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
]

for i in range(thread_count):
    start = i * chunk_size
    end = None if i == thread_count - 1 else (i + 1) * chunk_size
    imgs = all_images[start:end]
    thread = threading.Thread(target=process_images, args=(imgs,))
    threads.append(thread)
    thread.start()
    

for t in threads:
    t.join()
    
print("YIPPEEKIYAY MOTHERFUCKER")
