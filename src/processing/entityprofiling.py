import os
import sys
import argparse
from pathlib import Path
import threading
from threading import Thread
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import json

def get_iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw > 0 and ih > 0:
        int_area = iw * ih
        areaA = aw * ah
        areaB = bw * bh
        return int_area / (areaA + areaB - int_area)
    return 0.0

def is_contained_in(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw > 0 and ih > 0:
        int_area = iw * ih
        areaA = aw * ah
        return (int_area / areaA) > 0.85
    return False

def are_overlapping_heavily(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw > 0 and ih > 0:
        int_area = iw * ih
        min_area = min(aw * ah, bw * bh)
        return (int_area / min_area) > 0.7
    return False

def merge_split_boxes(boxes):
    changed = True
    while changed:
        changed = False
        n = len(boxes)
        merged_indices = set()
        new_boxes = []
        for i in range(n):
            if i in merged_indices:
                continue
            for j in range(i + 1, n):
                if j in merged_indices:
                    continue
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[j]
                
                # Case 1: Horizontally aligned and adjacent
                iy1 = max(y1, y2)
                iy2 = min(y1 + h1, y2 + h2)
                v_overlap = max(0, iy2 - iy1)
                
                # Check vertical overlap ratio relative to maximum height to ensure size similarity
                v_ratio = v_overlap / max(h1, h2)
                is_adj_x = abs((x1 + w1) - x2) <= 15 or abs((x2 + w2) - x1) <= 15
                
                # Case 2: Vertically aligned and adjacent
                ix1 = max(x1, x2)
                ix2 = min(x1 + w1, x2 + w2)
                h_overlap = max(0, ix2 - ix1)
                
                # Check horizontal overlap ratio relative to maximum width to ensure size similarity
                h_ratio = h_overlap / max(w1, w2)
                is_adj_y = abs((y1 + h1) - y2) <= 15 or abs((y2 + h2) - y1) <= 15
                
                should_merge = False
                if v_ratio > 0.95 and is_adj_x:
                    should_merge = True
                elif h_ratio > 0.95 and is_adj_y:
                    should_merge = True
                    
                if should_merge:
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nw = max(x1 + w1, x2 + w2) - nx
                    nh = max(y1 + h1, y2 + h2) - ny
                    new_boxes.append((nx, ny, nw, nh))
                    merged_indices.add(i)
                    merged_indices.add(j)
                    changed = True
                    break
            if changed:
                for k in range(n):
                    if k not in merged_indices:
                        new_boxes.append(boxes[k])
                boxes = new_boxes
                break
        if not changed:
            break
    return boxes

def detect_entities(img):
    """
    Detects green bounding boxes inside the image and returns a list of boxes (x, y, w, h).
    Uses a strict green mask to filter out foliage, merges adjacent split parts of boxes, and
    resolves overlapping/intersecting boxes (removing union and redundant intersection boxes).
    """
    # Strict green filter in BGR: B < 50, G > 180, R < 50
    mask = (img[:, :, 0] < 50) & (img[:, :, 1] > 180) & (img[:, :, 2] < 50)
    mask = mask.astype(np.uint8) * 255
    
    # Use RETR_LIST to find all contours, including inner and outer boundaries
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    h_img, w_img = img.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out noise (too small) or image frame borders (too large)
        if 20 < w < w_img * 0.98 and 20 < h < h_img * 0.98:
            raw_boxes.append((x, y, w, h))
            
    # 1. Remove virtually identical duplicate boxes using IoU
    unique_boxes = []
    sorted_by_area = sorted(raw_boxes, key=lambda b: b[2] * b[3], reverse=True)
    for box in sorted_by_area:
        is_dup = False
        for ubox in unique_boxes:
            if get_iou(box, ubox) > 0.8:
                is_dup = True
                break
        if not is_dup:
            unique_boxes.append(box)
            
    # 2. Merge split adjacent parts
    merged_boxes = merge_split_boxes(unique_boxes)
            
    # 3. Containment and Union/Intersection analysis
    n = len(merged_boxes)
    discard = [False] * n
    
    contained_in_count = [0] * n
    contains_count = [0] * n
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if is_contained_in(merged_boxes[i], merged_boxes[j]):
                contained_in_count[i] += 1
                contains_count[j] += 1
                
    for i in range(n):
        # A box is a Union box if it contains 2 or more separate subboxes
        if contains_count[i] >= 2:
            contained_subboxes = [j for j in range(n) if is_contained_in(merged_boxes[j], merged_boxes[i])]
            has_separate_subboxes = False
            for idx_a in range(len(contained_subboxes)):
                for idx_b in range(idx_a + 1, len(contained_subboxes)):
                    ja = contained_subboxes[idx_a]
                    jb = contained_subboxes[idx_b]
                    if not are_overlapping_heavily(merged_boxes[ja], merged_boxes[jb]):
                        has_separate_subboxes = True
                        break
                if has_separate_subboxes:
                    break
            if has_separate_subboxes:
                discard[i] = True  # Union box discarded
                
        # A box is an Intersection box if it is contained in 2 or more separate container boxes
        if contained_in_count[i] >= 2:
            discard[i] = True  # Intersection box discarded
                
    return [merged_boxes[i] for i in range(n) if not discard[i]]

def get_timestamp(img_path: Path) -> float:
    """
    Parses timestamp from filename (format HH-MM-SS.jpg) or falls back to file modification time.
    """
    stem = img_path.stem
    try:
        parts = stem.split('-')
        if len(parts) >= 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            parent_name = img_path.parent.name
            # If parent folder is YYYY-MM-DD
            import datetime
            try:
                date_parts = parent_name.split('-')
                if len(date_parts) == 3 and len(date_parts[0]) == 4:
                    dt = datetime.datetime(int(date_parts[0]), int(date_parts[1]), int(date_parts[2]), h, m, s)
                    return dt.timestamp()
            except:
                pass
            # Seconds from start of day
            return float(h * 3600 + m * 60 + s)
    except Exception:
        pass
    return float(img_path.stat().st_mtime)

def draw_label(img, text, x, y, color=(255, 0, 0), font_scale=0.8, thickness=2):
    """
    Draws a text label with a solid background box above the specified point.
    """
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    # Ensure background box doesn't go above image bounds
    bg_y1 = max(0, y - h - 10)
    bg_y2 = max(baseline, y - 5)
    
    cv2.rectangle(img, (x, bg_y1), (x + w + 10, bg_y2), color, -1)
    cv2.putText(img, text, (x + 5, bg_y2 - baseline + 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

class EntityFeatureExtractor:
    """
    Extracts high-dimensional feature vectors using a pre-trained ResNet-50.
    """
    def __init__(self, use_gpu=False):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        try:
            from torchvision.models import ResNet50_Weights
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except ImportError:
            self.model = models.resnet50(pretrained=True)
            
        # Replace fully connected class head with Identity to extract 2048-dim features
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def extract_features(self, cv2_crop):
        """
        Takes an OpenCV BGR crop, preprocesses it, and runs a forward pass through ResNet-50.
        Returns a normalized 1D float array of features.
        """
        rgb_crop = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_crop)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

class ProfileDatabase:
    """
    A database holding feature profiles of entities to calculate matches and similarities.
    """
    def __init__(self):
        self.profiles = {}  # entity_id -> list of feature vectors
        self.next_id = 1

    def add_feature(self, entity_id, feat):
        if entity_id not in self.profiles:
            self.profiles[entity_id] = []
        self.profiles[entity_id].append(feat)
        # Update next_id to be higher than any existing id
        self.next_id = max(self.next_id, int(entity_id) + 1)

    def get_next_id(self):
        nid = self.next_id
        self.next_id += 1
        return nid

    def predict_id(self, feat, threshold=0.75):
        """
        Finds the registered entity profile with the highest similarity to the feature vector.
        Returns (best_id, best_similarity).
        """
        best_id = None
        best_sim = -1.0
        for pid, feats in self.profiles.items():
            # Match against all feature exemplars stored in the profile and pick the best similarity
            for p_feat in feats:
                sim = float(np.dot(feat, p_feat))
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
        return best_id, best_sim

def process_images_worker(img_paths, input_folder, threshold, progress_bar, results_dict, lock):
    # One extractor per thread to prevent model contention
    extractor = EntityFeatureExtractor(use_gpu=False)
    
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            progress_bar.update(1)
            continue
            
        boxes = detect_entities(img)
        detections = []

        for box in boxes:
            x, y, w, h = box
            
            # Crop entity
            crop = img[y:y+h, x:x+w]
            if crop.size > 0:
                feat = extractor.extract_features(crop)
                ts = get_timestamp(img_path)
                detections.append({
                    'box': box,
                    'feature': feat,
                    'timestamp': ts
                })
        with lock:
            results_dict[img_path] = detections
        progress_bar.update(1)

def save_annotated_images_worker(img_paths, input_folder, output_folder, annotations_dict, progress_bar):
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            annotations = annotations_dict.get(img_path, [])
            for box, entity_id in annotations:
                x, y, w, h = box
                # Draw label
                draw_label(img, f"ID: {entity_id}", x, y)
                
            out_path = output_folder / img_path.relative_to(input_folder)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
        progress_bar.update(1)

def main():
    parser = argparse.ArgumentParser(
        description="Extract features from YOLO-annotated green bounding boxes using ResNet-50 and reidentify entities across frames."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory containing images with green bounding boxes."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory to save processed images."
    )
    parser.add_argument(
        "-c", "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate to processing (default: 1)."
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for reidentification matching (default: 0.75)."
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

    # Find images
    all_images = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ]

    print(f"Allocating {thread_count} CPU core(s) to ResNet-50 feature extraction...")

    if not all_images:
        print("No matching images found in the input directory.")
        return

    # Phase 1: Parallel feature extraction
    results_dict = {}
    lock = threading.Lock()
    progress_bar = tqdm(total=len(all_images), desc="Extracting Features", unit="image")
    
    chunk_size = max(1, len(all_images) // thread_count)
    threads: List[Thread] = []

    for i in range(thread_count):
        start = i * chunk_size
        end = None if i == thread_count - 1 else (i + 1) * chunk_size
        imgs = all_images[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=process_images_worker,
            args=(imgs, input_folder, args.threshold, progress_bar, results_dict, lock)
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()
    progress_bar.close()

    # Phase 2: Chronological Re-identification (Sequential)
    print("Matching and tracking entities across frames...")
    profile_db = ProfileDatabase()
    annotations_dict = {}

    # Sort images by timestamp
    sorted_images = sorted(all_images, key=get_timestamp)

    for img_path in sorted_images:
        detections = results_dict.get(img_path, [])
        annotations = []
        for det in detections:
            box = det['box']
            feat = det['feature']
            
            # Predict match ID
            best_id, best_sim = profile_db.predict_id(feat, threshold=args.threshold)
            
            if best_id is not None and best_sim >= args.threshold:
                entity_id = best_id
            else:
                entity_id = profile_db.get_next_id()
                
            profile_db.add_feature(entity_id, feat)
            annotations.append((box, entity_id))
            
        annotations_dict[img_path] = annotations

    # Phase 3: Parallel annotation drawing & saving
    print("Saving processed images...")
    save_progress = tqdm(total=len(all_images), desc="Saving Images", unit="image")
    save_threads: List[Thread] = []

    for i in range(thread_count):
        start = i * chunk_size
        end = None if i == thread_count - 1 else (i + 1) * chunk_size
        imgs = all_images[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=save_annotated_images_worker,
            args=(imgs, input_folder, output_folder, annotations_dict, save_progress)
        )
        save_threads.append(thread)
        thread.start()

    for t in save_threads:
        t.join()
    save_progress.close()
    
    print(f"Profiling complete. Total unique entities identified: {profile_db.next_id - 1}")
    print(f"Results saved to {output_folder}")

if __name__ == "__main__":
    main()
