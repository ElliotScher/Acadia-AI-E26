"""
Image Entity Profiler

A functional backend library and command-line tool to extract features from green bounding boxes
(detected inside images) using a pre-trained ResNet-50 network and re-identify/track
entities chronologically across images.
"""

import argparse
from dataclasses import dataclass
import datetime
import json
import logging
import sys
import threading
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from utility import imgutils


class Direction(Enum):
    LEFT_RIGHT = auto()
    RIGHT_LEFT = auto()
    UNKNOWN = auto()


class ProfilingMode(Enum):
    DWELL = auto()
    OCCUPANCY = auto()

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from src.utility.imgutils import (
    detect_entities,
    get_center_crop,
    get_hsv_hist,
    get_timestamp,
)

# Initialize Logger
logger = logging.getLogger("image_entityprofiler")


@dataclass
class ProfileRecord:
    """
    Represents a single registered feature record for an entity.

    Args:
        entity_id (int): The unique integer ID of the entity.
        feature (np.ndarray): 1D deep feature representation vector.
        hsv_hist (Optional[np.ndarray]): 3D HSV color histogram of the entity crop. Defaults to None.
        aspect_ratio (Optional[float]): Bounding box width / height ratio. Defaults to None.
        timestamp (Optional[float]): UNIX timestamp of the image capture. Defaults to None.
        img_name (Optional[str]): Source filename. Defaults to None.
        box (Optional[List[int]]): Bounding box coordinates [x, y, w, h]. Defaults to None.
    """

    entity_id: int
    feature: np.ndarray
    hsv_hist: Optional[np.ndarray] = None
    aspect_ratio: Optional[float] = None
    timestamp: Optional[float] = None
    img_name: Optional[str] = None
    box: Optional[List[int]] = None


def load_feature_extractor(
    checkpoint_path: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, torch.device, Any]:
    """
    Initializes and returns the ResNet-50 model, the execution device, and the preprocessing transform.

    Attempts to load a custom fine-tuned checkpoint if provided, adapting its state dictionary
    to the ResNet backbone format, and replaces the classification head (fc/classifier) with Identity
    to output raw feature vectors.

    Args:
        checkpoint_path (Optional[Union[str, Path]]): Optional file path to a custom fine-tuned weights file (.pth). Defaults to None.

    Returns:
        Tuple[nn.Module, torch.device, Any]: A tuple containing:
            - nn.Module: The configured ResNet-50 model ready for evaluation.
            - torch.device: The active PyTorch device (cpu) used for computation.
            - Any: The torchvision transforms pipeline for image preprocessing.

    Raises:
        RuntimeError: If default ResNet-50 weights failed to initialize and no custom checkpoint is provided.
    """
    device = torch.device("cpu")

    logger.debug("Feature extractor initialized on device: %s", device)
    model = None

    path_obj = Path(checkpoint_path) if checkpoint_path is not None else None

    if path_obj is not None and path_obj.exists():
        try:
            # Try loading the checkpoint as a serialized nn.Module directly (requires weights_only=False)
            loaded = torch.load(path_obj, map_location=device, weights_only=False)
            if isinstance(loaded, nn.Module):
                model = loaded
                logger.info("Loaded custom base model directly from %s", path_obj)
            elif (
                isinstance(loaded, dict)
                and "model" in loaded
                and isinstance(loaded["model"], nn.Module)
            ):
                model = loaded["model"]
                logger.info(
                    "Loaded custom base model from checkpoint 'model' field in %s",
                    path_obj,
                )
        except Exception as e:
            logger.debug(
                "Could not load %s as a serialized nn.Module (%s). Trying as state dict...",
                path_obj,
                e,
            )

    # If not loaded as a full model, initialize ResNet-50 as default and load state dict
    if model is None:
        try:
            from torchvision.models import ResNet50_Weights

            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as e:
            logger.warning(
                "Could not load default ResNet-50 weights from torchvision. Trying legacy pretrained flag. Reason: %s",
                e,
            )
            try:
                model = models.resnet50(pretrained=True)
            except Exception as e2:
                raise RuntimeError(
                    "Failed to initialize ResNet-50 with pre-trained weights. "
                    "This could be due to a lack of internet connectivity to download the weights. "
                    "Please ensure internet access is available, or provide a custom model checkpoint path via the 'checkpoint' parameter. "
                    f"Details: {e2}"
                ) from e2

        if path_obj is not None and path_obj.exists():
            try:
                # Load the state dict
                try:
                    state_dict = torch.load(
                        path_obj, map_location="cpu", weights_only=True
                    )
                except (RuntimeError, ValueError, AttributeError):
                    state_dict = torch.load(
                        path_obj, map_location="cpu", weights_only=False
                    )

                if isinstance(state_dict, dict):
                    # Extract state dict if it is wrapped
                    for key in ["state_dict", "model_state_dict", "model"]:
                        if key in state_dict and isinstance(state_dict[key], dict):
                            state_dict = state_dict[key]
                            break

                    # Adapt keys from the training model format (ReIDModel with self.backbone)
                    adapted_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("backbone."):
                            new_key = k[len("backbone.") :]
                            if not new_key.startswith("fc."):
                                adapted_state_dict[new_key] = v
                        elif not k.startswith("fc."):
                            adapted_state_dict[k] = v

                    model.load_state_dict(adapted_state_dict, strict=False)
                    logger.info("Loaded custom fine-tuned weights from %s", path_obj)
                else:
                    logger.warning(
                        "Checkpoint at %s did not contain a state_dict or model object.",
                        path_obj,
                    )
            except Exception as e:
                logger.error("Error loading checkpoint state dict: %s", e)

    # Replace fully connected class head with Identity to extract feature vectors
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        try:
            model.classifier = nn.Identity()
        except Exception as e:
            logger.debug("Failed to set model.classifier to Identity: %s", e)

    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return model, device, transform


@torch.no_grad()
def extract_features(
    cv2_crop: np.ndarray,
    model: nn.Module,
    device: torch.device,
    transform: Any,
) -> np.ndarray:
    """
    Preprocesses a BGR image crop, passes it through the ResNet backbone,
    and returns a normalized 1D float array of features.

    Args:
        cv2_crop (np.ndarray): OpenCV BGR crop representing the detected entity.
        model (nn.Module): Loaded feature extractor model.
        device (torch.device): Active PyTorch device.
        transform (Any): Image preprocessing pipeline.

    Returns:
        np.ndarray: A normalized 1D numpy array representing the feature embedding (cosine normalized).
    """
    rgb_crop = cv2.cvtColor(cv2_crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_crop)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    feat = model(tensor).squeeze(0).cpu().numpy()
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


def compute_similarities(
    feat: np.ndarray,
    hsv_hist: Optional[np.ndarray],
    aspect_ratio: Optional[float],
    timestamp: Optional[float],
    img_name: Optional[str],
    database: List[ProfileRecord],
) -> Dict[int, float]:
    """
    Computes the maximum similarity score for each registered entity profile in the database list.

    Combines cosine similarity of deep features, HSV color histogram similarity (Bhattacharyya distance),
    and aspect ratio similarity. Applies a spatial-temporal constraint where matching entities in the same
    image or within a microscopic timestamp range is penalized (similarity set to -1.0).

    Args:
        feat (np.ndarray): 1D normalized deep feature array of the target entity detection.
        hsv_hist (Optional[np.ndarray]): Optional normalized HSV color histogram of the target entity.
        aspect_ratio (Optional[float]): Optional aspect ratio of the target entity bounding box.
        timestamp (Optional[float]): Optional UNIX timestamp of the target entity detection.
        img_name (Optional[str]): Optional filename of the image containing the target entity detection.
        database (List[ProfileRecord]): List of already registered ProfileRecord objects representing historical detections.

    Returns:
        Dict[int, float]: A dictionary mapping each entity_id to its maximum computed similarity score.
    """
    similarities = {}
    grouped: Dict[int, List[ProfileRecord]] = {}
    for r in database:
        if r.entity_id not in grouped:
            grouped[r.entity_id] = []
        grouped[r.entity_id].append(r)

    for ent_id, exemplars in grouped.items():
        max_sim = -1.0
        for p in exemplars:
            # Cosine similarity of deep features
            sim = float(np.dot(feat, p.feature))

            # HSV Color Histogram similarity
            if hsv_hist is not None and p.hsv_hist is not None:
                h1 = (
                    hsv_hist.astype(np.float32)
                    if hsv_hist.dtype != np.float32
                    else hsv_hist
                )
                h2 = np.array(p.hsv_hist, dtype=np.float32)

                dist = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
                color_sim = 1.0 - dist
                sim = sim * 0.7 + color_sim * 0.3

            # Aspect Ratio Similarity
            if aspect_ratio is not None and isinstance(p.aspect_ratio, (int, float)):
                ar_diff = abs(aspect_ratio - p.aspect_ratio)
                ar_sim = float(np.exp(-ar_diff * 4.0))
                sim = sim * (0.8 + 0.2 * ar_sim)

            # Spatial-Temporal Constraint
            if img_name is not None and p.img_name is not None:
                if img_name == p.img_name:
                    sim = -1.0
            elif timestamp is not None and isinstance(p.timestamp, (int, float)):
                time_diff = abs(timestamp - p.timestamp)
                if time_diff < 0.1:
                    sim = -1.0

            if sim > max_sim:
                max_sim = sim
        similarities[ent_id] = max_sim
    return similarities


def assign_entity_id(
    feat: np.ndarray,
    hsv_hist: Optional[np.ndarray],
    aspect_ratio: Optional[float],
    timestamp: Optional[float],
    img_name: Optional[str],
    database: List[ProfileRecord],
    next_id: int,
    threshold: float = 0.75,
) -> Tuple[int, float, bool]:
    """
    Finds the best matching entity ID or returns a brand new one if no match meets the threshold.

    Args:
        feat (np.ndarray): 1D normalized deep feature array.
        hsv_hist (Optional[np.ndarray]): Optional HSV color histogram.
        aspect_ratio (Optional[float]): Optional aspect ratio.
        timestamp (Optional[float]): Optional UNIX timestamp.
        img_name (Optional[str]): Optional image filename.
        database (List[ProfileRecord]): List of registered ProfileRecord objects.
        next_id (int): The next available integer entity ID.
        threshold (float): Minimum similarity threshold for a match. Defaults to 0.75.

    Returns:
        Tuple[int, float, bool]: A tuple containing:
            - int: The assigned entity ID (either a matched historical ID or `next_id`).
            - float: The maximum similarity score computed against historical exemplars.
            - bool: True if this is a newly created entity, False if it was successfully matched.
    """
    similarities = compute_similarities(
        feat, hsv_hist, aspect_ratio, timestamp, img_name, database
    )
    best_id = None
    best_sim = -1.0
    for ent_id, sim in similarities.items():
        if sim > best_sim:
            best_sim = sim
            best_id = ent_id

    if best_id is not None and best_sim >= threshold:
        return best_id, best_sim, False
    return next_id, best_sim, True


def save_database_to_json(
    database: List[ProfileRecord], next_id: int, filepath: Union[str, Path]
) -> None:
    """
    Serializes and saves the database list of profile records to a JSON file.

    Args:
        database (List[ProfileRecord]): List of ProfileRecord objects to save.
        next_id (int): The next available unique entity ID.
        filepath (Union[str, Path]): Target output file path where the JSON will be written.
    """
    serialized_profiles = {}
    for r in database:
        ent_id_str = str(r.entity_id)
        if ent_id_str not in serialized_profiles:
            serialized_profiles[ent_id_str] = []

        feat_list = r.feature.tolist()
        hsv_list = r.hsv_hist.tolist() if r.hsv_hist is not None else None

        serialized_profiles[ent_id_str].append(
            {
                "feature": feat_list,
                "hsv_hist": hsv_list,
                "aspect_ratio": r.aspect_ratio,
                "timestamp": r.timestamp,
                "img_name": r.img_name,
                "box": r.box,
            }
        )

    data = {"profiles": serialized_profiles, "next_id": next_id}
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved database to %s", filepath)


def process_images_worker(
    img_paths: List[Path],
    progress_bar: tqdm,
    results_dict: Dict[Path, List[Dict[str, Any]]],
    lock: threading.Lock,
    model: nn.Module,
    device: torch.device,
    transform: Any,
) -> None:
    """
    Worker function executed in parallel threads to perform detection and feature extraction on a chunk of images.

    For each image in `img_paths`, it reads the image, detects bounding boxes, extracts deep features and
    HSV histograms for each crop, and stores the list of detections in `results_dict` in a thread-safe manner.

    Args:
        img_paths (List[Path]): List of image file paths allocated to this worker thread.
        progress_bar (tqdm): Shared tqdm progress bar instance to update after processing each image.
        results_dict (Dict[Path, List[Dict[str, Any]]]): Shared dictionary where detections are stored, keyed by image path.
        lock (threading.Lock): Shared lock to synchronize writes to `results_dict`.
        model (nn.Module): Feature extractor model instance shared across threads.
        device (torch.device): The device (CPU/GPU) on which the model is executed.
        transform (Any): Image preprocessing transforms pipeline.
    """
    for img_path in img_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                progress_bar.update(1)
                continue

            boxes = detect_entities(img)
            detections = []

            for box in boxes:
                x, y, w, h = box
                crop = img[y : y + h, x : x + w]
                if crop.size > 0:
                    feat = extract_features(crop, model, device, transform)
                    ts = get_timestamp(img_path)
                    hsv_hist = get_hsv_hist(get_center_crop(crop, 0.12))
                    aspect_ratio = float(w) / h
                    detections.append(
                        {
                            "box": box,
                            "feature": feat,
                            "timestamp": ts,
                            "hsv_hist": hsv_hist,
                            "aspect_ratio": aspect_ratio,
                        }
                    )
            with lock:
                results_dict[img_path] = detections
        except Exception as e:
            logger.error("Error processing image %s: %s", img_path, e)
        finally:
            progress_bar.update(1)




def determine_image_direction(img_path: Union[str, Path]) -> Direction:
    """
    Placeholder function to determine if a vehicle in the input image is going left or right.
    Currently returns Direction.UNKNOWN as a placeholder stub.
    """
    # TODO: Implement actual image-level direction classification/flow analysis model
    return Direction.UNKNOWN



def track_entities_in_directory(
    results_dict: Dict[Path, List[Dict[str, Any]]],
    image_paths: List[Path],
    threshold: float = 0.75,
    max_gap: Optional[float] = None,
) -> Tuple[List[ProfileRecord], Dict[int, List[ProfileRecord]]]:
    """
    Sequentially processes images chronologically, matches crops to existing entities,
    and returns the database and grouped profile records.
    """

    sorted_paths = sorted(image_paths, key=imgutils.get_timestamp)
    database: List[ProfileRecord] = []
    next_id = 1

    for img_path in sorted_paths:
        detections = results_dict.get(img_path, [])
        for det in detections:
            feat = det.get("feature")
            if not isinstance(feat, np.ndarray):
                continue
            box = det.get("box")
            hsv_hist = det.get("hsv_hist")
            aspect_ratio = det.get("aspect_ratio")
            ts = det.get("timestamp")
            if ts is None:
                ts = imgutils.get_timestamp(img_path)

            # Filter database by max_gap if specified
            if max_gap is not None and ts is not None:
                valid_db = [
                    r for r in database 
                    if r.timestamp is not None and abs(ts - r.timestamp) <= max_gap
                ]
            else:
                valid_db = database

            entity_id, sim, is_new = assign_entity_id(
                feat=feat,
                hsv_hist=hsv_hist,
                aspect_ratio=aspect_ratio,
                timestamp=ts,
                img_name=img_path.name,
                database=valid_db,
                next_id=next_id,
                threshold=threshold,
            )

            if is_new:
                next_id += 1

            rec = ProfileRecord(
                entity_id=entity_id,
                feature=feat,
                hsv_hist=hsv_hist,
                aspect_ratio=aspect_ratio,
                timestamp=ts,
                img_name=img_path.name,
                box=box,
            )
            database.append(rec)

    grouped: Dict[int, List[ProfileRecord]] = {}
    for r in database:
        if r.entity_id not in grouped:
            grouped[r.entity_id] = []
        grouped[r.entity_id].append(r)

    return database, grouped


def match_entry_exit_entities(
    entry_entities: Dict[int, List[ProfileRecord]],
    exit_entities: Dict[int, List[ProfileRecord]],
    threshold: float = 0.75,
) -> List[Dict[str, Any]]:
    """
    Matches entry entities to exit entities using deep feature and color/aspect ratio similarity.
    Calculates dwell time as the difference between exit and entry timestamps.

    Args:
        entry_entities (Dict[int, List[ProfileRecord]]): Entry entities grouped by entity ID.
        exit_entities (Dict[int, List[ProfileRecord]]): Exit entities grouped by entity ID.
        threshold (float): Minimum similarity threshold for matching. Defaults to 0.75.

    Returns:
        List[Dict[str, Any]]: List of matching results containing entry/exit IDs,
            similarity, and dwell time.
    """
    matches = []
    matched_entry_ids = set()

    sorted_exit_ids = sorted(
        exit_entities.keys(),
        key=lambda eid: max(r.timestamp for r in exit_entities[eid] if r.timestamp is not None)
        if any(r.timestamp is not None for r in exit_entities[eid]) else 0.0
    )

    for exit_id in sorted_exit_ids:
        exit_records = exit_entities[exit_id]
        exit_ts_list = [r.timestamp for r in exit_records if r.timestamp is not None]
        if not exit_ts_list:
            continue
        exit_ts = max(exit_ts_list)

        best_entry_id = None
        best_sim = -1.0

        for entry_id, entry_records in entry_entities.items():
            if entry_id in matched_entry_ids:
                continue

            entry_ts_list = [r.timestamp for r in entry_records if r.timestamp is not None]
            if not entry_ts_list:
                continue
            entry_ts = min(entry_ts_list)

            # Exit must occur after entry
            if exit_ts < entry_ts:
                continue

            # Compute max pair similarity
            max_pair_sim = -1.0
            for exit_rec in exit_records:
                sims = compute_similarities(
                    feat=exit_rec.feature,
                    hsv_hist=exit_rec.hsv_hist,
                    aspect_ratio=exit_rec.aspect_ratio,
                    timestamp=exit_rec.timestamp,
                    img_name=exit_rec.img_name,
                    database=entry_records,
                )
                for sim in sims.values():
                    if sim > max_pair_sim:
                        max_pair_sim = sim

            if max_pair_sim > best_sim:
                best_sim = max_pair_sim
                best_entry_id = entry_id

        if best_entry_id is not None and best_sim >= threshold:
            matched_entry_ids.add(best_entry_id)
            entry_records = entry_entities[best_entry_id]
            entry_ts = min(r.timestamp for r in entry_records if r.timestamp is not None)
            
            matches.append({
                "entry_id": best_entry_id,
                "exit_id": exit_id,
                "similarity": best_sim,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "dwell_time": exit_ts - entry_ts
            })

    return matches


def calculate_occupancy_timeline(
    entry_entities: Dict[int, List[ProfileRecord]],
    exit_entities: Dict[int, List[ProfileRecord]],
) -> List[Dict[str, Any]]:
    """
    Calculates running occupancy events.
    Adapts initial occupancy to avoid negative values.

    Args:
        entry_entities (Dict[int, List[ProfileRecord]]): Entry entities grouped by ID.
        exit_entities (Dict[int, List[ProfileRecord]]): Exit entities grouped by ID.

    Returns:
        List[Dict[str, Any]]: Chronologically sorted list of timeline states representing
            the running occupancy counts.
    """
    events = []
    for ent_id, records in entry_entities.items():
        ts_list = [r.timestamp for r in records if r.timestamp is not None]
        if ts_list:
            events.append((min(ts_list), 1, f"entry_{ent_id}"))

    for ent_id, records in exit_entities.items():
        ts_list = [r.timestamp for r in records if r.timestamp is not None]
        if ts_list:
            events.append((max(ts_list), -1, f"exit_{ent_id}"))

    events.sort(key=lambda x: x[0])

    running = 0
    min_running = 0
    raw_timeline = []
    
    for ts, change, label in events:
        running += change
        if running < min_running:
            min_running = running
        raw_timeline.append({"timestamp": ts, "change": change, "label": label})

    initial_offset = -min_running if min_running < 0 else 0
    
    current_occupancy = initial_offset
    timeline = []
    for event in raw_timeline:
        current_occupancy += event["change"]
        timeline.append({
            "timestamp": event["timestamp"],
            "occupancy": current_occupancy,
            "label": event["label"]
        })

    return timeline


def extract_features_for_directory(
    image_paths: List[Path],
    model: nn.Module,
    device: torch.device,
    transform: Any,
    cores: int = 1,
) -> Dict[Path, List[Dict[str, Any]]]:
    """
    Helper function to run parallel feature extraction for a list of image paths.

    Args:
        image_paths (List[Path]): List of all image paths to process.
        model (nn.Module): Feature extractor model instance.
        device (torch.device): PyTorch device to execute on.
        transform (Any): Image pre-processing transform pipeline.
        cores (int): Number of parallel CPU worker threads to spawn. Defaults to 1.

    Returns:
        Dict[Path, List[Dict[str, Any]]]: A dictionary of detected crop features, aspect ratios,
            and timestamps, keyed by image path.
    """
    results_dict: Dict[Path, List[Dict[str, Any]]] = {}
    lock = threading.Lock()
    progress_bar = tqdm(total=len(image_paths), desc="Extracting Features", unit="image")

    chunk_size = max(1, len(image_paths) // cores)
    threads = []

    for i in range(cores):
        start = i * chunk_size
        end = None if i == cores - 1 else (i + 1) * chunk_size
        imgs = image_paths[start:end]
        if not imgs:
            continue
        thread = threading.Thread(
            target=process_images_worker,
            args=(
                imgs,
                progress_bar,
                results_dict,
                lock,
                model,
                device,
                transform,
            ),
        )
        threads.append(thread)
        thread.start()

    for t in threads:
        t.join()
    progress_bar.close()
    return results_dict


def annotate_and_save_images(
    image_entity_mappings: Dict[Path, List[Tuple[List[int], int]]],
    input_folder: Path,
    output_folder: Path,
) -> None:
    """
    Draws green bounding boxes and entity IDs on images and saves them,
    preserving the folder structure.

    Args:
        image_entity_mappings (Dict[Path, List[Tuple[List[int], int]]]): Mapping from image path
            to list of (bounding box, entity ID) annotations.
        input_folder (Path): Base input directory path.
        output_folder (Path): Base output directory path.
    """
    for img_path, mappings in image_entity_mappings.items():
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for box, entity_id in mappings:
            x, y, bx_w, bx_h = box

            # Draw green bounding box
            cv2.rectangle(img, (x, y), (x + bx_w, y + bx_h), (0, 255, 0), thickness=3)

            # Draw contrast-friendly ID annotation box
            label_text = f"ID: {entity_id}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2
            )

            # Position label on top of box, fallback if top margin is too tight
            label_y = max(y - 10, text_h + baseline + 5)
            label_x = max(x, 5)

            # Draw black background rectangle for high contrast
            cv2.rectangle(
                img,
                (label_x - 3, label_y - text_h - baseline - 3),
                (label_x + text_w + 3, label_y + baseline + 3),
                (0, 0, 0),
                cv2.FILLED,
            )
            # Write white text label
            cv2.putText(
                img,
                label_text,
                (label_x, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=2,
            )

        # Save annotated image preserving tree structure
        out_path = output_folder / img_path.relative_to(input_folder)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)


def save_entry_exit_database_to_json(
    entry_database: List[ProfileRecord],
    exit_database: List[ProfileRecord],
    filepath: Union[str, Path],
) -> None:
    """
    Saves entry and exit database profile records to a JSON file.

    Args:
        entry_database (List[ProfileRecord]): The database of entry records.
        exit_database (List[ProfileRecord]): The database of exit records.
        filepath (Union[str, Path]): Target output file path to write JSON database.

    Raises:
        OSError: If writing to the file fails.
    """
    def serialize_db(db: List[ProfileRecord]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Helper function to serialize a list of profile records.

        Args:
            db (List[ProfileRecord]): Database to serialize.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Serialized database records grouped by ID.
        """
        serialized_profiles = {}
        for r in db:
            ent_id_str = str(r.entity_id)
            if ent_id_str not in serialized_profiles:
                serialized_profiles[ent_id_str] = []

            feat_list = r.feature.tolist()
            hsv_list = r.hsv_hist.tolist() if r.hsv_hist is not None else None

            serialized_profiles[ent_id_str].append(
                {
                    "feature": feat_list,
                    "hsv_hist": hsv_list,
                    "aspect_ratio": r.aspect_ratio,
                    "timestamp": r.timestamp,
                    "img_name": r.img_name,
                    "box": r.box,
                }
            )
        return serialized_profiles

    data = {
        "entry_profiles": serialize_db(entry_database),
        "exit_profiles": serialize_db(exit_database),
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    logger.info("Saved combined entry/exit database to %s", filepath)


def run_entry_exit_profiling(
    entry_dir: Union[str, Path],
    exit_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    mode: Union[str, ProfilingMode] = ProfilingMode.DWELL,
    entry_direction: Union[str, Direction] = Direction.UNKNOWN,
    exit_direction: Union[str, Direction] = Direction.UNKNOWN,
    threshold: float = 0.75,
    max_gap: Optional[float] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    categories: str = "car,person",
    cores: int = 1,
    db_save: Optional[str] = None,
    report: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Exposes a unified entry/exit profiling workflow as a backend API.
    Can operate in either 'dwell' or 'occupancy' mode.

    Args:
        entry_dir (Union[str, Path]): Path to directory containing entry camera images.
        exit_dir (Union[str, Path]): Path to directory containing exit camera images.
        output_dir (Optional[Union[str, Path]]): Base output path for annotated images. Defaults to None.
        mode (Union[str, ProfilingMode]): Operation mode, either DWELL or OCCUPANCY. Defaults to ProfilingMode.DWELL.
        entry_direction (Union[str, Direction]): Target direction filter for entry camera. Defaults to Direction.UNKNOWN.
        exit_direction (Union[str, Direction]): Target direction filter for exit camera. Defaults to Direction.UNKNOWN.
        threshold (float): Similarity matching threshold. Defaults to 0.75.
        max_gap (Optional[float]): Max gap in seconds for tracking. Defaults to None.
        checkpoint (Optional[Union[str, Path]]): Path to custom ResNet-50 weights. Defaults to None.
        categories (str): Comma-separated class suffix list to target (e.g. 'car,person'). Defaults to 'car,person'.
        cores (int): Number of parallel CPU threads to extract features. Defaults to 1.
        db_save (Optional[str]): Filepath to serialize final ProfileDatabase. Defaults to None.
        report (Optional[str]): Filepath to save summary JSON report. Defaults to None.

    Returns:
        Dict[str, Any]: The summary report dict containing profiling results and stats.
    """
    entry_folder = Path(entry_dir)
    exit_folder = Path(exit_dir)

    # Convert direction parameters to Direction Enum if they are strings
    def to_enum(d):
        """
        Converts a raw string or direction object to a Direction Enum.

        Args:
            d (Union[str, Direction]): Target value to convert.

        Returns:
            Direction: Resolved Direction enum member.
        """
        if isinstance(d, Direction):
            return d
        if not d:
            return Direction.UNKNOWN
        d_str = str(d).lower().strip()
        if d_str in ("any", "unknown"):
            return Direction.UNKNOWN
        if d_str in ("left", "right-left", "going-left", "going_left"):
            return Direction.RIGHT_LEFT
        if d_str in ("right", "left-right", "going-right", "going_right"):
            return Direction.LEFT_RIGHT
        for m in Direction:
            if m.name.lower() == d_str.replace("-", "_"):
                return m
        return Direction.UNKNOWN

    entry_dir_enum = to_enum(entry_direction)
    exit_dir_enum = to_enum(exit_direction)

    def to_mode_enum(m):
        """
        Converts a raw string or mode object to a ProfilingMode Enum.

        Args:
            m (Union[str, ProfilingMode]): Target value to convert.

        Returns:
            ProfilingMode: Resolved ProfilingMode enum member.
        """
        if isinstance(m, ProfilingMode):
            return m
        if not m:
            return ProfilingMode.DWELL
        m_str = str(m).lower().strip()
        if m_str == "occupancy":
            return ProfilingMode.OCCUPANCY
        return ProfilingMode.DWELL

    mode_enum = to_mode_enum(mode)

    allowed_categories = [cat.strip().lower() for cat in categories.split(",")]

    def find_images(folder, direction_interest: Direction):
        """
        Scans folder for images ending in target category and matching direction of interest.

        Args:
            folder (Path): Directory path to search.
            direction_interest (Direction): Direction filter.

        Returns:
            List[Path]: List of matching image file paths.
        """
        imgs = []
        for p in folder.rglob("*"):
            if (
                p.is_file()
                and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
                and any(p.stem.lower().endswith(cat) for cat in allowed_categories)
            ):
                if direction_interest != Direction.UNKNOWN:
                    if determine_image_direction(p) != direction_interest:
                        continue
                imgs.append(p)
        return imgs

    entry_images = find_images(entry_folder, entry_dir_enum)
    exit_images = find_images(exit_folder, exit_dir_enum)

    logger.info("Found %d entry images and %d exit images after directional filtering.", len(entry_images), len(exit_images))

    if not entry_images and not exit_images:
        logger.warning("No matching images found in either entry or exit directories.")
        return {}

    # Load feature extractor model
    model, device, transform = load_feature_extractor(
        checkpoint_path=checkpoint
    )

    # Process entry images
    entry_results = {}
    if entry_images:
        logger.info("Processing entry directory images...")
        entry_results = extract_features_for_directory(
            entry_images, model, device, transform, cores=cores
        )
    entry_db, entry_grouped = track_entities_in_directory(
        entry_results, entry_images, threshold=threshold, max_gap=max_gap
    )
    entry_filtered = entry_grouped

    # Process exit images
    exit_results = {}
    if exit_images:
        logger.info("Processing exit directory images...")
        exit_results = extract_features_for_directory(
            exit_images, model, device, transform, cores=cores
        )
    exit_db, exit_grouped = track_entities_in_directory(
        exit_results, exit_images, threshold=threshold, max_gap=max_gap
    )
    exit_filtered = exit_grouped

    summary_report: Dict[str, Any] = {
        "metadata": {
            "entry_dir": str(entry_folder),
            "exit_dir": str(exit_folder),
            "mode": mode_enum.name.lower(),
            "entry_direction": entry_dir_enum.name.lower(),
            "exit_direction": exit_dir_enum.name.lower(),
            "threshold": threshold,
            "max_gap": max_gap,
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "statistics": {
            "entry_entities_detected": len(entry_grouped),
            "entry_entities_after_filtering": len(entry_filtered),
            "exit_entities_detected": len(exit_grouped),
            "exit_entities_after_filtering": len(exit_filtered),
        }
    }

    # Dwell vs Occupancy calculation
    if mode_enum == ProfilingMode.DWELL:
        logger.info("Calculating dwell times between entry and exit cameras...")
        matches = match_entry_exit_entities(entry_filtered, exit_filtered, threshold=threshold)
        summary_report["dwell_time_matches"] = matches
        summary_report["statistics"]["matched_entities"] = len(matches)
        
        if matches:
            avg_dwell = sum(m["dwell_time"] for m in matches) / len(matches)
        else:
            avg_dwell = 0.0
        summary_report["statistics"]["average_dwell_time"] = avg_dwell
        
        print("\n--- Dwell Time Summary ---")
        print(f"Total Matched Entities: {len(matches)}")
        print(f"Average Dwell Time: {avg_dwell:.2f} seconds\n")

    elif mode_enum == ProfilingMode.OCCUPANCY:
        logger.info("Calculating occupancy timeline...")
        timeline = calculate_occupancy_timeline(entry_filtered, exit_filtered)
        summary_report["occupancy_timeline"] = timeline
        max_occ = max([t["occupancy"] for t in timeline]) if timeline else 0
        summary_report["statistics"]["maximum_occupancy"] = max_occ
        
        print("\n--- Occupancy Summary ---")
        print(f"Maximum Running Occupancy: {max_occ}")
        print("Occupancy profiling complete!\n")

    # Annotate and save images if output_dir is specified
    if output_dir:
        out_folder = Path(output_dir)
        logger.info("Exporting entry annotated visual outputs to %s...", out_folder / "entry")
        
        # Build image mapping
        def build_mapping(db, results_dict):
            """
            Builds annotation mapping from DB record boxes and IDs.

            Args:
                db (List[ProfileRecord]): Bounding box records database.
                results_dict (Dict[Path, Any]): Original image paths results.

            Returns:
                Dict[Path, List[Tuple[List[int], int]]]: Image path mapped to list of annotations.
            """
            mapping = {}
            for r in db:
                matching_path = None
                for path in results_dict.keys():
                    if path.name == r.img_name:
                        matching_path = path
                        break
                if matching_path:
                    if matching_path not in mapping:
                        mapping[matching_path] = []
                    mapping[matching_path].append((r.box, r.entity_id))
            return mapping

        entry_mapping = build_mapping(entry_db, entry_results)
        annotate_and_save_images(entry_mapping, entry_folder, out_folder / "entry")

        logger.info("Exporting exit annotated visual outputs to %s...", out_folder / "exit")
        exit_mapping = build_mapping(exit_db, exit_results)
        annotate_and_save_images(exit_mapping, exit_folder, out_folder / "exit")

    # Save database JSON
    if db_save:
        save_entry_exit_database_to_json(entry_db, exit_db, db_save)

    # Save report JSON
    if report:
        logger.info("Generating profiling report at %s...", report)
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the image entity profiler script.

    Raises:
        SystemExit: If invalid configuration options are provided or model loading fails.
    """
    parser = argparse.ArgumentParser(
        description="Extract features and track unique entities chronologically across images."
    )
    # Made positional arguments optional to gracefully support --entry-dir and --exit-dir
    parser.add_argument(
        "input_dir", type=str, nargs="?", default=None, help="Path to the input directory of images."
    )
    parser.add_argument(
        "output_dir", type=str, nargs="?", default=None, help="Path to save visual detection output images."
    )
    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate for feature extraction (default: 1).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for reidentification matching (default: 0.75).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom fine-tuned PyTorch model checkpoint (.pth).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="car,person",
        help="Comma-separated list of category suffixes to profile (default: car,person).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )
    parser.add_argument(
        "--db-save",
        type=str,
        default=None,
        help="Path to save the final ProfileDatabase serialized JSON.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dwell",
        choices=["dwell", "occupancy"],
        help="The profiling operation mode (dwell: calculate travel/dwell times, occupancy: track concurrent counts).",
    )
    parser.add_argument(
        "--entry-dir",
        type=str,
        default=None,
        help="Path to the entry-camera directory of images.",
    )
    parser.add_argument(
        "--exit-dir",
        type=str,
        default=None,
        help="Path to the exit-camera directory of images.",
    )
    parser.add_argument(
        "--entry-direction",
        type=str,
        default="any",
        choices=["left", "right", "left-right", "right-left", "any"],
        help="The direction of interest for the entry camera.",
    )
    parser.add_argument(
        "--exit-direction",
        type=str,
        default="any",
        choices=["left", "right", "left-right", "right-left", "any"],
        help="The direction of interest for the exit camera.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help="Maximum time gap (in seconds) between sequential detections to allow matching.",
    )

    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)

    if args.cores <= 0:
        logger.error("The number of allocated CPU cores must be at least 1.")
        sys.exit(1)
    thread_count = args.cores

    if args.entry_dir and args.exit_dir:
        # Run dual-camera entry/exit workflow
        run_entry_exit_profiling(
            entry_dir=args.entry_dir,
            exit_dir=args.exit_dir,
            output_dir=output_folder or args.output_dir,
            mode=args.mode,
            entry_direction=args.entry_direction,
            exit_direction=args.exit_direction,
            threshold=args.threshold,
            max_gap=args.max_gap,
            checkpoint=args.checkpoint,
            categories=args.categories,
            cores=args.cores,
            db_save=args.db_save,
            report=args.report,
        )
    else:
        # Run standard single-camera workflow
        if not input_folder:
            logger.error("Must specify input_dir for single-camera profiling mode.")
            sys.exit(1)

        allowed_categories = [cat.strip().lower() for cat in args.categories.split(",")]
        all_images = [
            p
            for p in input_folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            and any(p.stem.lower().endswith(cat) for cat in allowed_categories)
        ]

        logger.info(
            "Allocating %d CPU core(s) to ResNet-50 feature extraction...", thread_count
        )

        if not all_images:
            logger.warning("No matching images found in the input directory.")
            return

        # Load feature extractor model once in main thread to share
        try:
            model, device, transform = load_feature_extractor(
                checkpoint_path=args.checkpoint
            )
        except Exception as e:
            logger.error("Failed to load feature extractor model: %s", e)
            sys.exit(1)

        # Phase 1: Parallel feature extraction
        results_dict = extract_features_for_directory(
            all_images, model, device, transform, cores=thread_count
        )

        # Phase 2: Chronological Re-identification (Sequential)
        logger.info("Matching and tracking entities across images...")
        database, grouped = track_entities_in_directory(
            results_dict, all_images, threshold=args.threshold, max_gap=args.max_gap
        )

        # Print summary to CLI
        print("\n--- Summary ---")
        print(f"Total Unique Entities: {len(grouped)}")
        print("Entity profiling complete!\n")

        # Export annotated visual outputs
        if output_folder:
            logger.info("Exporting annotated visual outputs to %s...", output_folder)
            
            image_entity_mappings = {}
            for r in database:
                matching_path = None
                for path in results_dict.keys():
                    if path.name == r.img_name:
                        matching_path = path
                        break
                if matching_path:
                    if matching_path not in image_entity_mappings:
                        image_entity_mappings[matching_path] = []
                    image_entity_mappings[matching_path].append((r.box, r.entity_id))

            annotate_and_save_images(image_entity_mappings, input_folder, output_folder)

        # Save database JSON serialization if requested
        if args.db_save:
            save_database_to_json(database, len(grouped) + 1, args.db_save)

        # Generate JSON summary report if requested
        if args.report:
            logger.info("Generating profiling report at %s...", args.report)
            report_data = {
                "metadata": {
                    "input_dir": str(input_folder),
                    "output_dir": str(output_folder) if output_folder else None,
                    "model_checkpoint": args.checkpoint,
                    "reid_threshold": args.threshold,
                    "total_images_processed": len(all_images),
                    "generated_at": datetime.datetime.now().isoformat(),
                },
                "statistics": {
                    "total_unique_entities": len(grouped),
                },
                "entities": {},
            }

            for entity_id, records in grouped.items():
                entity_occurrences = []
                for rec in records:
                    entity_occurrences.append(
                        {
                            "image_name": rec.img_name,
                            "timestamp": rec.timestamp,
                            "box": rec.box,
                            "aspect_ratio": rec.aspect_ratio,
                        }
                    )
                report_data["entities"][str(entity_id)] = entity_occurrences

            with open(args.report, "w") as f:
                json.dump(report_data, f, indent=4)


if __name__ == "__main__":
    main()
