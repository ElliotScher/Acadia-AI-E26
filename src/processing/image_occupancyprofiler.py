"""
Image Occupancy Profiler

A backend library and command-line tool to extract features from green bounding boxes
(detected inside images) using a pre-trained ResNet-50 network and re-identify/track
entities chronologically across images to compute occupancy.
"""

import argparse
import csv
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

import matplotlib

from utility.geometryutils import Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Direction(Enum):
    LEFT_RIGHT = auto()
    RIGHT_LEFT = auto()
    UNKNOWN = auto()


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
logger = logging.getLogger("image_occupancyprofiler")


class GeM(nn.Module):
    """
    Generalized-Mean (GeM) pooling layer.

    Generalizes global average pooling (p=1) toward max pooling (p -> inf),
    emphasizing the most salient spatial activations in the feature map. This
    is a standard re-identification "bag of tricks" swap for the plain
    average pool in a classification backbone, and produces embeddings that
    are noticeably more discriminative for instance-level retrieval/matching
    than vanilla GAP features, without requiring any re-id-specific training
    or additional pretrained weights.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / self.p)


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

    # Swap the plain global-average-pool for GeM pooling to sharpen the
    # discriminative power of the embedding for instance re-identification
    if hasattr(model, "avgpool"):
        model.avgpool = GeM()

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


def select_best_match(
    similarities: Dict[int, float],
    threshold: float = 0.75,
    ratio_threshold: Optional[float] = 0.8,
) -> Tuple[Optional[int], float]:
    """
    Selects the best-matching entity ID from a similarity dict, rejecting ambiguous matches.

    A candidate must clear an absolute similarity `threshold`, and (when a second-best
    candidate exists) must also pass a ratio test comparing its distance (1 - similarity)
    against the second-best candidate's distance - analogous to Lowe's ratio test used for
    SIFT keypoint matching. This rejects matches that merely clear the threshold but are not
    clearly better than the next-closest competing entity, which a bare absolute threshold
    would otherwise accept as a false positive whenever the gallery contains multiple
    similar-looking entities.

    Args:
        similarities (Dict[int, float]): Mapping of candidate entity_id to similarity score.
        threshold (float): Minimum absolute similarity threshold for a match. Defaults to 0.75.
        ratio_threshold (Optional[float]): Maximum allowed ratio of best-to-second-best distance.
            Lower is stricter. Set to None to disable the ratio test entirely. Defaults to 0.8.

    Returns:
        Tuple[Optional[int], float]: A tuple containing:
            - Optional[int]: The matched entity ID, or None if no candidate qualifies.
            - float: The best similarity score found (even if no candidate qualifies).
    """
    best_id = None
    best_sim = -1.0
    second_best_sim = -1.0
    for ent_id, sim in similarities.items():
        if sim > best_sim:
            second_best_sim = best_sim
            best_sim = sim
            best_id = ent_id
        elif sim > second_best_sim:
            second_best_sim = sim

    if best_id is None or best_sim < threshold:
        return None, best_sim

    if ratio_threshold is not None and second_best_sim > -1.0:
        best_dist = max(1.0 - best_sim, 1e-6)
        second_dist = max(1.0 - second_best_sim, 1e-6)
        if best_dist / second_dist > ratio_threshold:
            return None, best_sim

    return best_id, best_sim


def assign_entity_id(
    feat: np.ndarray,
    hsv_hist: Optional[np.ndarray],
    aspect_ratio: Optional[float],
    timestamp: Optional[float],
    img_name: Optional[str],
    database: List[ProfileRecord],
    next_id: int,
    threshold: float = 0.75,
    ratio_threshold: Optional[float] = 0.8,
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
        ratio_threshold (Optional[float]): Maximum allowed ratio of best-to-second-best distance
            used to reject ambiguous matches. Set to None to disable. Defaults to 0.8.

    Returns:
        Tuple[int, float, bool]: A tuple containing:
            - int: The assigned entity ID (either a matched historical ID or `next_id`).
            - float: The maximum similarity score computed against historical exemplars.
            - bool: True if this is a newly created entity, False if it was successfully matched.
    """
    similarities = compute_similarities(
        feat, hsv_hist, aspect_ratio, timestamp, img_name, database
    )
    best_id, best_sim = select_best_match(similarities, threshold, ratio_threshold)

    if best_id is not None:
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


def select_primary_box(boxes: List[Rectangle]) -> List[Rectangle]:
    """
    Reduces a list of raw detected boxes down to a single box for the entity
    in the image.

    Since each image is guaranteed to contain exactly one entity, any extra
    boxes returned by the upstream detector (e.g. nested/duplicate contours
    around the same green box) are noise. We keep only the single
    largest-area box and discard the rest, which guarantees exactly one
    bounding box is ever drawn per image.

    Args:
        boxes (List[Rectangle]): Raw detected boxes as Rectangle(x, y, w, h).

    Returns:
        List[Rectangle]: A list containing just the single largest box, or
            an empty list if no boxes were detected.
    """
    if not boxes:
        return []

    largest = max(boxes, key=lambda b: b[2] * b[3])
    return [largest]


def process_images_worker(
    img_paths: List[Path],
    progress_bar: tqdm,
    results_dict: Dict[Path, List[Dict[str, Any]]],
    lock: threading.Lock,
    model: nn.Module,
    device: torch.device,
    transform: Any,
    flip: bool = False,
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
        flip (bool): If True, horizontally mirrors each crop before feature extraction. Used when a
            single camera captures both directions of travel (e.g. a driveway), since a vehicle
            exiting is physically the same side of the vehicle as when it entered, just moving the
            opposite way on screen - a mirror image of its entry appearance. Defaults to False.
    """
    for img_path in img_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                progress_bar.update(1)
                continue

            boxes = detect_entities(img)
            boxes = select_primary_box(boxes)
            detections = []

            for box in boxes:
                x, y, w, h = box
                crop = img[y : y + h, x : x + w]
                if crop.size > 0:
                    if flip:
                        crop = cv2.flip(crop, 1)
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


def determine_image_direction(img_path: Path) -> Direction:
    """
    Placeholder function to determine if a vehicle in the input image is going left or right.
    Currently returns Direction.UNKNOWN as a placeholder stub.
    """
    # TODO: Implement actual image-level direction classification/flow analysis model
    if img_path.name.__contains__("left"):
        return Direction.RIGHT_LEFT
    elif img_path.name.__contains__("right"):
        return Direction.LEFT_RIGHT
    else:
        return Direction.UNKNOWN


def track_entities_in_directory(
    results_dict: Dict[Path, List[Dict[str, Any]]],
    image_paths: List[Path],
    threshold: float = 0.75,
    ratio_threshold: Optional[float] = 0.8,
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
                    r
                    for r in database
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
                ratio_threshold=ratio_threshold,
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
        List[Dict[str, Any]]: Chronologically sorted list of timeline states, each
            with the running "occupancy" as well as "entered_total" and
            "exited_total" cumulative counts up to and including that event.
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
    entered_total = 0
    exited_total = 0
    timeline = []
    for event in raw_timeline:
        current_occupancy += event["change"]
        if event["change"] > 0:
            entered_total += 1
        else:
            exited_total += 1
        timeline.append(
            {
                "timestamp": event["timestamp"],
                "occupancy": current_occupancy,
                "label": event["label"],
                "entered_total": entered_total,
                "exited_total": exited_total,
            }
        )

    return timeline


def save_occupancy_csv(
    timeline: List[Dict[str, Any]], filepath: Union[str, Path]
) -> None:
    """
    Writes the occupancy timeline to a CSV file, one row per entry/exit event.

    Args:
        timeline (List[Dict[str, Any]]): Chronological timeline records, as
            returned by `calculate_occupancy_timeline`.
        filepath (Union[str, Path]): Target output CSV file path.
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "datetime",
                "event",
                "entered_total",
                "exited_total",
                "occupancy",
            ]
        )
        for event in timeline:
            ts = event["timestamp"]
            iso = (
                datetime.datetime.fromtimestamp(ts).isoformat()
                if ts is not None
                else ""
            )
            writer.writerow(
                [
                    ts,
                    iso,
                    event["label"],
                    event["entered_total"],
                    event["exited_total"],
                    event["occupancy"],
                ]
            )
    logger.info("Saved occupancy CSV to %s", filepath)


def save_occupancy_graph(
    timeline: List[Dict[str, Any]], filepath: Union[str, Path]
) -> None:
    """
    Renders and saves a chart of occupancy and cumulative entries/exits over time.

    Args:
        timeline (List[Dict[str, Any]]): Chronological timeline records, as
            returned by `calculate_occupancy_timeline`.
        filepath (Union[str, Path]): Target output image file path (e.g. .png).
    """
    if not timeline:
        logger.warning("Occupancy timeline is empty; skipping graph generation.")
        return

    timestamps = [datetime.datetime.fromtimestamp(e["timestamp"]) for e in timeline]
    occupancy = [e["occupancy"] for e in timeline]
    entered = [e["entered_total"] for e in timeline]
    exited = [e["exited_total"] for e in timeline]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.step(
        timestamps,
        occupancy,
        where="post",
        color="#1a73e8",
        linewidth=2,
        label="Occupancy",
    )
    ax.plot(
        timestamps,
        entered,
        color="#34a853",
        linewidth=1.2,
        linestyle="--",
        label="Cumulative entries",
    )
    ax.plot(
        timestamps,
        exited,
        color="#ea4335",
        linewidth=1.2,
        linestyle="--",
        label="Cumulative exits",
    )

    ax.set_xlabel("Time", color="#52514e")
    ax.set_ylabel("Count", color="#52514e")
    ax.set_title(
        "Occupancy Over Time", color="#0b0b0b", fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors="#52514e")
    ax.grid(True, which="major", axis="both", color="#e3e2dd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper left", labelcolor="#0b0b0b")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(str(filepath), facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved occupancy graph to %s", filepath)


def extract_features_for_directory(
    image_paths: List[Path],
    model: nn.Module,
    device: torch.device,
    transform: Any,
    num_threads: int = 1,
    flip: bool = False,
) -> Dict[Path, List[Dict[str, Any]]]:
    """
    Helper function to run parallel feature extraction for a list of image paths.

    Args:
        image_paths (List[Path]): List of all image paths to process.
        model (nn.Module): Feature extractor model instance.
        device (torch.device): PyTorch device to execute on.
        transform (Any): Image pre-processing transform pipeline.
        num_threads (int): Number of parallel CPU worker threads to spawn. Defaults to 1.
        flip (bool): If True, horizontally mirrors each crop before feature extraction.
            See `process_images_worker` for rationale. Defaults to False.

    Returns:
        Dict[Path, List[Dict[str, Any]]]: A dictionary of detected crop features, aspect ratios,
            and timestamps, keyed by image path.
    """
    results_dict: Dict[Path, List[Dict[str, Any]]] = {}
    lock = threading.Lock()
    progress_bar = tqdm(
        total=len(image_paths), desc="Extracting Features", unit="image"
    )

    chunk_size = max(1, len(image_paths) // num_threads)
    threads = []

    for i in range(num_threads):
        start = i * chunk_size
        end = None if i == num_threads - 1 else (i + 1) * chunk_size
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
                flip,
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
    entry_direction: Union[str, Direction] = Direction.UNKNOWN,
    exit_direction: Union[str, Direction] = Direction.UNKNOWN,
    threshold: float = 0.75,
    ratio_threshold: Optional[float] = 0.8,
    max_gap: Optional[float] = None,
    checkpoint: Optional[Union[str, Path]] = None,
    categories: str = "car,person",
    threads: int = 1,
    db_save: Optional[str] = None,
    report: Optional[str] = None,
    occupancy_csv: Optional[str] = None,
    occupancy_graph: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified entry/exit occupancy profiling workflow.

    If `entry_dir` and `exit_dir` resolve to the same directory (a single camera
    used for both directions of travel, distinguished only by
    entry_direction/exit_direction), exit crops are horizontally flipped before
    feature extraction. A vehicle exiting past that same fixed camera shows the
    same physical side as it did entering, just moving the opposite way on
    screen, so its crop is a mirror image of its entry appearance. Two distinct
    cameras positioned across from each other don't need this, since each
    camera's placement already captures the corresponding side directly.

    Args:
        entry_dir (Union[str, Path]): Path to directory containing entry camera images.
        exit_dir (Union[str, Path]): Path to directory containing exit camera images.
        output_dir (Optional[Union[str, Path]]): Base output path for annotated images. Defaults to None.
        entry_direction (Union[str, Direction]): Target direction filter for entry camera. Defaults to Direction.UNKNOWN.
        exit_direction (Union[str, Direction]): Target direction filter for exit camera. Defaults to Direction.UNKNOWN.
        threshold (float): Similarity matching threshold. Defaults to 0.75.
        ratio_threshold (Optional[float]): Maximum allowed ratio of best-to-second-best distance
            used to reject ambiguous matches. Set to None to disable. Defaults to 0.8.
        max_gap (Optional[float]): Max gap in seconds for tracking. Defaults to None.
        checkpoint (Optional[Union[str, Path]]): Path to custom ResNet-50 weights. Defaults to None.
        categories (str): Comma-separated class suffix list to target (e.g. 'car,person'). Defaults to 'car,person'.
        threads (int): Number of parallel CPU threads to extract features. Defaults to 1.
        db_save (Optional[str]): Filepath to serialize final ProfileDatabase. Defaults to None.
        report (Optional[str]): Filepath to save summary JSON report. Defaults to None.
        occupancy_csv (Optional[str]): Filepath to save the occupancy timeline as CSV. Defaults to None.
        occupancy_graph (Optional[str]): Filepath to save an occupancy-over-time chart image. Defaults to None.

    Returns:
        Dict[str, Any]: The summary report dict containing profiling results and stats.
    """
    entry_folder = Path(entry_dir)
    exit_folder = Path(exit_dir)

    # A single camera capturing both directions of travel (e.g. one camera on a
    # driveway used for both entry and exit) sees the same physical side of an
    # exiting vehicle as it did on entry, just moving the opposite way on
    # screen - a horizontal mirror image of its entry appearance. Two distinct
    # cameras positioned across from each other don't have this issue, since
    # each camera's fixed placement (selected via --entry-direction/
    # --exit-direction) already captures the correct corresponding side.
    same_directory = entry_folder.resolve() == exit_folder.resolve()

    # Convert direction parameters to Direction Enum if they are strings
    def to_enum(d : Union[str, Direction]):
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

    logger.info(
        "Found %d entry images and %d exit images after directional filtering.",
        len(entry_images),
        len(exit_images),
    )

    if not entry_images and not exit_images:
        logger.warning("No matching images found in either entry or exit directories.")
        return {}

    # Load feature extractor model
    model, device, transform = load_feature_extractor(checkpoint_path=checkpoint)

    # Process entry images
    entry_results = {}
    if entry_images:
        logger.info("Processing entry directory images...")
        entry_results = extract_features_for_directory(
            entry_images, model, device, transform, num_threads=threads
        )
    entry_db, entry_grouped = track_entities_in_directory(
        entry_results,
        entry_images,
        threshold=threshold,
        ratio_threshold=ratio_threshold,
        max_gap=max_gap,
    )
    entry_filtered = entry_grouped

    # Process exit images
    exit_results = {}
    if exit_images:
        logger.info("Processing exit directory images...")
        if same_directory:
            logger.info(
                "Entry and exit directories are the same; horizontally "
                "flipping exit crops before feature extraction."
            )
        exit_results = extract_features_for_directory(
            exit_images, model, device, transform, num_threads=threads, flip=same_directory
        )
    exit_db, exit_grouped = track_entities_in_directory(
        exit_results,
        exit_images,
        threshold=threshold,
        ratio_threshold=ratio_threshold,
        max_gap=max_gap,
    )
    exit_filtered = exit_grouped

    summary_report: Dict[str, Any] = {
        "metadata": {
            "entry_dir": str(entry_folder),
            "exit_dir": str(exit_folder),
            "entry_direction": entry_dir_enum.name.lower(),
            "exit_direction": exit_dir_enum.name.lower(),
            "threshold": threshold,
            "ratio_threshold": ratio_threshold,
            "max_gap": max_gap,
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "statistics": {
            "entry_entities_detected": len(entry_grouped),
            "entry_entities_after_filtering": len(entry_filtered),
            "exit_entities_detected": len(exit_grouped),
            "exit_entities_after_filtering": len(exit_filtered),
        },
    }

    # Occupancy calculation
    logger.info("Calculating occupancy timeline...")
    timeline = calculate_occupancy_timeline(entry_filtered, exit_filtered)
    summary_report["occupancy_timeline"] = timeline
    max_occ = max([t["occupancy"] for t in timeline]) if timeline else 0
    summary_report["statistics"]["maximum_occupancy"] = max_occ

    total_entered = timeline[-1]["entered_total"] if timeline else 0
    total_exited = timeline[-1]["exited_total"] if timeline else 0
    summary_report["statistics"]["total_entered"] = total_entered
    summary_report["statistics"]["total_exited"] = total_exited

    print("\n--- Occupancy Summary ---")
    print(f"Total Entered: {total_entered}")
    print(f"Total Exited: {total_exited}")
    print(f"Maximum Running Occupancy: {max_occ}")
    print("Occupancy profiling complete!\n")

    # Save occupancy timeline CSV if requested
    if occupancy_csv:
        save_occupancy_csv(timeline, occupancy_csv)

    # Save occupancy timeline graph if requested
    if occupancy_graph:
        save_occupancy_graph(timeline, occupancy_graph)

    # Annotate and save images if output_dir is specified
    if output_dir:
        out_folder = Path(output_dir)
        logger.info(
            "Exporting entry annotated visual outputs to %s...", out_folder / "entry"
        )

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

        logger.info(
            "Exporting exit annotated visual outputs to %s...", out_folder / "exit"
        )
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
    parser = argparse.ArgumentParser(
        description="Extract features and track unique entities chronologically across images."
    )
    # Made positional arguments optional to gracefully support --entry-dir and --exit-dir
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to the input directory of images.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to save visual detection output images.",
    )
    parser.add_argument(
        "-c",
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads to allocate for feature extraction (default: 1).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for reidentification matching (default: 0.75).",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.8,
        help="Maximum allowed ratio of best-to-second-best match distance; rejects ambiguous "
        "matches even if they clear --threshold (default: 0.8). Pass a value <= 0 to disable.",
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
    parser.add_argument(
        "--occupancy-csv",
        type=str,
        default=None,
        help="Path to save the occupancy timeline (entries, exits, running occupancy) as CSV. "
        "Only applies to dual-camera --entry-dir/--exit-dir mode.",
    )
    parser.add_argument(
        "--occupancy-graph",
        type=str,
        default=None,
        help="Path to save a chart image (e.g. .png) of occupancy and cumulative entries/exits "
        "over time. Only applies to dual-camera --entry-dir/--exit-dir mode.",
    )

    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)

    if args.threads <= 0:
        logger.error("The number of allocated CPU threads must be at least 1.")
        sys.exit(1)
    thread_count = args.threads

    ratio_threshold = args.ratio_threshold if args.ratio_threshold > 0 else None

    if args.entry_dir and args.exit_dir:
        # Run dual-camera entry/exit workflow
        run_entry_exit_profiling(
            entry_dir=args.entry_dir,
            exit_dir=args.exit_dir,
            output_dir=output_folder or args.output_dir,
            entry_direction=args.entry_direction,
            exit_direction=args.exit_direction,
            threshold=args.threshold,
            ratio_threshold=ratio_threshold,
            max_gap=args.max_gap,
            checkpoint=args.checkpoint,
            categories=args.categories,
            threads=args.threads,
            db_save=args.db_save,
            report=args.report,
            occupancy_csv=args.occupancy_csv,
            occupancy_graph=args.occupancy_graph,
        )
    else:
        # Run standard single-camera workflow
        if not input_folder:
            logger.error("Must specify input_dir for single-camera profiling mode.")
            sys.exit(1)

        if args.occupancy_csv or args.occupancy_graph:
            logger.warning(
                "--occupancy-csv/--occupancy-graph require dual-camera "
                "--entry-dir/--exit-dir mode; ignoring in single-camera mode."
            )

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
            all_images, model, device, transform, num_threads=thread_count
        )

        # Phase 2: Chronological Re-identification (Sequential)
        logger.info("Matching and tracking entities across images...")
        database, grouped = track_entities_in_directory(
            results_dict,
            all_images,
            threshold=args.threshold,
            ratio_threshold=ratio_threshold,
            max_gap=args.max_gap,
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
                    "reid_ratio_threshold": ratio_threshold,
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
