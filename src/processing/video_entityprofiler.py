"""
Video Entity Extractor

Parses video files to track unique entities frame-to-frame by locating
green bounding boxes, and exports the best (largest and sharpest) raw frames
for downstream analysis by image_occupancyprofiler.py.
"""

import argparse
import datetime
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.detection.classes import CLASS_ID_MAPPING, TARGET_CLASSES
from src.utility.geometryutils import Rectangle
from src.utility.imgutils import (
    detect_entities,
    get_center_crop,
    get_hsv_hist,
    get_timestamp,
    load_video_start_times,
    validate_video_start_times,
)
from src.utility.loggingutils import setup_logging_and_paths
from utility import imgutils

# Initialize Logger
logger = logging.getLogger("video_entityprofiler")


@dataclass
class VideoEntityRecord:
    """
    Holds profile information for a unique entity tracked throughout a video.

    Args:
        video_path (Path): Path to the source video file.
        entity_id (int): Unique integer ID of the entity.
        best_frame_idx (int): The index of the frame with the highest quality score.
        best_frame (np.ndarray): Full BGR image of the best frame.
        best_crop (np.ndarray): BGR crop of the entity from the best frame.
        best_box (Rectangle): Bounding box coordinates of the best crop.
        timestamp (float): Absolute UNIX timestamp of the best frame.
        hsv_hist (np.ndarray): Normalised 3D HSV color histogram.
        aspect_ratio (float): Width / Height ratio of the crop.
        direction (str): "left" or "right" indicating travel direction.
        entity_type (Optional[str]): The track's majority-vote YOLO
            classification (e.g. "car", "bicycle", "motorcycle"), letting
            speed be scoped to one type via --entity-type. None if
            classification is disabled (--no-classify) or never found a
            confident match.
        relative_speed (float): The track's raw pixel-displacement-per-second
            rate straight out of process_video, until compute_relative_speeds
            rescales it into a 0-1 ratio of the fastest entity in the batch
            (which running the normal pipeline via main() does automatically).
        absolute_speed (Optional[float]): The entity's real-world speed, in
            whatever unit was used to calibrate it. None until
            calibrate_absolute_speeds is explicitly called with a reference
            entity of known speed.
    """

    video_path: Path
    entity_id: int
    best_frame_idx: int
    best_frame: np.ndarray  # Full BGR image of the best frame
    best_crop: np.ndarray  # BGR crop of the entity from the best frame
    best_box: Rectangle  # Rectangle of the best bounding box
    timestamp: float  # Absolute UNIX timestamp of the best frame
    hsv_hist: np.ndarray  # L2-normalized 3D HSV color histogram
    aspect_ratio: float  # Width / Height of the best crop
    direction: str  # "left" or "right" indicating travel direction
    entity_type: Optional[str] = None  # Majority-vote YOLO classification
    relative_speed: float = 0.0  # Raw pixels/sec until normalized to [0, 1]
    absolute_speed: Optional[float] = None  # Calibrated real-world speed


@dataclass
class Track:
    """
    Represents an active track of an entity across video frames.

    Args:
        entity_id (int): Unique integer ID of the entity.
        last_box (Rectangle): The last matched bounding box coordinates.
        last_frame_idx (int): The frame index where the entity was last seen.
        hsv_hists (List[np.ndarray]): Historical list of HSV histograms.
        best_score (float): Maximum computed frame quality score. Defaults to -1.0.
        best_frame_idx (int): Frame index of the best quality representation. Defaults to -1.
        best_frame (Optional[np.ndarray]): The full image frame of the best representation. Defaults to None.
        best_crop (Optional[np.ndarray]): BGR crop of the entity of the best representation. Defaults to None.
        best_box (Optional[Rectangle]): Bounding box of the best representation. Defaults to None.
        best_hsv_hist (Optional[np.ndarray]): HSV histogram of the best representation. Defaults to None.
        best_aspect_ratio (float): Aspect ratio of the best representation. Defaults to -1.0.
        boxes_history (List[Rectangle]): Bounding box positions over time. Defaults to None.
        first_frame_idx (int): The frame index where the entity was first seen,
            used to compute the track's speed. Defaults to -1.
        type_votes (Counter): Tally of classified entity_type seen across the
            track's lifetime - finalization picks the majority, so a few
            misclassified frames don't flip the whole track's type.
    """

    entity_id: int
    last_box: Rectangle
    last_frame_idx: int
    hsv_hists: List[np.ndarray]
    best_score: float = -1.0
    best_frame_idx: int = -1
    best_frame: Optional[np.ndarray] = None
    best_crop: Optional[np.ndarray] = None
    best_box: Optional[Rectangle] = None
    best_hsv_hist: Optional[np.ndarray] = None
    best_aspect_ratio: float = -1.0
    boxes_history: List[Rectangle] = None
    first_frame_idx: int = -1
    type_votes: Counter = field(default_factory=Counter)

    def __post_init__(self):
        """
        Initializes the boxes history with the initial bounding box, and
        first_frame_idx with the frame the track was spawned on if not
        given explicitly.
        """
        if self.boxes_history is None:
            self.boxes_history = [self.last_box]
        if self.first_frame_idx == -1:
            self.first_frame_idx = self.last_frame_idx

    def update_track(
        self,
        box: Rectangle,
        frame_idx: int,
        hsv_hist: np.ndarray,
    ) -> None:
        """
        Updates the position and history of the track.

        Args:
            box (Rectangle): BGR bounding box coordinates.
            frame_idx (int): Current frame index.
            hsv_hist (np.ndarray): Normalised 3D HSV histogram for the frame crop.
        """
        self.last_box = box
        self.last_frame_idx = frame_idx
        self.hsv_hists.append(hsv_hist)
        if self.boxes_history is None:
            self.boxes_history = []
        self.boxes_history.append(box)

    def update_best_frame(
        self,
        box: Rectangle,
        frame_idx: int,
        quality_score: float,
        frame: np.ndarray,
        crop: np.ndarray,
        hsv_hist: np.ndarray,
        aspect_ratio: float,
    ) -> None:
        """
        Updates the best frame representation of the track.

        Args:
            box (Rectangle): BGR bounding box coordinates.
            frame_idx (int): Current frame index.
            quality_score (float): Computed quality score for the frame crop.
            frame (np.ndarray): Full BGR image frame.
            crop (np.ndarray): BGR image crop of the entity.
            hsv_hist (np.ndarray): Normalised 3D HSV histogram for the crop.
            aspect_ratio (float): Width / Height ratio of the crop.
        """
        if quality_score > self.best_score:
            self.best_score = quality_score
            self.best_frame_idx = frame_idx
            self.best_frame = frame.copy()
            self.best_crop = crop.copy()
            self.best_box = box
            self.best_hsv_hist = hsv_hist
            self.best_aspect_ratio = aspect_ratio

    def record_type(self, entity_type: Optional[str]) -> None:
        """
        Tallies a vote for this frame's classified entity_type, if any.

        Args:
            entity_type (Optional[str]): The current frame's classification,
                or None if classification is disabled or found nothing
                confident enough - either way, no vote is recorded.
        """
        if entity_type is not None:
            self.type_votes[entity_type] += 1


@dataclass
class Detection:
    """
    Holds metadata and crop data for a single detected entity in a video frame.

    Args:
        box (Rectangle): Bounding box coordinates.
        hsv_hist (np.ndarray): HSV histogram of the entity crop.
        aspect_ratio (float): Bounding box width / height ratio.
        crop (np.ndarray): Image crop of the entity.
        entity_type (Optional[str]): YOLO classification of this frame's
            crop, or None if classification is disabled or found nothing
            confident enough. Defaults to None.
    """

    box: Rectangle
    hsv_hist: np.ndarray
    aspect_ratio: float
    crop: np.ndarray
    entity_type: Optional[str] = None


def classify_entity_type(
    model: YOLO,
    crop: np.ndarray,
    target_classes: List[int],
    conf_threshold: float,
) -> Optional[str]:
    """
    Classifies a single tracked entity's crop with a general-purpose YOLO
    model, so speed can later be scoped to one entity type (e.g. only
    bicycles) instead of every entity in a run being pooled together.

    Runs detection on the crop itself - already tightly bounded by the green
    marker box detect_entities found - rather than matching against
    full-frame YOLO detections, so no separate IoU-matching step against the
    tracker's own boxes is needed.

    Args:
        model (YOLO): Loaded YOLO model instance.
        crop (np.ndarray): BGR image crop of the tracked entity.
        target_classes (List[int]): COCO class IDs to consider.
        conf_threshold (float): Minimum confidence for a classification to count.

    Returns:
        Optional[str]: The highest-confidence class label found in the crop
            (bus/truck merged into "car", matching video_yolo.py's category
            grouping), or None if nothing cleared conf_threshold.
    """
    results = model.predict(
        source=crop, conf=conf_threshold, classes=target_classes, verbose=False
    )

    best_label: Optional[str] = None
    best_conf = 0.0
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf <= best_conf:
                continue
            label = CLASS_ID_MAPPING.get(int(box.cls[0]))
            if label is None:
                continue
            best_conf = conf
            best_label = label

    if best_label in ("bus", "truck"):
        best_label = "car"

    return best_label


def process_video(
    video_path: Path,
    downsample_factor: int = 1,
    zones: Optional[List[Dict[str, Any]]] = None,
    progress_bar: Optional[tqdm] = None,
    start_time: Optional[datetime.datetime] = None,
    yolo_model: Optional[YOLO] = None,
    classify_classes: Optional[List[int]] = None,
    classify_conf: float = 0.25,
) -> List[VideoEntityRecord]:
    """
    Ingests a video file, tracks unique green bounding boxes frame-by-frame,
    and returns a list of VideoEntityRecord objects.

    Args:
        video_path (Path): Path to the video file to process.
        downsample_factor (int): Process every Nth frame. Defaults to 1.
        zones (Optional[List[Dict[str, Any]]]): Optional inclusion/exclusion zones list.
        progress_bar (Optional[tqdm]): Progress bar to update per-frame. Defaults to None.
        start_time (Optional[datetime.datetime]): This video's start-time
            override (the caller picks the right entry out of a start-times
            list positionally - see main()), for footage whose mtime is
            unreliable. Takes priority over get_timestamp's OCR/filename/mtime
            fallback chain when given. Defaults to None.
        yolo_model (Optional[YOLO]): Loaded YOLO model to classify each
            tracked entity's type (see classify_entity_type). Classification
            is skipped entirely, and every record's entity_type stays None,
            when not given. Defaults to None.
        classify_classes (Optional[List[int]]): COCO class IDs to consider
            when classifying. Defaults to None (src.detection.classes.TARGET_CLASSES).
        classify_conf (float): Minimum confidence for a classification to
            count. Only used when yolo_model is given. Defaults to 0.25.

    Returns:
        List[VideoEntityRecord]: List of tracked entity records with their best frames.
    """
    classify_classes = (
        classify_classes if classify_classes is not None else TARGET_CLASSES
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video file %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    video_start_time = start_time.timestamp() if start_time is not None else None
    if video_start_time is None:
        try:
            video_start_time = get_timestamp(video_path)
        except (FileNotFoundError, OSError):
            logger.warning(
                "Could not parse start timestamp for %s. Defaulting to 0.0.",
                video_path.name,
            )
            video_start_time = 0.0

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    inclusion_zones: List[Rectangle] = []
    exclusion_zones: List[Rectangle] = []

    if zones and img_w > 0 and img_h > 0:
        for zone in zones:
            if "rect" in zone:
                nx, ny, nw, nh = zone["rect"]
            elif "points" in zone and len(zone["points"]) > 0:
                points = zone["points"]
                nx = min(p[0] for p in points)
                ny = min(p[1] for p in points)
                nw = max(p[0] for p in points) - nx
                nh = max(p[1] for p in points) - ny
            else:
                continue

            rect = Rectangle(
                x=int(nx * img_w),
                y=int(ny * img_h),
                w=int(nw * img_w),
                h=int(nh * img_h),
            )
            if zone.get("type") == "include":
                inclusion_zones.append(rect)
            elif zone.get("type") == "exclude":
                exclusion_zones.append(rect)

    active_tracks: Dict[int, Track] = {}
    next_track_id = 1
    max_lost_frames = int(fps * 2)  # Keep track active for 2 seconds of missed frames

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if progress_bar is not None:
            progress_bar.update(1)

        if frame_idx % downsample_factor != 0:
            continue

        # Detect green bounding boxes
        boxes = detect_entities(frame)

        current_detections: List[Detection] = []
        for box in boxes:
            x, y, w, h = box.x, box.y, box.w, box.h
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            hsv_hist = get_hsv_hist(get_center_crop(crop, 0.12))
            aspect_ratio = float(w) / h

            entity_type = None
            if yolo_model is not None:
                entity_type = classify_entity_type(
                    yolo_model, crop, classify_classes, classify_conf
                )

            current_detections.append(
                Detection(
                    box=Rectangle(x=x, y=y, w=w, h=h),
                    hsv_hist=hsv_hist,
                    aspect_ratio=aspect_ratio,
                    crop=crop,
                    entity_type=entity_type,
                )
            )

        # Match current detections to active tracks by IoU
        match_matrix: List[Tuple[int, int, float]] = []
        for det_idx, det in enumerate(current_detections):
            for track_id, track in active_tracks.items():
                frames_since_last_seen = frame_idx - track.last_frame_idx
                if frames_since_last_seen > max_lost_frames:
                    continue

                iou = Rectangle.compute_iou(det.box, track.last_box)

                if iou >= 0.15:
                    match_matrix.append((det_idx, track_id, iou))

        # Sort matches by IoU score descending
        match_matrix.sort(key=lambda item: item[2], reverse=True)

        matched_dets = set()
        matched_tracks = set()

        for det_idx, track_id, score in match_matrix:
            if det_idx in matched_dets or track_id in matched_tracks:
                continue

            matched_dets.add(det_idx)
            matched_tracks.add(track_id)

            track = active_tracks[track_id]
            det = current_detections[det_idx]
            assert isinstance(det, Detection)

            # Always update tracking position and history
            track.update_track(
                box=det.box,
                frame_idx=frame_idx,
                hsv_hist=det.hsv_hist,
            )
            track.record_type(det.entity_type)

            # Check if detection falls within the allowed ROI zones
            is_excluded = Rectangle.is_box_excluded_by_zones(
                det.box,
                inclusion_zones=inclusion_zones,
                exclusion_zones=exclusion_zones,
            )

            if not is_excluded:
                # Compute quality score: area * image sharpness
                sharpness = imgutils.compute_sharpness(det.crop)
                area = det.box.w * det.box.h
                quality_score = area * sharpness

                track.update_best_frame(
                    box=det.box,
                    frame_idx=frame_idx,
                    quality_score=quality_score,
                    frame=frame,
                    crop=det.crop,
                    hsv_hist=det.hsv_hist,
                    aspect_ratio=det.aspect_ratio,
                )

        # Spawn new tracks for unmatched detections
        for det_idx, det in enumerate(current_detections):
            if det_idx not in matched_dets:
                active_tracks[next_track_id] = Track(
                    entity_id=next_track_id,
                    last_box=det.box,
                    last_frame_idx=frame_idx,
                    hsv_hists=[det.hsv_hist],
                )
                active_tracks[next_track_id].record_type(det.entity_type)

                # Check if detection falls within the allowed ROI zones
                is_excluded = Rectangle.is_box_excluded_by_zones(
                    det.box,
                    inclusion_zones=inclusion_zones,
                    exclusion_zones=exclusion_zones,
                )

                if not is_excluded:
                    # Compute quality score: area * image sharpness
                    sharpness = imgutils.compute_sharpness(det.crop)
                    area = det.box.w * det.box.h
                    quality_score = area * sharpness

                    active_tracks[next_track_id].update_best_frame(
                        box=det.box,
                        frame_idx=frame_idx,
                        quality_score=quality_score,
                        frame=frame,
                        crop=det.crop,
                        hsv_hist=det.hsv_hist,
                        aspect_ratio=det.aspect_ratio,
                    )
                next_track_id += 1

    cap.release()

    finalized_records = []
    for track_id, track in active_tracks.items():
        best_frame = track.best_frame
        best_crop = track.best_crop
        best_box = track.best_box
        best_hsv_hist = track.best_hsv_hist

        if (
            best_frame is None
            or best_crop is None
            or best_box is None
            or best_hsv_hist is None
        ):
            continue

        # Determine direction based on start and end position of the track
        first_box = track.boxes_history[0]
        last_box = track.boxes_history[-1]
        first_center_x = first_box.x + first_box.w / 2
        first_center_y = first_box.y + first_box.h / 2
        last_center_x = last_box.x + last_box.w / 2
        last_center_y = last_box.y + last_box.h / 2
        direction = "right" if last_center_x >= first_center_x else "left"

        # Raw speed: straight-line pixel displacement (first to last box,
        # same reasoning as direction above) over elapsed real time. This is
        # not yet comparable across entities/videos - compute_relative_speeds
        # rescales it into a 0-1 ratio, and calibrate_absolute_speeds can
        # further convert that ratio into a real-world unit.
        elapsed_frames = track.last_frame_idx - track.first_frame_idx
        elapsed_seconds = elapsed_frames / fps
        displacement_px = (
            (last_center_x - first_center_x) ** 2
            + (last_center_y - first_center_y) ** 2
        ) ** 0.5
        pixel_speed = displacement_px / elapsed_seconds if elapsed_seconds > 0 else 0.0

        best_frame_ts = video_start_time + (track.best_frame_idx / fps)

        entity_type = (
            track.type_votes.most_common(1)[0][0] if track.type_votes else None
        )

        record = VideoEntityRecord(
            video_path=video_path,
            entity_id=track.entity_id,
            best_frame_idx=track.best_frame_idx,
            best_frame=best_frame,
            best_crop=best_crop,
            best_box=best_box,
            timestamp=best_frame_ts,
            hsv_hist=best_hsv_hist,
            aspect_ratio=track.best_aspect_ratio,
            direction=direction,
            entity_type=entity_type,
            relative_speed=pixel_speed,
        )
        finalized_records.append(record)

    return finalized_records


def compute_relative_speeds(
    records: List[VideoEntityRecord],
) -> List[VideoEntityRecord]:
    """
    Normalizes every record's relative_speed - a raw pixel-displacement-per-
    second rate straight out of process_video - into a 0-1 ratio of the
    fastest entity observed across the whole batch, so speeds become
    comparable across entities and even across videos/cameras with different
    framing.

    This deliberately divides by the max rather than doing a min-max rescale:
    a plain ratio stays linearly proportional to the raw rate (ratio_i =
    speed_i / max_speed), which is exactly what lets calibrate_absolute_speeds
    convert it into a real-world unit from just one reference entity - a
    min-max rescale would break that (its slowest entity is pinned to 0
    regardless of its actual speed, and calibrating off a 0 is impossible).

    Args:
        records (List[VideoEntityRecord]): All tracked entities to normalize
            together - typically every entity found across an entire run (all
            videos), so speeds are compared on equal footing. Mutated in place.

    Returns:
        List[VideoEntityRecord]: The same records, with relative_speed
            rescaled to [0, 1].
    """
    if not records:
        return records

    max_speed = max(r.relative_speed for r in records)
    if max_speed <= 0:
        for r in records:
            r.relative_speed = 0.0
        return records

    for r in records:
        r.relative_speed = r.relative_speed / max_speed

    return records


def calibrate_absolute_speeds(
    records: List[VideoEntityRecord],
    reference_entity_id: int,
    reference_speed: float,
    reference_video_path: Optional[Union[str, Path]] = None,
) -> List[VideoEntityRecord]:
    """
    Scales every record's relative_speed into the same real-world unit as one
    reference entity's known actual speed (e.g. mph), populating absolute_speed.

    relative_speed (see compute_relative_speeds) is a pure ratio of each
    entity's raw pixel-per-second rate to the fastest observed rate, so it's
    linearly proportional to real-world speed. That means a single
    multiplicative factor, derived from one entity whose true speed is
    already known, converts every other entity's relative_speed into the
    same unit - no camera calibration (focal length, distance-to-road, lane
    width, etc.) required.

    Args:
        records (List[VideoEntityRecord]): All tracked entities, already
            normalized by compute_relative_speeds. Mutated in place.
        reference_entity_id (int): entity_id of the record to calibrate
            against. entity_id is only unique within a single video
            (process_video restarts numbering at 1 for every video), so pass
            reference_video_path too whenever records spans more than one video.
        reference_speed (float): The reference entity's known real-world
            speed, in whatever unit every record's absolute_speed should end
            up in (e.g. mph, km/h).
        reference_video_path (Optional[Union[str, Path]]): Video filename (or
            path) the reference entity came from, to disambiguate when
            multiple videos produced the same entity_id. Matched by filename
            alone, so a bare name like "clip.mp4" is enough. Defaults to None.

    Returns:
        List[VideoEntityRecord]: The same records, with absolute_speed populated.

    Raises:
        ValueError: If no record matches reference_entity_id (and
            reference_video_path, if given), more than one record matches and
            reference_video_path wasn't given to disambiguate, or the matched
            reference has a relative_speed of 0 (a stationary/degenerate
            reference can't be used to derive a scale factor).
    """
    candidates = [r for r in records if r.entity_id == reference_entity_id]

    if reference_video_path is not None:
        ref_name = Path(reference_video_path).name
        candidates = [r for r in candidates if Path(r.video_path).name == ref_name]

    if not candidates:
        raise ValueError(
            f"No entity with entity_id {reference_entity_id} found"
            + (f" in video '{reference_video_path}'" if reference_video_path else "")
            + "."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple entities with entity_id {reference_entity_id} found across "
            "different videos - pass reference_video_path to disambiguate."
        )

    reference = candidates[0]
    if reference.relative_speed <= 0:
        raise ValueError(
            f"Entity {reference_entity_id} has a relative speed of 0 and can't "
            "be used as a calibration reference."
        )

    scale = reference_speed / reference.relative_speed
    for r in records:
        r.absolute_speed = r.relative_speed * scale

    return records


def main() -> None:
    """
    Main CLI entry point for the video entity profiler.
    """
    parser = argparse.ArgumentParser(
        description="Extract best raw frames containing tracked green boxes for image_occupancyprofiler.py."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input directory containing video files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory to save best frames.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="car",
        help="Fallback category suffix appended to saved filenames for any "
        "entity that isn't classified (--no-classify given, or "
        "classification found nothing confident) (default: car).",
    )
    parser.add_argument(
        "--classify-model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights (e.g. yolo26s.pt) to classify each tracked "
        "entity's real object type (car/bicycle/motorcycle/etc.) from its "
        "crop, so speed can be scoped to one entity type via --entity-type "
        "(default: yolo26s.pt, matching video_yolo.py; see --no-classify to "
        "disable). Any entity classification doesn't find a confident match "
        "for falls back to --category.",
    )
    parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip YOLO classification entirely - every entity falls back "
        "to --category, matching the pipeline's pre-classification behavior. "
        "Faster, since it skips a YOLO forward pass per tracked crop.",
    )
    parser.add_argument(
        "--classify-conf",
        type=float,
        default=0.25,
        help="Minimum confidence for a classification to count. Only used "
        "when classification is enabled (default: 0.25).",
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        default=None,
        help="If given (e.g. 'bicycle'), only entities classified as this "
        "type are kept - every other entity is dropped before speed "
        "normalization, calibration, frame export, and the report. "
        "Incompatible with --no-classify.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Process every Nth frame to optimize speed (default: 2).",
    )
    parser.add_argument(
        "--start-times",
        type=str,
        default=None,
        help="Path to a JSON file overriding video start timestamps, for footage "
        "whose file mtime is unreliable (e.g. copied or re-encoded). A plain "
        "JSON array with exactly one entry per video found in input_dir, each "
        "either a UNIX epoch number or an ISO 8601 string (e.g. "
        "'2026-07-08T14:30:00'), given in the same order the videos are found "
        "in (sorted by full path, recursively). Defaults to None (use the "
        "usual OCR/filename/mtime resolution for every video).",
    )
    parser.add_argument(
        "--reference-entity-id",
        type=int,
        default=None,
        help="entity_id of one tracked vehicle whose actual real-world speed is "
        "known, to calibrate every entity's absolute_speed from its "
        "relative_speed (see calibrate_absolute_speeds). Requires "
        "--reference-speed. If more than one video produced this entity_id, "
        "also pass --reference-video to disambiguate.",
    )
    parser.add_argument(
        "--reference-speed",
        type=float,
        default=None,
        help="The reference entity's actual real-world speed (e.g. in mph) - "
        "used with --reference-entity-id. Whatever unit you give here is the "
        "unit every entity's absolute_speed ends up in.",
    )
    parser.add_argument(
        "--reference-video",
        type=str,
        default=None,
        help="Filename of the video --reference-entity-id came from, to "
        "disambiguate when multiple videos produced that entity_id. Only "
        "needed if --reference-entity-id is otherwise ambiguous.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)
    assert input_folder is not None and output_folder is not None

    if args.reference_entity_id is not None and args.reference_speed is None:
        logger.error("--reference-entity-id requires --reference-speed.")
        raise SystemExit(1)

    if args.entity_type is not None and args.no_classify:
        logger.error("--entity-type requires classification (can't combine with --no-classify).")
        raise SystemExit(1)

    yolo_model = YOLO(args.classify_model) if not args.no_classify else None

    start_times = load_video_start_times(args.start_times) if args.start_times else None

    # Load labels.json zones if present
    labels_file = input_folder / "labels.json"
    zones = []
    if labels_file.exists():
        try:
            with open(labels_file, "r") as f:
                db = json.load(f)
                zones = db.get("__zones__", [])
            logger.info("Loaded %d zones from labels.json", len(zones))
        except Exception as e:
            logger.warning("Failed to load zones from labels.json: %s", e)

    # Find videos - sorted by full path, since a --start-times list is matched
    # to videos positionally rather than by filename.
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    video_files = sorted(
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    )

    if not video_files:
        logger.warning("No video files found in the input directory.")
        return

    try:
        validate_video_start_times(start_times, len(video_files))
    except ValueError as e:
        logger.error(str(e))
        raise SystemExit(1)

    all_records: List[VideoEntityRecord] = []

    for i, video_path in enumerate(video_files):
        logger.info("Processing video: %s", video_path.name)
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        progress_bar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        records = process_video(
            video_path=video_path,
            downsample_factor=args.downsample,
            zones=zones,
            progress_bar=progress_bar,
            start_time=start_times[i] if start_times is not None else None,
            yolo_model=yolo_model,
            classify_conf=args.classify_conf,
        )
        progress_bar.close()

        logger.info("Found %d unique entities in %s.", len(records), video_path.name)
        all_records.extend(records)

    if args.entity_type is not None:
        before = len(all_records)
        all_records = [r for r in all_records if r.entity_type == args.entity_type]
        logger.info(
            "Filtered to entity_type '%s': %d/%d entities kept.",
            args.entity_type,
            len(all_records),
            before,
        )

    # Normalize every entity's raw speed into a 0-1 relative_speed together,
    # so speeds are comparable across entities/videos rather than each
    # video's tracks only being comparable to themselves.
    compute_relative_speeds(all_records)

    if args.reference_entity_id is not None:
        try:
            calibrate_absolute_speeds(
                all_records,
                reference_entity_id=args.reference_entity_id,
                reference_speed=args.reference_speed,
                reference_video_path=args.reference_video,
            )
            logger.info(
                "Calibrated absolute_speed for %d entities using entity_id %d "
                "at %.2f as the reference.",
                len(all_records),
                args.reference_entity_id,
                args.reference_speed,
            )
        except ValueError as e:
            logger.error("Could not calibrate absolute speeds: %s", e)
            raise SystemExit(1)

    # Export raw best frames
    logger.info("Saving best frames to %s...", output_folder)
    for record in all_records:
        video_name = record.video_path.stem
        # Append the direction and category suffix so image_occupancyprofiler.py
        # detects it - the classified entity_type when available, else the
        # manual --category fallback.
        category = record.entity_type or args.category
        out_name = f"entity_{record.entity_id}_{record.direction}_{category}.jpg"
        out_path = output_folder / video_name / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), record.best_frame)

    print("\n--- Summary ---")
    print(f"Total processed videos: {len(video_files)}")
    print(f"Total extracted entities: {len(all_records)}")
    left_count = sum(1 for r in all_records if r.direction == "left")
    right_count = sum(1 for r in all_records if r.direction == "right")
    print(f"Entities traveling left: {left_count}")
    print(f"Entities traveling right: {right_count}")
    print("Video processing and frame extraction complete!\n")

    # Generate JSON summary report if requested
    if args.report:
        logger.info("Generating report at %s...", args.report)
        report_data = {
            "metadata": {
                "input_dir": str(input_folder),
                "output_dir": str(output_folder),
                "total_videos_processed": len(video_files),
                "category": args.category,
                "classify_model": None if args.no_classify else args.classify_model,
                "entity_type_filter": args.entity_type,
                "start_times_file": args.start_times,
                "reference_entity_id": args.reference_entity_id,
                "reference_speed": args.reference_speed,
                "reference_video": args.reference_video,
                "generated_at": datetime.datetime.now().isoformat(),
            },
            "individual_entities": [
                {
                    "video": r.video_path.name,
                    "entity_id": r.entity_id,
                    "best_frame_idx": r.best_frame_idx,
                    "timestamp": r.timestamp,
                    "aspect_ratio": r.aspect_ratio,
                    "direction": r.direction,
                    "entity_type": r.entity_type,
                    "relative_speed": r.relative_speed,
                    "absolute_speed": r.absolute_speed,
                    "best_box": [
                        r.best_box.x,
                        r.best_box.y,
                        r.best_box.w,
                        r.best_box.h,
                    ],
                }
                for r in all_records
            ],
        }
        with open(args.report, "w") as f:
            json.dump(report_data, f, indent=4)


if __name__ == "__main__":
    main()
