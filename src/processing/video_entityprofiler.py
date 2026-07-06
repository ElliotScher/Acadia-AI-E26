"""
Video Entity Extractor

Parses video files to track unique entities frame-to-frame by locating
green bounding boxes, and exports the best (largest and sharpest) raw frames
for downstream analysis by image_entityprofiler.py.
"""

import argparse
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.utility.geometryutils import Rectangle
from src.utility.imgutils import (
    detect_entities,
    get_center_crop,
    get_hsv_hist,
    get_timestamp,
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

    def __post_init__(self):
        """
        Initializes the boxes history with the initial bounding box.
        """
        if self.boxes_history is None:
            self.boxes_history = [self.last_box]

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


@dataclass
class Detection:
    """
    Holds metadata and crop data for a single detected entity in a video frame.

    Args:
        box (Rectangle): Bounding box coordinates.
        hsv_hist (np.ndarray): HSV histogram of the entity crop.
        aspect_ratio (float): Bounding box width / height ratio.
        crop (np.ndarray): Image crop of the entity.
    """

    box: Rectangle
    hsv_hist: np.ndarray
    aspect_ratio: float
    crop: np.ndarray


def process_video(
    video_path: Path,
    downsample_factor: int = 1,
    zones: Optional[List[Dict[str, Any]]] = None,
    progress_bar: Optional[tqdm] = None,
) -> List[VideoEntityRecord]:
    """
    Ingests a video file, tracks unique green bounding boxes frame-by-frame,
    and returns a list of VideoEntityRecord objects.

    Args:
        video_path (Path): Path to the video file to process.
        downsample_factor (int): Process every Nth frame. Defaults to 1.
        zones (Optional[List[Dict[str, Any]]]): Optional inclusion/exclusion zones list.
        progress_bar (Optional[tqdm]): Progress bar to update per-frame. Defaults to None.

    Returns:
        List[VideoEntityRecord]: List of tracked entity records with their best frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video file %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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
            x, y, w, h = box
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            hsv_hist = get_hsv_hist(get_center_crop(crop, 0.12))
            aspect_ratio = float(w) / h

            current_detections.append(
                Detection(
                    box=Rectangle(x=x, y=y, w=w, h=h),
                    hsv_hist=hsv_hist,
                    aspect_ratio=aspect_ratio,
                    crop=crop,
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
        last_center_x = last_box.x + last_box.w / 2
        direction = "right" if last_center_x >= first_center_x else "left"

        best_frame_ts = video_start_time + (track.best_frame_idx / fps)

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
        )
        finalized_records.append(record)

    return finalized_records


def main() -> None:
    """
    Main CLI entry point for the video entity profiler.
    """
    parser = argparse.ArgumentParser(
        description="Extract best raw frames containing tracked green boxes for image_entityprofiler.py."
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
        help="Category suffix to append to saved filenames (default: car).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=2,
        help="Process every Nth frame to optimize speed (default: 2).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)

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

    # Find videos
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    video_files = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if not video_files:
        logger.warning("No video files found in the input directory.")
        return

    all_records: List[VideoEntityRecord] = []

    for video_path in video_files:
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
        )
        progress_bar.close()

        logger.info("Found %d unique entities in %s.", len(records), video_path.name)
        all_records.extend(records)

    # Export raw best frames
    logger.info("Saving best frames to %s...", output_folder)
    for record in all_records:
        video_name = record.video_path.stem
        # Append the direction and category suffix so image_entityprofiler.py detects it
        out_name = f"entity_{record.entity_id}_{record.direction}_{args.category}.jpg"
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
