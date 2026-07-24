"""
Video Plate Extractor

A standalone command-line tool that runs a license-plate-detection YOLO model
directly against every (optionally downsampled) frame of raw video footage and
crops tightly to each plate found, ready for plate_dwellprofiler.py.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from src.utility.imgutils import (
    PLATE_MANIFEST_FILENAME,
    get_timestamp,
    load_video_start_times,
    validate_video_start_times,
)
from utility.parallel import ProgressTracker
from utility.yoloutility import load_model

# Initialize Logger
logger = logging.getLogger("video_plateextractor")

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# Extra margin added around each detected plate box before cropping, since OCR
# tends to read a plate with a small border more reliably than a razor-tight crop.
CROP_PADDING_PCT = 0.08


def find_videos(folder: Path) -> List[Path]:
    """
    Finds all video files in a directory, recursively.

    Args:
        folder (Path): Directory path to search.

    Returns:
        List[Path]: Sorted list of matching video file paths.
    """
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def crop_with_padding(
    img, x1: int, y1: int, x2: int, y2: int, padding_pct: float = CROP_PADDING_PCT
):
    """
    Crops an image to a box, expanded by a small margin on every side.

    Args:
        img (np.ndarray): OpenCV BGR image to crop from.
        x1 (int): Box left edge.
        y1 (int): Box top edge.
        x2 (int): Box right edge.
        y2 (int): Box bottom edge.
        padding_pct (float): Fraction of the box's width/height to pad on each
            side. Defaults to CROP_PADDING_PCT.

    Returns:
        np.ndarray: The padded crop, clamped to the image bounds.
    """
    h_img, w_img = img.shape[:2]
    pad_x = int((x2 - x1) * padding_pct)
    pad_y = int((y2 - y1) * padding_pct)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)
    return img[y1:y2, x1:x2]


def extract_plate_crops(
    model: YOLO, frame, conf_threshold: float
) -> List[Dict[str, Any]]:
    """
    Runs plate detection on one frame and returns every plate crop found.

    Unlike a single best-frame-per-car extractor, a raw video frame may
    contain more than one vehicle's plate at once, so every detection above
    the confidence threshold is cropped and returned rather than only the
    highest-confidence box.

    Args:
        model (YOLO): Loaded plate-detection YOLO model.
        frame (np.ndarray): OpenCV BGR frame to search.
        conf_threshold (float): Minimum detection confidence.

    Returns:
        List[Dict[str, Any]]: A list of dicts, each with "crop" (np.ndarray)
            and "confidence" (float), one per detected plate.
    """
    predictions = model.predict(source=frame, conf=conf_threshold, verbose=False)

    results: List[Dict[str, Any]] = []
    for r in predictions:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = crop_with_padding(frame, x1, y1, x2, y2)
            if crop.size == 0:
                continue
            results.append({"crop": crop, "confidence": float(box.conf[0])})

    return results


def process_video(
    video_path: Path,
    input_folder: Path,
    output_folder: Path,
    model: YOLO,
    conf_threshold: float,
    downsample_factor: int,
    manifest: Dict[str, Dict[str, Any]],
    progress_bar: Optional[tqdm | ProgressTracker] = None,
    start_time: Optional[datetime.datetime] = None,
) -> int:
    """
    Runs plate detection frame-by-frame on one video, saving every crop found
    and recording it in the shared manifest.

    Args:
        video_path (Path): Path to the video file to process.
        input_folder (Path): Root input directory, for relative path resolution.
        output_folder (Path): Directory to save extracted plate crops to.
        model (YOLO): Loaded plate-detection YOLO model.
        conf_threshold (float): Minimum detection confidence.
        downsample_factor (int): Process every Nth frame.
        manifest (Dict[str, Dict[str, Any]]): Shared manifest dict to record
            each saved crop's timestamp/confidence/source into, keyed by the
            crop's path relative to output_folder.
        progress_bar (Optional[tqdm | ProgressTracker]): Progress bar to update per-frame. Defaults to None.
        start_time (Optional[datetime.datetime]): This video's start-time
            override (the caller picks the right entry out of a
            load_video_start_times list positionally), for footage whose
            mtime is unreliable. Takes priority over get_timestamp's
            OCR/filename/mtime fallback chain when given. Defaults to None.

    Returns:
        int: Number of plate crops saved for this video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Failed to open video file %s", video_path)
        return 0

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

    rel_dir = video_path.relative_to(input_folder).with_suffix("")

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if progress_bar is not None:
            progress_bar.update(1)

        if frame_idx % downsample_factor != 0:
            continue

        crops = extract_plate_crops(model, frame, conf_threshold)
        if not crops:
            continue

        timestamp = video_start_time + (frame_idx / fps)

        for i, result in enumerate(crops):
            out_name = f"frame{frame_idx}_{i}_plate.jpg"
            out_path = output_folder / rel_dir / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), result["crop"])

            manifest[str(out_path.relative_to(output_folder))] = {
                "timestamp": timestamp,
                "confidence": result["confidence"],
                "source_video": str(video_path),
                "frame_index": frame_idx,
            }
            saved += 1

    cap.release()
    return saved


def run_video_plate_extraction(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    plate_model: Union[str, Path],
    conf_threshold: float = 0.25,
    downsample_factor: int = 1,
    start_times: Optional[List[datetime.datetime]] = None,
    report: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Exposes the video-to-plate-crop extraction workflow as a backend API.

    Args:
        input_dir (Union[str, Path]): Directory of raw video files.
        output_dir (Union[str, Path]): Directory to save extracted plate crops to.
        plate_model (Union[str, Path]): Path to a YOLO model checkpoint trained
            specifically for license plate detection.
        conf_threshold (float): Minimum detection confidence. Defaults to 0.25.
        downsample_factor (int): Process every Nth frame to trade thoroughness
            for speed. Defaults to 1 (every frame).
        start_times (Optional[List[datetime.datetime]]): One start-time
            override per video, in the same sorted-by-path order find_videos
            returns - not read from a file here; the caller (e.g. main(), via
            load_video_start_times) is responsible for providing the actual
            list. Defaults to None (use get_timestamp's OCR/filename/mtime
            fallback chain for every video).
        report (Optional[Union[str, Path]]): Filepath to save summary JSON report. Defaults to None.

    Returns:
        Dict[str, Any]: The summary report dict containing extraction results and stats.

    Raises:
        ValueError: If start_times is given and its length doesn't match the
            number of videos found in input_dir.
    """
    input_folder = Path(input_dir)
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_folder)
    logger.info("Found %d videos.", len(videos))

    try:
        validate_video_start_times(start_times, len(videos))
    except ValueError as e:
        logger.error(str(e))
        raise

    try:
        model = load_model(str(plate_model))
    except Exception as e:
        logger.error("Error loading plate model '%s': %s", plate_model, e)
        raise

    manifest: Dict[str, Dict[str, Any]] = {}

    for i, video_path in enumerate(videos):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = 0
        if cap.isOpened():
            val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_frames = int(val) if isinstance(val, (int, float)) else 0
        cap.release()

        progress_bar = tqdm(total=total_frames, desc=video_path.name, unit="frame")
        saved = process_video(
            video_path=video_path,
            input_folder=input_folder,
            output_folder=output_folder,
            model=model,
            conf_threshold=conf_threshold,
            downsample_factor=downsample_factor,
            manifest=manifest,
            progress_bar=progress_bar,
            start_time=start_times[i] if start_times is not None else None,
        )
        progress_bar.close()

        logger.info("Extracted %d plate crops from %s.", saved, video_path.name)

    manifest_path = output_folder / PLATE_MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info(
        "Extracted %d plate crops total to %s (manifest: %s)",
        len(manifest),
        output_folder,
        manifest_path,
    )

    summary_report: Dict[str, Any] = {
        "metadata": {
            "input_dir": str(input_folder),
            "output_dir": str(output_folder),
            "plate_model": str(plate_model),
            "confidence_threshold": conf_threshold,
            "downsample_factor": downsample_factor,
            "start_times_provided": start_times is not None,
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "statistics": {
            "videos_processed": len(videos),
            "plates_extracted": len(manifest),
        },
    }

    print("\n--- Plate Extraction Summary ---")
    print(f"Videos Processed: {len(videos)}")
    print(f"Plates Extracted: {len(manifest)}")
    print("Plate extraction complete!\n")

    if report:
        logger.info("Generating extraction report at %s...", report)
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the video plate extractor script.

    Raises:
        SystemExit: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Crop every license plate detected in raw video footage "
        "using a plate-detection YOLO model, ready for plate_dwellprofiler.py."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory of raw video files.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to save extracted plate crops to.",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        required=True,
        help="Path to a YOLO model checkpoint trained specifically for license "
        "plate detection.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for plate detection (default: 0.25).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Process every Nth frame to trade thoroughness for speed (default: 1, "
        "every frame).",
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
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )

    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)
    assert input_folder is not None and output_folder is not None

    if args.downsample <= 0:
        logger.error("--downsample must be at least 1.")
        raise SystemExit(1)

    start_times = load_video_start_times(args.start_times) if args.start_times else None

    run_video_plate_extraction(
        input_dir=input_folder,
        output_dir=output_folder,
        plate_model=args.plate_model,
        conf_threshold=args.conf,
        downsample_factor=args.downsample,
        start_times=start_times,
        report=args.report,
    )


if __name__ == "__main__":
    main()
