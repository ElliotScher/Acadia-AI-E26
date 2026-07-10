"""
Plate Dwell Profiler

A functional backend library and command-line tool to compute dwell time for
vehicle crossings by matching on OCR'd license plate text, rather than visual
re-identification. A license plate is an exact, unique identifier, so matching
is a simple lookup rather than a feature-similarity/threshold problem.

Unlike image_occupancyprofiler.py, this does not split footage into entry vs.
exit directories/directions. Getting a readable plate requires the camera to
face the vehicle roughly head-on, which means the vehicle is moving toward or
away from the camera rather than laterally across the frame - so there's no
reliable left/right travel direction to filter on the way
video_entityprofiler.py's direction tag assumes. Instead, every plate sighting
from one or more directories is pooled together: for a given plate, the
earliest sighting is treated as its entry and the latest sighting as its exit.

Expects a directory of images that have already been produced by
video_plateextractor.py, so each image is already a tight crop of a single
license plate - no bounding box, no further detection needed, just OCR. The
directory's plate_manifest.json (written by video_plateextractor.py) is used
for accurate timestamps, since a plate crop no longer contains the source
frame's burned-in on-screen timestamp text.
"""

import argparse
import datetime
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

from src.utility.imgutils import (
    PLATE_MANIFEST_FILENAME,
    extract_plate_text_via_ocr,
    get_timestamp,
)

# Initialize Logger
logger = logging.getLogger("plate_dwellprofiler")

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


@dataclass
class PlateDetection:
    """
    A single OCR'd license plate reading.

    Args:
        plate_text (str): The normalized (uppercase, alphanumeric-only) plate text.
        timestamp (float): UNIX timestamp the image was captured.
        img_path (Path): Source image path.
    """

    plate_text: str
    timestamp: float
    img_path: Path


def find_images(folder: Path) -> List[Path]:
    """
    Finds all image files in a directory, recursively.

    Deliberately does not filter by the "left"/"right" travel-direction tag
    video_entityprofiler.py encodes in each filename, unlike
    image_occupancyprofiler.py - a plate camera faces the vehicle head-on to
    get a readable plate, so there's no meaningful lateral travel direction
    to filter on here.

    Args:
        folder (Path): Directory path to search.

    Returns:
        List[Path]: Sorted list of matching image file paths.
    """
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_manifest(folder: Path) -> Dict[Path, float]:
    """
    Loads video_plateextractor.py's plate_manifest.json from a directory, if present.

    Args:
        folder (Path): Directory that may contain a plate_manifest.json.

    Returns:
        Dict[Path, float]: Maps each crop's absolute path to its recorded
            timestamp. Empty if no manifest is found.
    """
    manifest_path = folder / PLATE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read manifest %s: %s", manifest_path, e)
        return {}

    return {folder / rel_path: entry["timestamp"] for rel_path, entry in data.items()}


def extract_plate_detections(
    image_paths: List[Path], manifest: Optional[Dict[Path, float]] = None
) -> List[PlateDetection]:
    """
    OCRs each image's plate text and returns a chronologically sorted list of
    successful detections.

    Each image is expected to already be a tight plate crop (produced by
    video_plateextractor.py), so no bounding box detection is needed here -
    just OCR. Images whose text can't be OCR'd are skipped and logged, since
    a vehicle can't be identified without its plate.

    Args:
        image_paths (List[Path]): Image file paths to process.
        manifest (Optional[Dict[Path, float]]): Path-to-timestamp mapping from
            load_manifest, preferred over get_timestamp since a plate crop no
            longer contains the source frame's burned-in timestamp text.
            Defaults to None (always use get_timestamp).

    Returns:
        List[PlateDetection]: Detections sorted chronologically by timestamp.
    """
    manifest = manifest or {}
    detections: List[PlateDetection] = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image %s; skipping.", img_path)
            continue

        plate_text = extract_plate_text_via_ocr(img)
        if not plate_text:
            logger.warning("Could not read plate text from %s; skipping.", img_path)
            continue

        timestamp = manifest.get(img_path)
        if timestamp is None:
            timestamp = get_timestamp(img_path)

        detections.append(
            PlateDetection(
                plate_text=plate_text, timestamp=timestamp, img_path=img_path
            )
        )

    detections.sort(key=lambda d: d.timestamp)
    return detections


def compute_plate_dwell_times(
    detections: List[PlateDetection],
) -> Tuple[List[Dict[str, Any]], List[PlateDetection]]:
    """
    Groups plate readings by plate text and derives dwell time from timing
    alone: for each plate, its earliest sighting is treated as the entry and
    its latest sighting as the exit, regardless of which camera/directory
    either reading came from.

    A plate seen only once can't produce a dwell time (there's no second
    sighting to mark an exit) and is returned separately. A plate seen three
    or more times only uses its first and last sighting - readings in between
    aren't a distinct crossing, just the same vehicle still present.

    Args:
        detections (List[PlateDetection]): All plate readings to group, from
            any number of pooled directories/cameras.

    Returns:
        Tuple[List[Dict[str, Any]], List[PlateDetection]]: A tuple containing:
            - List[Dict[str, Any]]: Dwell records with plate text, image paths,
              entry/exit timestamps, dwell time, and total sighting count.
            - List[PlateDetection]: Plates seen only once, so no dwell time
              could be computed.
    """
    sightings: Dict[str, List[PlateDetection]] = defaultdict(list)
    for detection in detections:
        sightings[detection.plate_text].append(detection)

    matches: List[Dict[str, Any]] = []
    single_sightings: List[PlateDetection] = []

    for plate_text, readings in sightings.items():
        readings.sort(key=lambda d: d.timestamp)

        if len(readings) < 2:
            single_sightings.append(readings[0])
            continue

        entry_det = readings[0]
        exit_det = readings[-1]
        matches.append(
            {
                "plate_text": plate_text,
                "entry_image": str(entry_det.img_path),
                "exit_image": str(exit_det.img_path),
                "entry_time": entry_det.timestamp,
                "exit_time": exit_det.timestamp,
                "dwell_time": exit_det.timestamp - entry_det.timestamp,
                "num_sightings": len(readings),
            }
        )

    matches.sort(key=lambda m: m["entry_time"])
    single_sightings.sort(key=lambda d: d.timestamp)

    return matches, single_sightings


def run_plate_dwell_profiling(
    input_dir: Union[str, Path],
    report: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Exposes the plate-based dwell time profiling workflow as a backend API.

    Args:
        input_dir (Union[str, Path]): Path to directory containing plate images,
            already processed by video_plateextractor.py. Pooled from however
            many cameras/directories feed into it - not split by entry/exit.
        report (Optional[Union[str, Path]]): Filepath to save summary JSON report. Defaults to None.

    Returns:
        Dict[str, Any]: The summary report dict containing profiling results and stats.
    """
    input_folder = Path(input_dir)

    images = find_images(input_folder)
    logger.info("Found %d plate images.", len(images))

    manifest = load_manifest(input_folder)

    logger.info("Reading plates from images...")
    detections = extract_plate_detections(images, manifest=manifest)

    logger.info(
        "Successfully read plates from %d/%d images.", len(detections), len(images)
    )

    matches, single_sightings = compute_plate_dwell_times(detections)

    avg_dwell = sum(m["dwell_time"] for m in matches) / len(matches) if matches else 0.0

    summary_report: Dict[str, Any] = {
        "metadata": {
            "input_dir": str(input_folder),
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "statistics": {
            "input_images": len(images),
            "plates_read": len(detections),
            "matched_crossings": len(matches),
            "single_sightings": len(single_sightings),
            "average_dwell_time": avg_dwell,
        },
        "dwell_time_matches": matches,
        "single_sightings": [
            {
                "plate_text": d.plate_text,
                "image": str(d.img_path),
                "timestamp": d.timestamp,
            }
            for d in single_sightings
        ],
    }

    print("\n--- Plate Dwell Time Summary ---")
    print(f"Plates Read: {len(detections)}/{len(images)}")
    print(f"Total Matched Crossings: {len(matches)}")
    print(f"Single Sightings (no dwell computable): {len(single_sightings)}")
    print(f"Average Dwell Time: {avg_dwell:.2f} seconds\n")

    if report:
        logger.info("Generating profiling report at %s...", report)
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the plate dwell profiler script.

    Raises:
        SystemExit: If required directories are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Compute dwell time for vehicle crossings by matching OCR'd "
        "license plate text: a plate's earliest sighting is its entry, its "
        "latest sighting is its exit."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory of plate-cropped images (already processed "
        "by video_plateextractor.py). Pool however many cameras/directories "
        "feed into this one directory - entry and exit aren't distinguished "
        "by source, only by which sighting of a plate comes first vs. last.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )

    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, _ = setup_logging_and_paths(parser, logger)
    assert input_folder is not None

    run_plate_dwell_profiling(
        input_dir=input_folder,
        report=args.report,
    )


if __name__ == "__main__":
    main()
