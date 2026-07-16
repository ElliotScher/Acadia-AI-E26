"""
Plate Dwell Profiler

A backend library and command-line tool to compute dwell time for
vehicle crossings by matching on OCR'd license plate text, rather than visual
re-identification.
"""

import argparse
import datetime
import json
import logging
import sys
from collections import Counter, defaultdict
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


def levenshtein_distance(a: str, b: str) -> int:
    """
    Computes the Levenshtein (edit) distance between two strings.

    Args:
        a (str): First string.
        b (str): Second string.

    Returns:
        int: Minimum number of single-character insertions, deletions, or
            substitutions required to turn a into b.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def _build_dwell_record(readings: List[PlateDetection]) -> Dict[str, Any]:
    """
    Builds one match's dwell record from its readings (assumed 2+, already
    time-sorted): entry/exit images and timestamps, dwell time, sighting
    count, and a plate_text picked by majority vote across the readings (all
    identical under exact matching; may vary under fuzzy matching, where
    plate_text_variants lists every distinct string actually seen).

    Args:
        readings (List[PlateDetection]): List of PlateDetection objects.

    Returns:
        Dict[str, Any] Dictionary of plate matches
    """
    entry_det = readings[0]
    exit_det = readings[-1]
    text_counts = Counter(d.plate_text for d in readings)
    plate_text = text_counts.most_common(1)[0][0]
    return {
        "plate_text": plate_text,
        "entry_image": str(entry_det.img_path),
        "exit_image": str(exit_det.img_path),
        "entry_time": entry_det.timestamp,
        "exit_time": exit_det.timestamp,
        "dwell_time": exit_det.timestamp - entry_det.timestamp,
        "num_sightings": len(readings),
        "images": [str(d.img_path) for d in readings],
        "plate_text_variants": sorted(text_counts),
    }


def _group_by_exact_text(
    detections: List[PlateDetection],
) -> Tuple[List[Dict[str, Any]], List[PlateDetection]]:
    """
    Readings share a crossing only if their plate_text is identical.

    Args:
        detections (List[PlateDetection]): List of PlateDetection objects.

    Returns:
        Tuple[List[Dict[str, Any]]: List of fully matched detections
        List[PlateDetection]]: List of single sighted plates
    """
    sightings: Dict[str, List[PlateDetection]] = defaultdict(list)
    for detection in detections:
        sightings[detection.plate_text].append(detection)

    matches: List[Dict[str, Any]] = []
    single_sightings: List[PlateDetection] = []

    for readings in sightings.values():
        readings.sort(key=lambda d: d.timestamp)
        if len(readings) < 2:
            single_sightings.append(readings[0])
        else:
            matches.append(_build_dwell_record(readings))

    return matches, single_sightings


def _group_by_fuzzy_text(
    detections: List[PlateDetection],
    max_edit_distance: int,
    max_time_gap: Optional[float],
) -> Tuple[List[Dict[str, Any]], List[PlateDetection]]:
    """
    Clusters readings by chained similarity instead of exact equality: two
    readings join the same crossing if their plate_text is within
    max_edit_distance of each other AND (if max_time_gap is given) within
    max_time_gap seconds of each other - chained transitively via union-find,
    so A~B~C cluster into one crossing even if A and C aren't directly close.
    This absorbs the single-character misreads that would otherwise fracture
    one real, continuous presence into several disconnected "crossings".

    Args:
        detections (List[PlateDetection]): List of PlateDetection objects.
        max_edit_distance (int): Maximum edit distance between two readings.
        max_time_gap (Optional[float]): Maximum time gap between two readings.

    Returns:
        Tuple[List[Dict[str, Any]]: List of fully matched detections
        List[PlateDetection]]: List of single sighted plates
    """
    ordered = sorted(detections, key=lambda d: d.timestamp)
    n = len(ordered)
    parent = list(range(n))

    def find(element: int) -> int:
        while parent[element] != element:
            parent[element] = parent[parent[element]]
            element = parent[element]
        return element

    def union(element_1: int, element_2: int) -> None:
        root_i, root_j = find(element_1), find(element_2)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            if (
                max_time_gap is not None
                and ordered[j].timestamp - ordered[i].timestamp > max_time_gap
            ):
                break  # sorted by time - no later j can satisfy the gap either
            if (
                levenshtein_distance(ordered[i].plate_text, ordered[j].plate_text)
                <= max_edit_distance
            ):
                union(i, j)

    clusters: Dict[int, List[PlateDetection]] = defaultdict(list)
    for i, detection in enumerate(ordered):
        clusters[find(i)].append(detection)

    matches: List[Dict[str, Any]] = []
    single_sightings: List[PlateDetection] = []

    for readings in clusters.values():
        readings.sort(key=lambda d: d.timestamp)
        if len(readings) < 2:
            single_sightings.append(readings[0])
        else:
            matches.append(_build_dwell_record(readings))

    return matches, single_sightings


def compute_plate_dwell_times(
    detections: List[PlateDetection],
    max_edit_distance: int = 0,
    max_time_gap: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[PlateDetection]]:
    """
    Groups plate readings into crossings and derives dwell time from timing
    alone: for each crossing, its earliest sighting is treated as the entry
    and its latest sighting as the exit, regardless of which camera/directory
    either reading came from.

    Args:
        detections (List[PlateDetection]): All plate readings to group, from
            any number of pooled directories/cameras.
        max_edit_distance (int): Maximum Levenshtein distance between two
            readings' plate_text for them to join the same crossing. Defaults
            to 0 (exact match only - fuzzy clustering is opt-in).
        max_time_gap (Optional[float]): Maximum seconds between two readings
            for them to join the same crossing, on top of max_edit_distance.
            Only meaningful when max_edit_distance > 0. Defaults to None (no
            time constraint).

    Returns:
        Tuple[List[Dict[str, Any]], List[PlateDetection]]: A tuple containing:
            - List[Dict[str, Any]]: Dwell records with plate text (and, under
              fuzzy matching, every text variant seen), image paths,
              entry/exit timestamps, dwell time, and total sighting count.
            - List[PlateDetection]: Crossings with only one sighting, so no
              dwell time could be computed.
    """
    if max_edit_distance <= 0:
        matches, single_sightings = _group_by_exact_text(detections)
    else:
        matches, single_sightings = _group_by_fuzzy_text(
            detections, max_edit_distance, max_time_gap
        )

    matches.sort(key=lambda m: m["entry_time"])
    single_sightings.sort(key=lambda d: d.timestamp)

    return matches, single_sightings


def compute_average_dwell_time(
    matches: List[Dict[str, Any]], min_dwell_time: float = 0.0
) -> Tuple[float, int]:
    """
    Averages dwell time across matches, excluding any whose dwell_time falls
    below min_dwell_time.

    A short "crossing" is often an artifact - two OCR hits on what's really
    one sighting, or a vehicle that barely paused - rather than a genuine
    dwell, and would otherwise drag the average toward zero. Excluded
    matches aren't removed from anything else, only left out of this average;
    each match dict is annotated in place with a counted_in_average flag so
    callers (and the JSON report) can see exactly which ones were excluded.

    Args:
        matches (List[Dict[str, Any]]): Dwell records from
            compute_plate_dwell_times. Mutated in place: each gets a
            counted_in_average: bool key.
        min_dwell_time (float): Minimum dwell_time (in seconds) for a match
            to count toward the average. Defaults to 0.0 (no filtering).

    Returns:
        Tuple[float, int]: The average dwell time in seconds across the
            counted matches (0.0 if none qualify), and how many matches were
            excluded for falling below the threshold.
    """
    for match in matches:
        match["counted_in_average"] = match["dwell_time"] >= min_dwell_time

    counted = [m for m in matches if m["counted_in_average"]]
    excluded = len(matches) - len(counted)
    avg_dwell = sum(m["dwell_time"] for m in counted) / len(counted) if counted else 0.0

    return avg_dwell, excluded


def normalize_input_dirs(
    input_dirs: Union[str, Path, List[Union[str, Path]]],
) -> List[Path]:
    """
    Normalizes a single directory or a list of directories into a Path list.

    Args:
        input_dirs (Union[str, Path, List[Union[str, Path]]]): One directory,
            or a list of them (e.g. a separate entry-camera and exit-camera
            output directory).

    Returns:
        List[Path]: The input directories as Path objects, in the given order.
    """
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [input_dirs]
    return [Path(d) for d in input_dirs]


def run_plate_dwell_profiling(
    input_dirs: Union[str, Path, List[Union[str, Path]]],
    report: Optional[Union[str, Path]] = None,
    min_dwell_time: float = 0.0,
    max_edit_distance: int = 0,
    max_time_gap: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Plate-based dwell time pipeline.

    Args:
        input_dirs (Union[str, Path, List[Union[str, Path]]]): One or more
            directories of plate images, already processed by
            video_plateextractor.py, pooled into a single dwell-time analysis
            - e.g. a separate entry-camera output directory and exit-camera
            output directory, each with its own plate_manifest.json. Not
            split by entry/exit; a plate's earliest sighting across every
            given directory is its entry, its latest is its exit.
        report (Optional[Union[str, Path]]): Filepath to save summary JSON report. Defaults to None.
        min_dwell_time (float): Minimum dwell time (in seconds) for a matched
            crossing to count toward average_dwell_time - see
            compute_average_dwell_time. Matches below this are still included
            in dwell_time_matches, just excluded from the average. Defaults to
            0.0 (no filtering).
        max_edit_distance (int): Passed to compute_plate_dwell_times - see
            there. Defaults to 0 (exact match only).
        max_time_gap (Optional[float]): Passed to compute_plate_dwell_times -
            see there. Defaults to None (no time constraint).

    Returns:
        Dict[str, Any]: The summary report dict containing profiling results and stats.
    """
    input_folders = normalize_input_dirs(input_dirs)

    images: List[Path] = []
    manifest: Dict[Path, float] = {}
    for folder in input_folders:
        folder_images = find_images(folder)
        images.extend(folder_images)
        manifest.update(load_manifest(folder))
        logger.info("Found %d plate images in %s.", len(folder_images), folder)

    logger.info(
        "Found %d plate images total across %d director%s.",
        len(images),
        len(input_folders),
        "y" if len(input_folders) == 1 else "ies",
    )

    logger.info("Reading plates from images...")
    detections = extract_plate_detections(images, manifest=manifest)

    logger.info(
        "Successfully read plates from %d/%d images.", len(detections), len(images)
    )

    matches, single_sightings = compute_plate_dwell_times(
        detections, max_edit_distance=max_edit_distance, max_time_gap=max_time_gap
    )

    avg_dwell, excluded_from_average = compute_average_dwell_time(
        matches, min_dwell_time=min_dwell_time
    )

    summary_report: Dict[str, Any] = {
        "metadata": {
            "input_dirs": [str(folder) for folder in input_folders],
            "generated_at": datetime.datetime.now().isoformat(),
        },
        "statistics": {
            "input_images": len(images),
            "plates_read": len(detections),
            "matched_crossings": len(matches),
            "single_sightings": len(single_sightings),
            "min_dwell_time": min_dwell_time,
            "max_edit_distance": max_edit_distance,
            "max_time_gap": max_time_gap,
            "matches_excluded_from_average": excluded_from_average,
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
    if min_dwell_time > 0:
        print(
            f"Excluded from Average (dwell < {min_dwell_time:g}s): "
            f"{excluded_from_average}"
        )
    print(f"Average Dwell Time: {avg_dwell:.2f} seconds\n")

    if report:
        logger.info("Generating profiling report at %s...", report)
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    return summary_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute dwell time for vehicle crossings by matching OCR'd "
        "license plate text: a plate's earliest sighting is its entry, its "
        "latest sighting is its exit."
    )
    parser.add_argument(
        "input_dirs",
        type=str,
        nargs="+",
        metavar="input_dir",
        help="One or more directories of plate-cropped images (already "
        "processed by video_plateextractor.py) to pool together - e.g. a "
        "separate entry-camera output directory and exit-camera output "
        "directory, each with its own plate_manifest.json. Entry and exit "
        "aren't distinguished by source, only by which sighting of a plate "
        "comes first vs. last.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )
    parser.add_argument(
        "-m",
        "--min-dwell-time",
        type=float,
        default=0.0,
        help="Minimum dwell time in seconds for a matched crossing to count "
        "toward the average - shorter matches are still listed in the report, "
        "just excluded from the average. Defaults to 0.0 (no filtering).",
    )
    parser.add_argument(
        "-e",
        "--max-edit-distance",
        type=int,
        default=0,
        help="Maximum Levenshtein distance between two plate readings for "
        "them to be treated as the same vehicle - chained transitively. "
        "Defaults to 0 (exact match only, today's behavior). Set to 1 or 2 "
        "to absorb single-character OCR misreads that would otherwise split "
        "one real crossing into several.",
    )
    parser.add_argument(
        "-g",
        "--max-time-gap",
        type=float,
        default=None,
        help="Maximum seconds between two plate readings for them to be "
        "treated as the same vehicle, on top of --max-edit-distance. Only "
        "meaningful when --max-edit-distance > 0. Defaults to no limit.",
    )

    from src.utility.loggingutils import setup_logging_and_paths

    # setup_logging_and_paths only resolves/validates a singular args.input_dir;
    # this script takes one or more via args.input_dirs instead, so its
    # returned input_folder is unused and directories are validated here.
    args, _, _ = setup_logging_and_paths(parser, logger)

    input_folders = normalize_input_dirs(args.input_dirs)
    for folder in input_folders:
        if not folder.is_dir():
            logger.error(
                "Input directory '%s' does not exist or is not a directory.", folder
            )
            sys.exit(1)

    run_plate_dwell_profiling(
        input_dirs=input_folders,
        report=args.report,
        min_dwell_time=args.min_dwell_time,
        max_edit_distance=args.max_edit_distance,
        max_time_gap=args.max_time_gap,
    )


if __name__ == "__main__":
    main()
