"""
Vehicle Speed Benchmark

Benchmarks src/processing/video_entityprofiler.py's speed algorithm -
specifically compute_relative_speeds and calibrate_absolute_speeds, the two
functions that turn a raw pixel-displacement track into a real-world speed
from a single known reference vehicle - against the BrnoCompSpeed dataset
(Sochor et al., "Comprehensive Data Set for Automatic Single Camera Visual
Speed Measurement", IEEE T-ITS), which provides LiDAR-measured real-world
speeds for vehicles crossing two measurement lines in raw, unannotated
traffic camera footage.

process_video's own tracker can't run on this footage as-is: it locates
green bounding boxes painted onto this project's own recordings (see
imgutils.detect_entities), and BrnoCompSpeed obviously has no such overlay.
This module instead runs a general YOLO car/bus/truck detector through the
same IoU-matching, first-to-last-box pixel-displacement math process_video
uses (see track_vehicles), to produce the kind of track a real detector
would find. Those tracks are then handed to compute_relative_speeds and
calibrate_absolute_speeds completely unmodified - those two functions, not
the box-finding step, are what's actually under test: this validates the
"one reference vehicle's true speed linearly rescales every relative speed
into a real-world unit" assumption end to end, on real footage with real
ground truth. See video_entityprofiler.py's module docstring for that
assumption's rationale (a pure pixel-per-second ratio needs no camera
calibration - focal length, distance-to-road, lane width, etc.).

Ground truth is read directly from BrnoCompSpeed's own gt_data.pkl per
recording - a Python 2 pickle, see load_ground_truth. Access to the dataset
itself isn't self-service (BrnoCompSpeed's README asks you to email the
maintainers for a copy), so unlike vehicle_direction_benchmark.py's
auto-fetched weights, this expects an already-downloaded recording
directory (one session/camera-angle pair, e.g. "session1_center/",
containing a video file and gt_data.pkl side by side) rather than fetching
one itself.
"""

import argparse
import base64
import io
import json
import math
import pickle
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib

# Headless/non-interactive backend - this module generates chart images to a
# file or embeds them in HTML, it never shows a window, and the default
# backend would otherwise try (and fail) to open a display in a server/CI
# environment. Must be set before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from src.detection.image_yolo import load_model
from src.processing.video_entityprofiler import (
    VideoEntityRecord,
    calibrate_absolute_speeds,
    compute_relative_speeds,
)
from src.utility.geometryutils import Rectangle
from src.utility.htmlreport import html_heading, html_table, wrap_html_document

# COCO classes: 2=car, 5=bus, 7=truck. No "person" here (unlike
# video_yolo.py's DEFAULT_TARGET_CLASSES) - only vehicles have a
# BrnoCompSpeed ground-truth speed to score against.
DEFAULT_VEHICLE_CLASSES: Tuple[int, ...] = (2, 5, 7)

_VIDEO_EXTENSIONS: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv", ".webm")

_ACCURACY_BUCKETS: Tuple[str, ...] = ("<=5%", "<=20%", "<=50%", ">50%")

# Predicted-value color, reused from the same fixed identity every other
# benchmark in this project assigns to "pipeline output" (see
# direction_benchmark_common.PRED_COLOR / plate_dwell_benchmark._OCR_COLOR) -
# a scatter of (ground truth, predicted) pairs only has one series to color,
# and this is that series. The y=x reference line uses the same neutral
# separator gray already established for "this is a reference, not data" in
# direction_benchmark_common's bar chart.
_PRED_COLOR = "#e34948"
_REFERENCE_LINE_COLOR = "#c3c2b7"

_BANNER_WIDTH = 46
_LABEL_WIDTH = 28


def _stat_line(label: str, value: str) -> str:
    return f"{label:<{_LABEL_WIDTH}}: {value}"


@dataclass
class GroundTruthCar:
    """
    One BrnoCompSpeed ground-truth vehicle, as read from gt_data.pkl.

    Args:
        car_id (int): The car's id within this recording (gt_data.pkl's
            "carId" - not to be confused with a system/track id).
        valid (bool): Whether BrnoCompSpeed considers this car's ground
            truth reliable enough to score against. Invalid cars are still
            loaded (for completeness/debugging) but excluded from matching.
        speed_kmh (float): LiDAR-measured real-world speed, in km/h.
        first_crossing_time (float): Video time (seconds) the car crossed
            the first measurement line.
        last_crossing_time (float): Video time (seconds) the car crossed the
            second (last) measurement line - what BrnoCompSpeed's own
            evaluation code matches system output against.
    """

    car_id: int
    valid: bool
    speed_kmh: float
    first_crossing_time: float
    last_crossing_time: float


@dataclass
class GroundTruth:
    """
    A recording's full ground truth, as read from gt_data.pkl.

    Args:
        fps (float): The recording's frame rate, as reported by BrnoCompSpeed
            (used to convert its own frame-based timing into seconds -
            should agree with the source video's own fps).
        cars (List[GroundTruthCar]): Every annotated vehicle in this recording.
    """

    fps: float
    cars: List[GroundTruthCar]


def load_ground_truth(gt_path: Union[str, Path]) -> GroundTruth:
    """
    Loads a BrnoCompSpeed recording's gt_data.pkl.

    Only the fields BrnoCompSpeed's own evaluation code (eval.py) actually
    reads for matching/scoring are extracted - "carId", "valid", "speed",
    and "intersections" (each entry's "videoTime") per car, plus the
    recording's "fps". Camera calibration (vp1/vp2/pp/scale) and lane
    geometry aren't needed here: this project's speed algorithm has no
    camera calibration step to feed them into (see module docstring), so
    matching (match_tracks_to_ground_truth) is done by crossing time alone.

    Args:
        gt_path (Union[str, Path]): Path to a recording's gt_data.pkl.

    Returns:
        GroundTruth: The recording's fps and every annotated car.

    Raises:
        FileNotFoundError: If gt_path doesn't exist.
    """
    gt_path = Path(gt_path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    # gt_data.pkl was written by Python 2's cPickle - encoding="latin1" is
    # the standard way to unpickle those (numpy arrays included) under
    # Python 3 without mangling 8-bit string/bytes data. Harmless no-op for
    # a plain Python 3-written pickle (e.g. in tests).
    with open(gt_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    cars: List[GroundTruthCar] = []
    for car in raw["cars"]:
        intersections = car["intersections"]
        cars.append(
            GroundTruthCar(
                car_id=int(car["carId"]),
                valid=bool(car["valid"]),
                speed_kmh=float(car["speed"]),
                first_crossing_time=float(intersections[0]["videoTime"]),
                last_crossing_time=float(intersections[-1]["videoTime"]),
            )
        )

    return GroundTruth(fps=float(raw["fps"]), cars=cars)


def find_video_file(recording_dir: Union[str, Path]) -> Path:
    """
    Locates the single video file in a BrnoCompSpeed recording directory.

    Args:
        recording_dir (Union[str, Path]): Directory expected to contain
            exactly one video file (plus gt_data.pkl, mask/screenshot
            images, etc.).

    Returns:
        Path: The video file found.

    Raises:
        FileNotFoundError: If no video file is found.
        ValueError: If more than one video file is found (pass video_path
            explicitly to run_vehicle_speed_benchmark instead).
    """
    recording_dir = Path(recording_dir)
    candidates = sorted(
        p
        for p in recording_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"No video file found in {recording_dir}.")
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple video files found in {recording_dir}: "
            f"{[c.name for c in candidates]}. Pass video_path explicitly."
        )
    return candidates[0]


@dataclass
class VehicleTrack:
    """
    One tracked vehicle's raw speed measurement, produced by track_vehicles.

    Args:
        track_id (int): Unique id within this video (1-indexed, restarts per
            video - same convention as video_entityprofiler.Track).
        first_frame_idx (int): Frame index the track was first seen on.
        last_frame_idx (int): Frame index the track was last seen on.
        first_box (Rectangle): Bounding box the first time this vehicle was seen.
        last_box (Rectangle): Bounding box the last time this vehicle was seen.
        pixel_speed (float): Raw pixels/second - identical formula to
            process_video's pixel_speed (straight-line first-to-last-box
            center displacement over elapsed time).
    """

    track_id: int
    first_frame_idx: int
    last_frame_idx: int
    first_box: Rectangle
    last_box: Rectangle
    pixel_speed: float


@dataclass
class _Track:
    """Internal mutable tracking state while a video is being scanned."""

    track_id: int
    first_box: Rectangle
    first_frame_idx: int
    last_box: Rectangle
    last_frame_idx: int


def _detect_vehicles(
    model: YOLO, frame: np.ndarray, conf: float, classes: Sequence[int]
) -> List[Rectangle]:
    """
    Runs one frame through a YOLO model and returns every vehicle box found.
    """
    results = model.predict(
        source=frame, conf=conf, classes=list(classes), verbose=False
    )
    boxes: List[Rectangle] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append(Rectangle(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
    return boxes


def track_vehicles(
    video_path: Union[str, Path],
    model: YOLO,
    conf: float = 0.25,
    classes: Sequence[int] = DEFAULT_VEHICLE_CLASSES,
    downsample_factor: int = 1,
    iou_threshold: float = 0.15,
    max_lost_seconds: float = 2.0,
) -> Tuple[List[VehicleTrack], float]:
    """
    Scans a video with a general-purpose YOLO detector and tracks vehicles
    frame-to-frame by IoU, the same detect-then-greedily-match-by-IoU
    approach process_video uses for its green boxes (see that module) -
    just with a real object detector standing in for the color mask, since
    BrnoCompSpeed footage has no green-box overlay to key off of.

    Args:
        video_path (Union[str, Path]): Path to the video to scan.
        model (YOLO): A loaded general-purpose YOLO model (car/bus/truck
            classes, e.g. yolo26s.pt).
        conf (float): Minimum detection confidence. Defaults to 0.25.
        classes (Sequence[int]): COCO class ids to detect. Defaults to
            DEFAULT_VEHICLE_CLASSES (car/bus/truck).
        downsample_factor (int): Process every Nth frame, like
            process_video's downsample_factor. Defaults to 1.
        iou_threshold (float): Minimum IoU for a detection to match an
            existing track. Defaults to 0.15 (process_video's threshold).
        max_lost_seconds (float): Keep a track alive this many seconds after
            its last matched detection before giving up on it. Defaults to 2.0.

    Returns:
        Tuple[List[VehicleTrack], float]: Every track found (in id order),
            and the video's fps (needed to convert frame indices to seconds
            when matching against ground truth crossing times).

    Raises:
        RuntimeError: If the video file can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    max_lost_frames = int(fps * max_lost_seconds)
    active_tracks: Dict[int, _Track] = {}
    next_track_id = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % downsample_factor != 0:
            continue

        detections = _detect_vehicles(model, frame, conf, classes)

        match_candidates: List[Tuple[int, int, float]] = []
        for det_idx, box in enumerate(detections):
            for track_id, track in active_tracks.items():
                if frame_idx - track.last_frame_idx > max_lost_frames:
                    continue
                iou = Rectangle.compute_iou(box, track.last_box)
                if iou >= iou_threshold:
                    match_candidates.append((det_idx, track_id, iou))

        match_candidates.sort(key=lambda item: item[2], reverse=True)

        matched_dets = set()
        matched_tracks = set()
        for det_idx, track_id, _score in match_candidates:
            if det_idx in matched_dets or track_id in matched_tracks:
                continue
            matched_dets.add(det_idx)
            matched_tracks.add(track_id)
            track = active_tracks[track_id]
            track.last_box = detections[det_idx]
            track.last_frame_idx = frame_idx

        for det_idx, box in enumerate(detections):
            if det_idx not in matched_dets:
                active_tracks[next_track_id] = _Track(
                    track_id=next_track_id,
                    first_box=box,
                    first_frame_idx=frame_idx,
                    last_box=box,
                    last_frame_idx=frame_idx,
                )
                next_track_id += 1

    cap.release()

    tracks: List[VehicleTrack] = []
    for track in active_tracks.values():
        elapsed_frames = track.last_frame_idx - track.first_frame_idx
        elapsed_seconds = elapsed_frames / fps
        first_cx = track.first_box.x + track.first_box.w / 2
        first_cy = track.first_box.y + track.first_box.h / 2
        last_cx = track.last_box.x + track.last_box.w / 2
        last_cy = track.last_box.y + track.last_box.h / 2
        displacement_px = ((last_cx - first_cx) ** 2 + (last_cy - first_cy) ** 2) ** 0.5
        pixel_speed = displacement_px / elapsed_seconds if elapsed_seconds > 0 else 0.0

        tracks.append(
            VehicleTrack(
                track_id=track.track_id,
                first_frame_idx=track.first_frame_idx,
                last_frame_idx=track.last_frame_idx,
                first_box=track.first_box,
                last_box=track.last_box,
                pixel_speed=pixel_speed,
            )
        )

    return tracks, fps


@dataclass
class SpeedMatch:
    """
    Pairs one ground-truth car with the track judged to be it.

    Args:
        car_id (int): The ground-truth car's id.
        valid (bool): The ground-truth car's own valid flag.
        ground_truth_speed (float): The ground-truth car's real speed, km/h.
        track_id (int): The matched VehicleTrack's track_id.
        time_delta (float): |track's last-seen time - car's last crossing
            time|, in seconds - how good this match is.
    """

    car_id: int
    valid: bool
    ground_truth_speed: float
    track_id: int
    time_delta: float


def match_tracks_to_ground_truth(
    tracks: List[VehicleTrack],
    ground_truth: GroundTruth,
    fps: float,
    max_time_diff: float = 1.0,
) -> Tuple[List[SpeedMatch], int]:
    """
    Greedily pairs each ground-truth car with the closest not-yet-claimed
    track, by comparing the car's last measurement-line crossing time
    against each track's last-seen time.

    Mirrors BrnoCompSpeed's own eval.py matching approach (nearest by
    timeIntersectionLast, within a max time difference) with one
    simplification: no lane filtering. The official evaluation also
    requires a matched track's laneIndex to be one of the ground-truth
    car's plausible lanes, derived by projecting through camera calibration
    - this project's speed algorithm has no calibration step to derive lane
    geometry from (see module docstring), so time proximity alone decides
    matches here.

    Args:
        tracks (List[VehicleTrack]): Tracks from track_vehicles.
        ground_truth (GroundTruth): From load_ground_truth.
        fps (float): The video's fps, from track_vehicles - used to convert
            each track's last_frame_idx into seconds on the same time axis
            as ground_truth's crossing times.
        max_time_diff (float): Maximum seconds between a track's last-seen
            time and a car's last crossing time for them to count as a
            match. Defaults to 1.0.

    Returns:
        Tuple[List[SpeedMatch], int]: Every match found (ground-truth cars
            with no track within max_time_diff are dropped), and how many
            ground-truth cars went unmatched.
    """
    track_last_time = {t.track_id: t.last_frame_idx / fps for t in tracks}
    available = set(track_last_time)

    matches: List[SpeedMatch] = []
    unmatched = 0

    for car in sorted(ground_truth.cars, key=lambda c: c.last_crossing_time):
        candidates = sorted(
            available,
            key=lambda tid: abs(track_last_time[tid] - car.last_crossing_time),
        )
        if candidates:
            best = candidates[0]
            delta = abs(track_last_time[best] - car.last_crossing_time)
        else:
            delta = None

        if candidates and delta is not None and delta <= max_time_diff:
            available.discard(best)
            matches.append(
                SpeedMatch(
                    car_id=car.car_id,
                    valid=car.valid,
                    ground_truth_speed=car.speed_kmh,
                    track_id=best,
                    time_delta=delta,
                )
            )
        else:
            unmatched += 1

    return matches, unmatched


def _placeholder_records(
    tracks: List[VehicleTrack], video_path: Path
) -> Dict[int, VideoEntityRecord]:
    """
    Wraps each VehicleTrack in a minimal VideoEntityRecord so
    compute_relative_speeds/calibrate_absolute_speeds - which only look at
    entity_id/video_path/relative_speed - can run completely unmodified.
    Every other field is a placeholder; nothing here has (or needs) a real
    best frame/crop/histogram, since this benchmark never renders one.

    Args:
        tracks (List[VehicleTrack]): Tracks from track_vehicles.
        video_path (Path): Source video, for VideoEntityRecord.video_path.

    Returns:
        Dict[int, VideoEntityRecord]: track_id -> its placeholder record.
    """
    placeholder_frame = np.zeros((1, 1, 3), dtype=np.uint8)
    placeholder_hist = np.zeros((8, 8, 8))

    records: Dict[int, VideoEntityRecord] = {}
    for track in tracks:
        direction = "right" if track.last_box.x >= track.first_box.x else "left"
        records[track.track_id] = VideoEntityRecord(
            video_path=video_path,
            entity_id=track.track_id,
            best_frame_idx=track.last_frame_idx,
            best_frame=placeholder_frame,
            best_crop=placeholder_frame,
            best_box=track.last_box,
            timestamp=0.0,
            hsv_hist=placeholder_hist,
            aspect_ratio=1.0,
            direction=direction,
            relative_speed=track.pixel_speed,
        )
    return records


def compute_speed_accuracy(
    comparisons: List[Dict[str, Any]], tolerance: float = 0.2
) -> Dict[str, Any]:
    """
    Summarizes how close calibrate_absolute_speeds' predicted speeds landed
    to ground truth, across every scored car (see run_vehicle_speed_benchmark
    - the calibration reference car itself is never included here, since its
    error is trivially zero by construction).

    Args:
        comparisons (List[Dict[str, Any]]): Per-car dicts, each with
            "abs_error_kmh" and "relative_error" (the latter may be None if
            ground_truth_speed was 0).
        tolerance (float): Maximum relative error for a car to count as
            "reasonable". Defaults to 0.2 (20%).

    Returns:
        Dict[str, Any]: tolerance, total scored, mean/median absolute error
            (km/h), RMSE (km/h), mean relative error, reasonable count/rate,
            and a histogram of relative-error buckets (<=5%, <=20%, <=50%,
            >50%).
    """
    abs_errors = [c["abs_error_kmh"] for c in comparisons]
    rel_errors = [
        c["relative_error"] for c in comparisons if c["relative_error"] is not None
    ]

    counts: Dict[str, int] = {bucket: 0 for bucket in _ACCURACY_BUCKETS}
    for err in rel_errors:
        if err <= 0.05:
            counts["<=5%"] += 1
        elif err <= 0.20:
            counts["<=20%"] += 1
        elif err <= 0.50:
            counts["<=50%"] += 1
        else:
            counts[">50%"] += 1

    total_rel = len(rel_errors)
    histogram = {
        bucket: {"count": count, "share": count / total_rel if total_rel else 0.0}
        for bucket, count in counts.items()
    }
    reasonable_count = sum(1 for err in rel_errors if err <= tolerance)

    return {
        "tolerance": tolerance,
        "total_scored": len(comparisons),
        "mean_abs_error_kmh": statistics.mean(abs_errors) if abs_errors else 0.0,
        "median_abs_error_kmh": statistics.median(abs_errors) if abs_errors else 0.0,
        "rmse_kmh": (
            math.sqrt(sum(e**2 for e in abs_errors) / len(abs_errors))
            if abs_errors
            else 0.0
        ),
        "mean_relative_error": statistics.mean(rel_errors) if rel_errors else 0.0,
        "reasonable_count": reasonable_count,
        "reasonable_rate": reasonable_count / total_rel if total_rel else 0.0,
        "histogram": histogram,
    }


def render_speed_scatter_plot(comparisons: List[Dict[str, Any]]) -> bytes:
    """
    Renders a scatter plot of ground-truth vs predicted speed for every
    scored car, with a dashed y=x reference line - points on the line are
    exact predictions, points below/above it are under/over-estimates.

    Args:
        comparisons (List[Dict[str, Any]]): Per-car dicts with
            "ground_truth_speed" and "predicted_speed_kmh" (both km/h).

    Returns:
        bytes: PNG image data. Empty if there's nothing to plot.
    """
    if not comparisons:
        return b""

    gt_values = [c["ground_truth_speed"] for c in comparisons]
    pred_values = [c["predicted_speed_kmh"] for c in comparisons]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    axis_min = min(gt_values + pred_values) * 0.9
    axis_max = max(gt_values + pred_values) * 1.1
    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color=_REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=1.5,
        zorder=1,
        label="Perfect prediction",
    )

    ax.scatter(
        gt_values,
        pred_values,
        color=_PRED_COLOR,
        alpha=0.75,
        edgecolor=_PRED_COLOR,
        s=40,
        zorder=2,
        label="Scored vehicles",
    )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_xlabel("Ground truth speed (km/h)", color="#52514e")
    ax.set_ylabel("Predicted speed (km/h)", color="#52514e")
    ax.set_title(
        "Ground Truth vs Predicted Speed",
        color="#0b0b0b",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="#52514e")
    ax.grid(True, which="major", axis="both", color="#e3e2dd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="upper left", labelcolor="#0b0b0b")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _print_summary(
    ground_truth: GroundTruth,
    valid_matches: List[SpeedMatch],
    unmatched_gt: int,
    reference_match: SpeedMatch,
    accuracy: Dict[str, Any],
) -> None:
    print()
    print("=" * _BANNER_WIDTH)
    print("VEHICLE SPEED BENCHMARK".center(_BANNER_WIDTH))
    print("=" * _BANNER_WIDTH)
    print(_stat_line("Ground truth cars", str(len(ground_truth.cars))))
    print(_stat_line("  - valid", str(sum(1 for c in ground_truth.cars if c.valid))))
    print(_stat_line("  - matched to a track", str(len(valid_matches))))
    print(_stat_line("  - unmatched", str(unmatched_gt)))
    print(
        _stat_line(
            "Reference car (calibration)",
            f"car {reference_match.car_id} "
            f"({reference_match.ground_truth_speed:.2f} km/h true, "
            f"track {reference_match.track_id})",
        )
    )
    print(_stat_line("Scored cars", str(accuracy["total_scored"])))
    print("-" * _BANNER_WIDTH)
    print(_stat_line("Mean |error|", f"{accuracy['mean_abs_error_kmh']:.2f} km/h"))
    print(_stat_line("Median |error|", f"{accuracy['median_abs_error_kmh']:.2f} km/h"))
    print(_stat_line("RMSE", f"{accuracy['rmse_kmh']:.2f} km/h"))
    print(
        _stat_line(
            "Mean relative error", f"{accuracy['mean_relative_error'] * 100:.2f}%"
        )
    )
    print("=" * _BANNER_WIDTH)

    total = accuracy["total_scored"]
    tolerance_pct = accuracy["tolerance"] * 100
    print()
    print("SPEED ACCURACY".center(_BANNER_WIDTH, "-"))
    if total:
        value = (
            f"{accuracy['reasonable_count']}/{total}"
            f"  ({accuracy['reasonable_rate'] * 100:.2f}%)"
        )
    else:
        value = "n/a (no scored cars)"
    print(_stat_line(f"Reasonable (within {tolerance_pct:.0f}%)", value))
    for bucket in _ACCURACY_BUCKETS:
        entry = accuracy["histogram"][bucket]
        print(
            _stat_line(
                f"  {bucket} error", f"{entry['count']}  ({entry['share'] * 100:.2f}%)"
            )
        )
    print()


def render_html_report(summary_report: Dict[str, Any]) -> str:
    """
    Renders a vehicle speed benchmark summary_report as a standalone HTML
    page of tables, meant to be opened in a browser and copy-pasted
    (select-all, copy) directly into an Outlook email.

    Args:
        summary_report (Dict[str, Any]): The dict returned by
            run_vehicle_speed_benchmark.

    Returns:
        str: A complete HTML document.
    """
    stats = summary_report["statistics"]
    accuracy = summary_report["accuracy"]
    reference = summary_report["reference"]

    overview_table = html_table(
        ["Metric", "Value"],
        [
            ["Ground truth cars", stats["total_ground_truth_cars"]],
            ["  - valid", stats["valid_ground_truth_cars"]],
            ["  - matched to a track", stats["matched_cars"]],
            ["  - unmatched", stats["unmatched_cars"]],
            [
                "Reference car (calibration)",
                f"car {reference['car_id']} ({reference['ground_truth_speed']:.2f} km/h "
                f"true, track {reference['track_id']})",
            ],
            ["Scored cars", accuracy["total_scored"]],
            ["Mean |error|", f"{accuracy['mean_abs_error_kmh']:.2f} km/h"],
            ["Median |error|", f"{accuracy['median_abs_error_kmh']:.2f} km/h"],
            ["RMSE", f"{accuracy['rmse_kmh']:.2f} km/h"],
            ["Mean relative error", f"{accuracy['mean_relative_error'] * 100:.2f}%"],
        ],
    )

    total = accuracy["total_scored"]
    tolerance_pct = accuracy["tolerance"] * 100
    reasonable_value = (
        f"{accuracy['reasonable_count']}/{total} ({accuracy['reasonable_rate'] * 100:.2f}%)"
        if total
        else "n/a (no scored cars)"
    )
    accuracy_rows = [[f"Reasonable (within {tolerance_pct:.0f}%)", reasonable_value]]
    for bucket in _ACCURACY_BUCKETS:
        entry = accuracy["histogram"][bucket]
        accuracy_rows.append(
            [f"  {bucket} error", f"{entry['count']} ({entry['share'] * 100:.2f}%)"]
        )
    accuracy_table = html_table(["Metric", "Value"], accuracy_rows)

    comparisons_table = html_table(
        [
            "Car ID",
            "Ground Truth (km/h)",
            "Predicted (km/h)",
            "|Error| (km/h)",
            "Relative Error",
        ],
        [
            [
                c["car_id"],
                f"{c['ground_truth_speed']:.2f}",
                f"{c['predicted_speed_kmh']:.2f}",
                f"{c['abs_error_kmh']:.2f}",
                (
                    f"{c['relative_error'] * 100:.2f}%"
                    if c["relative_error"] is not None
                    else "-"
                ),
            ]
            for c in summary_report["comparisons"]
        ],
    )

    sections = [
        html_heading("Vehicle Speed Benchmark", level=2),
        overview_table,
        html_heading("Speed Accuracy", level=3),
        accuracy_table,
    ]

    scatter_png = render_speed_scatter_plot(summary_report["comparisons"])
    if scatter_png:
        scatter_b64 = base64.b64encode(scatter_png).decode("ascii")
        sections.append(html_heading("Ground Truth vs Predicted Speed", level=3))
        sections.append(
            f'<img src="data:image/png;base64,{scatter_b64}" '
            f'alt="Scatter plot of ground truth vs predicted speed per vehicle" '
            f'style="max-width:100%; height:auto;">'
        )

    sections.append(html_heading("Scored Vehicles", level=3))
    sections.append(comparisons_table)

    return wrap_html_document(sections)


def run_vehicle_speed_benchmark(
    recording_dir: Union[str, Path],
    video_path: Optional[Union[str, Path]] = None,
    gt_path: Optional[Union[str, Path]] = None,
    model_name: str = "yolo26s.pt",
    model: Optional[YOLO] = None,
    conf: float = 0.25,
    classes: Sequence[int] = DEFAULT_VEHICLE_CLASSES,
    downsample: int = 1,
    max_time_diff: float = 1.0,
    reference_car_id: Optional[int] = None,
    tolerance: float = 0.2,
    report: Optional[Union[str, Path]] = None,
    html_report: Optional[Union[str, Path]] = None,
    chart: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs video_entityprofiler.py's speed algorithm against a single
    BrnoCompSpeed recording and scores its accuracy.

    One ground-truth car (reference_car_id, or the lowest-numbered valid
    matched car if not given) is used as calibrate_absolute_speeds'
    reference - its predicted speed is trivially exact by construction, so
    it's excluded from every accuracy statistic; every other valid, matched
    car is scored against it.

    Args:
        recording_dir (Union[str, Path]): A BrnoCompSpeed recording
            directory (one session/camera-angle pair), containing a video
            file and gt_data.pkl.
        video_path (Optional[Union[str, Path]]): The recording's video file.
            Defaults to None, which auto-detects the one video file in
            recording_dir (see find_video_file).
        gt_path (Optional[Union[str, Path]]): Path to gt_data.pkl. Defaults
            to None, which looks for "gt_data.pkl" directly in recording_dir.
        model_name (str): YOLO weights to load if model isn't given. Defaults
            to "yolo26s.pt" (Ultralytics' standard pretrained COCO checkpoint,
            auto-downloaded on first use).
        model (Optional[YOLO]): Pre-loaded YOLO model. Defaults to None,
            which loads model_name.
        conf (float): Minimum detection confidence. Defaults to 0.25.
        classes (Sequence[int]): COCO class ids to detect. Defaults to
            DEFAULT_VEHICLE_CLASSES (car/bus/truck).
        downsample (int): Process every Nth frame. Defaults to 1. BrnoCompSpeed
            videos run up to an hour at 50fps, so a larger value trades
            tracking precision for a much faster run.
        max_time_diff (float): Passed to match_tracks_to_ground_truth.
            Defaults to 1.0 (seconds).
        reference_car_id (Optional[int]): Ground-truth car_id to use as the
            calibration reference. Defaults to None, which picks the
            lowest-numbered valid matched car.
        tolerance (float): Passed to compute_speed_accuracy. Defaults to 0.2
            (20%).
        report (Optional[Union[str, Path]]): Filepath to save a detailed JSON
            report. Defaults to None.
        html_report (Optional[Union[str, Path]]): Filepath to save an HTML
            version of the summary - see render_html_report. Defaults to None.
        chart (Optional[Union[str, Path]]): Filepath to save a standalone PNG
            of the ground-truth-vs-predicted scatter plot. Defaults to None.

    Returns:
        Dict[str, Any]: Summary report dict containing overall statistics,
            the calibration reference used, per-car comparisons, and
            accuracy metrics.

    Raises:
        ValueError: If no valid ground-truth car could be matched to a
            track, or reference_car_id doesn't match a valid matched car.
    """
    recording_dir = Path(recording_dir)
    gt_path = Path(gt_path) if gt_path is not None else recording_dir / "gt_data.pkl"
    ground_truth = load_ground_truth(gt_path)

    resolved_video_path = (
        Path(video_path) if video_path is not None else find_video_file(recording_dir)
    )

    if model is None:
        model = load_model(model_name)

    tracks, fps = track_vehicles(
        resolved_video_path,
        model,
        conf=conf,
        classes=classes,
        downsample_factor=downsample,
    )

    matches, unmatched_gt = match_tracks_to_ground_truth(
        tracks, ground_truth, fps, max_time_diff=max_time_diff
    )
    valid_matches = [m for m in matches if m.valid]

    if not valid_matches:
        raise ValueError(
            "No valid ground-truth car could be matched to a detected track - "
            "check the video/model/conf, or loosen max_time_diff."
        )

    if reference_car_id is None:
        reference_match = min(valid_matches, key=lambda m: m.car_id)
    else:
        candidates = [m for m in valid_matches if m.car_id == reference_car_id]
        if not candidates:
            raise ValueError(
                f"reference_car_id {reference_car_id} has no valid matched track."
            )
        reference_match = candidates[0]

    records = _placeholder_records(tracks, resolved_video_path)
    all_records = list(records.values())
    compute_relative_speeds(all_records)
    calibrate_absolute_speeds(
        all_records,
        reference_entity_id=reference_match.track_id,
        reference_speed=reference_match.ground_truth_speed,
    )

    comparisons: List[Dict[str, Any]] = []
    for m in valid_matches:
        if m.car_id == reference_match.car_id:
            continue
        predicted = records[m.track_id].absolute_speed
        assert predicted is not None
        abs_error = abs(predicted - m.ground_truth_speed)
        relative_error = (
            abs_error / m.ground_truth_speed if m.ground_truth_speed else None
        )
        comparisons.append(
            {
                "car_id": m.car_id,
                "track_id": m.track_id,
                "ground_truth_speed": m.ground_truth_speed,
                "predicted_speed_kmh": predicted,
                "abs_error_kmh": abs_error,
                "relative_error": relative_error,
                "time_delta": m.time_delta,
            }
        )

    accuracy = compute_speed_accuracy(comparisons, tolerance=tolerance)

    _print_summary(ground_truth, valid_matches, unmatched_gt, reference_match, accuracy)

    summary_report: Dict[str, Any] = {
        "metadata": {
            "recording_dir": str(recording_dir),
            "video_path": str(resolved_video_path),
            "gt_path": str(gt_path),
            "model_name": model_name,
            "conf": conf,
            "classes": list(classes),
            "downsample": downsample,
            "max_time_diff": max_time_diff,
            "fps": fps,
        },
        "statistics": {
            "total_ground_truth_cars": len(ground_truth.cars),
            "valid_ground_truth_cars": sum(1 for c in ground_truth.cars if c.valid),
            "matched_cars": len(valid_matches),
            "unmatched_cars": unmatched_gt,
            "total_tracks": len(tracks),
        },
        "reference": {
            "car_id": reference_match.car_id,
            "track_id": reference_match.track_id,
            "ground_truth_speed": reference_match.ground_truth_speed,
        },
        "comparisons": comparisons,
        "accuracy": accuracy,
    }

    if report:
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    if chart:
        with open(chart, "wb") as f:
            f.write(render_speed_scatter_plot(comparisons))

    if html_report:
        with open(html_report, "w") as f:
            f.write(render_html_report(summary_report))

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the vehicle speed benchmark script.

    Raises:
        SystemExit: If recording_dir is missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark src/processing/video_entityprofiler.py's speed "
        "algorithm (compute_relative_speeds/calibrate_absolute_speeds) against "
        "a BrnoCompSpeed recording directory (video + gt_data.pkl). Reports "
        "mean/median/RMSE error in km/h, mean relative error, and a "
        "within-tolerance accuracy breakdown."
    )
    parser.add_argument(
        "recording_dir",
        type=str,
        help="Path to a BrnoCompSpeed recording directory (e.g. "
        "'session1_center/'), containing a video file and gt_data.pkl.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to the recording's video file. Defaults to auto-detecting "
        "the one video file in recording_dir.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default=None,
        help="Path to gt_data.pkl. Defaults to 'gt_data.pkl' inside recording_dir.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights to use for vehicle detection (default: yolo26s.pt).",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.25,
        help="Minimum detection confidence (default: 0.25).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Process every Nth frame to trade tracking precision for speed "
        "(default: 1). BrnoCompSpeed videos run up to an hour at 50fps.",
    )
    parser.add_argument(
        "--max-time-diff",
        type=float,
        default=1.0,
        help="Maximum seconds between a track's last-seen time and a ground-"
        "truth car's crossing time for them to count as a match (default: 1.0).",
    )
    parser.add_argument(
        "--reference-car-id",
        type=int,
        default=None,
        help="Ground-truth car_id to use as the calibration reference. "
        "Defaults to the lowest-numbered valid matched car.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=0.2,
        help='Maximum relative error for a car to count as "reasonable" '
        "(default: 0.2, i.e. 20%%).",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save a detailed per-car JSON report.",
    )
    parser.add_argument(
        "-w",
        "--html",
        type=str,
        default=None,
        help="Path to save an HTML version of the summary. Open it in a "
        "browser and copy/paste the tables directly into an Outlook email. "
        "Includes the same scatter plot as --chart.",
    )
    parser.add_argument(
        "--chart",
        type=str,
        default=None,
        help="Path to save a standalone PNG scatter plot of ground-truth vs "
        "predicted speed.",
    )
    args = parser.parse_args()

    recording_dir = Path(args.recording_dir)
    if not recording_dir.is_dir():
        parser.error(f"{recording_dir} is not a directory.")

    run_vehicle_speed_benchmark(
        recording_dir=recording_dir,
        video_path=args.video,
        gt_path=args.gt,
        model_name=args.model,
        conf=args.conf,
        downsample=args.downsample,
        max_time_diff=args.max_time_diff,
        reference_car_id=args.reference_car_id,
        tolerance=args.tolerance,
        report=args.report,
        html_report=args.html,
        chart=args.chart,
    )


if __name__ == "__main__":
    main()
