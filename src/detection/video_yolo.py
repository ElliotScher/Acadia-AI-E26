"""
YOLO Video Object Detection Utility

Processes videos in an input directory using YOLO, maps target categories (e.g., bus/truck to car).
"""

import argparse
import datetime
import json
import logging
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from detection.classes import CLASS_ID_MAPPING

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from utility.geometryutils import Rectangle
from detection.classes import TARGET_CLASSES

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
DEFAULT_TARGET_CLASSES = TARGET_CLASSES

# --classes tokens that request license plate detection instead of a COCO class ID
PLATE_CLASS_TOKENS = {"plate", "license_plate", "license-plate"}

# Initialize Logger
logger = logging.getLogger("yolo_video_detection")


@dataclass
class DetectionResult:
    """
    Holds the YOLO detection results for a single video.

    Args:
        video_path (Path): Path to the source video.
        boxes (List[Tuple[int, Rectangle, int, float]]): List of (frame_idx, rectangle, class_id, confidence) detections.
    """

    video_path: Path
    boxes: List[
        Tuple[int, Rectangle, int, float]
    ]  # List of (frame_idx, rectangle, id, confidence)


def open_video_capture(video_path: Union[str, Path]) -> cv2.VideoCapture:
    """
    Opens a cv2.VideoCapture object.

    Args:
        video_path (Union[str, Path]): Path to the video file.

    Returns:
        cv2.VideoCapture: The opened VideoCapture object.
    """
    return cv2.VideoCapture(str(video_path))


def _frame_worker(
    model_name: str,
    frame_queue: queue.Queue,
    results_by_video: dict[Path, list[tuple[int, Rectangle, int, float]]],
    results_lock: threading.Lock,
    progress_bar: tqdm,
    inclusion_region: Optional[Rectangle],
    conf_threshold: float,
    classes_list: List[int],
    error_flag: threading.Event,
    device: str = "cpu",
    plate_model_name: Optional[str] = None,
    run_base_model: bool = True,
) -> None:
    """
    Worker thread that pulls frames from the queue, runs YOLO, and records detections.

    Args:
        model_name (str): YOLO weights name or path.
        frame_queue (queue.Queue): Frame task queue.
        results_by_video (Dict[Path, List[Tuple[int, Rectangle, int, float]]]): Shared results accumulator.
        results_lock (threading.Lock): Thread lock for safe writes.
        progress_bar (tqdm): Shared progress bar instance.
        inclusion_region (Optional[Rectangle]): Optional spatial filter.
        conf_threshold (float): Detection confidence threshold.
        classes_list (List[int]): COCO class filter list.
        error_flag (threading.Event): Error notification event.
        device (str): PyTorch device to run inference on.
        plate_model_name (Optional[str]): Optional path to a separate YOLO model checkpoint
            trained specifically for license plate detection. When provided, every frame is
            additionally run through this model and any detections are recorded with the
            label "license_plate". Defaults to None (license plate detection disabled).
        run_base_model (bool): Whether to run the general-purpose COCO model at all. False
            when the caller only requested license plate detection, so no COCO classes
            (person/car/etc.) are detected or drawn. Defaults to True.
    """
    thread_model = None
    if run_base_model:
        try:
            thread_model = YOLO(model_name)
        except Exception as e:
            logger.error(
                "Failed to load YOLO model '%s' in worker thread: %s", model_name, e
            )
            error_flag.set()
            return

    thread_plate_model = None
    if plate_model_name:
        try:
            thread_plate_model = YOLO(plate_model_name)
        except Exception as e:
            logger.error(
                "Failed to load license plate YOLO model '%s' in worker thread: %s",
                plate_model_name,
                e,
            )
            error_flag.set()
            return

    while not error_flag.is_set():
        try:
            task = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if task is None:
            frame_queue.task_done()
            break

        video_path, frame_idx, frame = task

        try:
            boxes_found: list[tuple[int, Rectangle, int, float]] = []

            if thread_model is not None:
                results = thread_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    classes=classes_list,
                    verbose=False,
                    device=device,
                )

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1

                        # Apply crop region filter if provided
                        if inclusion_region is not None:
                            box_rect = Rectangle(x1, y1, w, h)
                            if not Rectangle.bounding_box_intersects(
                                box_rect, inclusion_region
                            ):
                                continue

                        conf = float(box.conf[0])
                        rect = Rectangle(x=x1, y=y1, w=w, h=h)
                        boxes_found.append((frame_idx, rect, cls, conf))

            if thread_plate_model is not None:
                plate_results = thread_plate_model.predict(
                    source=frame,
                    conf=conf_threshold,
                    verbose=False,
                    device=device,
                )
                for r in plate_results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1

                        if inclusion_region is not None:
                            box_rect = Rectangle(x1, y1, w, h)
                            if not Rectangle.bounding_box_intersects(
                                box_rect, inclusion_region
                            ):
                                continue

                        conf = float(box.conf[0])
                        rect = Rectangle(x=x1, y=y1, w=w, h=h)
                        boxes_found.append((frame_idx, rect, -1, conf))

            if boxes_found:
                with results_lock:
                    results_by_video[video_path].extend(boxes_found)

        except Exception as e:
            logger.error(
                "Error predicting frame %d of video %s: %s", frame_idx, video_path, e
            )
        finally:
            if progress_bar is not None:
                progress_bar.update(1)
            frame_queue.task_done()


def process_videos(
    video_paths: List[Path],
    model_name: str,
    progress_bar: tqdm,
    inclusion_region: Optional[Rectangle] = None,
    conf_threshold: float = 0.5,
    target_classes: Optional[List[int]] = None,
    threads: int = 1,
    plate_model_name: Optional[str] = None,
    run_base_model: bool = True,
) -> List[DetectionResult]:
    """
    Processes a list of video paths using YOLO and extracts detection boxes.
    Uses frame-level multithreading with a producer-consumer model.

    Args:
        video_paths (List[Path]): List of video paths to process.
        model_name (str): YOLO weights name or path.
        progress_bar (tqdm): Thread-safe progress bar instance.
        inclusion_region (Optional[Rectangle]): Optional spatial filter region.
        conf_threshold (float): Minimum confidence threshold for detections.
        target_classes (Optional[List[int]]): List of COCO class IDs to filter.
        threads (int): Number of worker threads to spawn.
        plate_model_name (Optional[str]): Optional path to a separate YOLO model checkpoint
            trained specifically for license plate detection. When provided, detections from
            this model are included in the results labeled "license_plate". Defaults to None.
        run_base_model (bool): Whether to run the general-purpose COCO model at all. Defaults to True.

    Returns:
        List[DetectionResult]: List of detection results per video.
    """
    classes_list = (
        target_classes if target_classes is not None else DEFAULT_TARGET_CLASSES
    )
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    valid_video_paths = [
        vp for vp in video_paths if vp.suffix.lower() in video_extensions
    ]

    if not valid_video_paths:
        return []

    results_by_video: dict[Path, list[tuple[int, Rectangle, int, float]]] = {
        vp: [] for vp in valid_video_paths
    }
    results_lock = threading.Lock()
    error_flag = threading.Event()

    # Limit queue size to avoid high memory consumption from loading many frames in RAM
    frame_queue = queue.Queue(maxsize=threads * 4)

    # Spawn worker threads
    worker_threads = []
    for i in range(threads):
        device = "cpu"
        t = threading.Thread(
            target=_frame_worker,
            args=(
                model_name,
                frame_queue,
                results_by_video,
                results_lock,
                progress_bar,
                inclusion_region,
                conf_threshold,
                classes_list,
                error_flag,
                device,
                plate_model_name,
                run_base_model,
            ),
        )
        t.daemon = True
        t.start()
        worker_threads.append(t)

    try:
        # Producer: read frames from video files sequentially and queue them
        for video_path in valid_video_paths:
            if error_flag.is_set():
                break

            cap = open_video_capture(video_path)
            if not cap.isOpened():
                logger.error("Failed to open video %s", video_path)
                continue

            expected_frames = 0
            try:
                val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                expected_frames = int(val) if isinstance(val, (int, float)) else 0
            except (TypeError, ValueError):
                pass

            frame_idx = 0
            while not error_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                # Queue the frame. If queue is full, timeout to check thread health.
                while not error_flag.is_set():
                    if not any(t.is_alive() for t in worker_threads):
                        logger.error("All worker threads have died. Aborting.")
                        error_flag.set()
                        break
                    try:
                        frame_queue.put((video_path, frame_idx, frame), timeout=0.1)
                        break
                    except queue.Full:
                        continue

                frame_idx += 1

            cap.release()

            # If we read fewer frames than expected, update the progress bar for the missing frames
            if expected_frames > frame_idx and progress_bar is not None:
                progress_bar.update(expected_frames - frame_idx)

    except Exception as e:
        logger.error("Error during video frames generation: %s", e)
        error_flag.set()

    # Send termination to all worker threads
    for _ in range(threads):
        frame_queue.put(None)

    # Wait for all workers to finish
    for t in worker_threads:
        t.join()

    # Collect and format detection results, sorting boxes by frame index
    results_list: List[DetectionResult] = []
    for video_path in valid_video_paths:
        boxes = results_by_video[video_path]
        boxes.sort(key=lambda x: x[0])
        results_list.append(DetectionResult(video_path=video_path, boxes=boxes))

    return results_list


def save_annotated_videos(
    results: List[DetectionResult],
    input_folder: Path,
    output_folder: Path,
) -> None:
    """
    Saves detection results to disk with annotations.

    Args:
        results (List[DetectionResult]): YOLO detection outputs.
        input_folder (Path): Source directory for relative path resolution.
        output_folder (Path): Destination directory for saved videos.
    """
    # Auto-detect the best working encoder once before processing
    import os
    import tempfile

    codec_code = "avc1"
    os.environ["OPENCV_FFMPEG_WRITER_OPTIONS"] = "vcodec;libx264"

    temp_file = Path(tempfile.gettempdir()) / "test_codec.mp4"
    test_writer = cv2.VideoWriter(
        str(temp_file), cv2.VideoWriter.fourcc(*codec_code), 30.0, (100, 100)
    )
    if test_writer.isOpened():
        test_writer.release()
        try:
            temp_file.unlink()
        except OSError:
            pass
    else:
        codec_code = "mp4v"
        os.environ.pop("OPENCV_FFMPEG_WRITER_OPTIONS", None)
        logger.warning(
            "H.264 (avc1) codec not supported by OpenCV on this system. Falling back to mp4v."
        )

    fourcc = cv2.VideoWriter.fourcc(*codec_code)

    # Count total frames of all videos for the tqdm progress bar
    total_frames = 0
    for res in results:
        cap = open_video_capture(res.video_path)
        if cap.isOpened():
            try:
                val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                total_frames += int(val) if isinstance(val, (int, float)) else 0
            except (TypeError, ValueError):
                pass
            cap.release()

    progress_bar = tqdm(
        total=total_frames, desc="Saving annotated videos", unit="frame"
    )

    for res in results:
        video_path = res.video_path
        boxes = res.boxes

        out_path = output_folder / video_path.relative_to(input_folder)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = open_video_capture(video_path)
        if not cap.isOpened():
            logger.error(
                "Failed to open input video for saving annotations: %s", video_path
            )
            continue

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        # Group boxes by frame index for quick lookup
        boxes_by_frame: Dict[int, List[Tuple[Rectangle, int, float]]] = {}
        for frame_idx, rect, coco_id, conf in boxes:
            if frame_idx not in boxes_by_frame:
                boxes_by_frame[frame_idx] = []
            boxes_by_frame[frame_idx].append((rect, coco_id, conf))

        frame_idx = 0
        expected_frames = 0
        try:
            val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            expected_frames = int(val) if isinstance(val, (int, float)) else 0
        except (TypeError, ValueError):
            pass

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in boxes_by_frame:
                # Draw rectangles on the frame - every detected class (including
                # license plates) is drawn the same way, since only the classes
                # actually requested via --classes ever appear here.
                for rect, coco_id, conf in boxes_by_frame[frame_idx]:
                    cv2.rectangle(
                        frame,
                        (rect.x, rect.y),
                        (rect.x + rect.w, rect.y + rect.h),
                        (0, 255, 0),
                        thickness=5,
                    )

            out.write(frame)
            frame_idx += 1
            progress_bar.update(1)

        cap.release()
        out.release()

        # If we read fewer frames than expected, update the progress bar for the missing frames
        if expected_frames > frame_idx:
            progress_bar.update(expected_frames - frame_idx)

        logger.debug("Saved annotated video: %s", out_path)

    progress_bar.close()
    os.environ.pop("OPENCV_FFMPEG_WRITER_OPTIONS", None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process videos in an input directory using YOLO and save results to an output directory."
    )
    parser.add_argument("input_dir", type=str, help="Path to the input directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolo26s.pt",
        help="YOLO model weights to use (default: yolo26s.pt).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of CPU threads to allocate to YOLO detections (default: 1).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even if GPUs are available (force CPU).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions (default: 0.25).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of classes to detect: COCO category IDs (e.g. "
        "'0,2,5,7') and/or the special value 'plate' for license plates (requires "
        "--plate-model). Only the classes listed here are detected/drawn - e.g. "
        "'--classes plate' alone detects license plates only, with no COCO "
        "detection running at all. Defaults to person/car/bus/truck if omitted.",
    )
    parser.add_argument(
        "--inclusion-region",
        type=str,
        default=None,
        help="Inclusion region as 'x,y,w,h' in pixels (default: None).",
    )
    parser.add_argument(
        "--plate-model",
        type=str,
        default=None,
        help="Path to a YOLO model checkpoint trained specifically for license plate "
        "detection. Required when 'plate' is included in --classes; ultralytics only "
        "auto-downloads its own general-purpose weights, so a plate-specific "
        "model must be supplied.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save the summary report in JSON format.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated videos.",
    )
    from src.utility.loggingutils import setup_logging_and_paths

    args, input_folder, output_folder = setup_logging_and_paths(parser, logger)
    assert input_folder is not None and output_folder is not None

    output_folder.mkdir(parents=True, exist_ok=True)

    if args.threads <= 0:
        logger.error("The number of allocated CPU threads must be at least 1.")
        sys.exit(1)
    thread_count = args.threads

    # --classes selects everything this run detects: COCO category IDs run through
    # the general-purpose model, and the special 'plate' token runs the plate
    # model (via --plate-model). If --classes is omitted entirely, the old
    # default (person/car/bus/truck, no plates) is preserved. If --classes is
    # given and only contains 'plate', the COCO model doesn't run at all, so
    # nothing but license plates is detected or drawn.
    target_classes = DEFAULT_TARGET_CLASSES
    plate_requested = False
    run_base_model = True

    if args.classes:
        coco_tokens = []
        for raw_token in args.classes.split(","):
            token = raw_token.strip()
            if not token:
                continue
            if token.lower() in PLATE_CLASS_TOKENS:
                plate_requested = True
            else:
                coco_tokens.append(token)

        try:
            target_classes = [int(t) for t in coco_tokens]
        except ValueError:
            logger.error(
                "Invalid class list format: '%s'. Expected comma-separated COCO "
                "category IDs and/or 'plate'.",
                args.classes,
            )
            sys.exit(1)

        run_base_model = len(target_classes) > 0

    if plate_requested and not args.plate_model:
        logger.error(
            "The 'plate' class requires --plate-model to specify a license-plate-"
            "detection YOLO checkpoint."
        )
        sys.exit(1)
    plate_model_name = args.plate_model if plate_requested else None

    inclusion_region = None
    if args.inclusion_region:
        try:
            parts = [int(x.strip()) for x in args.inclusion_region.split(",")]
            if len(parts) == 4:
                inclusion_region = Rectangle(parts[0], parts[1], parts[2], parts[3])
            else:
                logger.error(
                    "Invalid inclusion region format: '%s'. Expected 'x,y,w,h'.",
                    args.inclusion_region,
                )
                sys.exit(1)
        except ValueError:
            logger.error(
                "Invalid inclusion region coordinates in '%s'. Expected integers.",
                args.inclusion_region,
            )
            sys.exit(1)

    total_counts: Dict[str, int] = {}
    if run_base_model:
        try:
            model = YOLO(args.model)
            for cls in target_classes:
                if cls in model.names:
                    label = model.names[cls]
                    if label in ["car", "bus", "truck"]:
                        label = "car"
                    total_counts[label] = 0
                else:
                    logger.warning("Class ID %d not found in model names.", cls)
        except Exception as e:
            logger.error("Error loading model '%s': %s", args.model, e)
            sys.exit(1)

    if plate_model_name:
        total_counts["license_plate"] = 0

    # Find videos
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    all_videos = [
        p
        for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in video_extensions
    ]

    if not all_videos:
        logger.warning("No matching video files found in the input directory.")
        return

    # Count total frames of all videos for the tqdm progress bar
    total_frames = 0
    for video_path in all_videos:
        cap = open_video_capture(video_path)
        if cap.isOpened():
            try:
                val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frames = int(val) if isinstance(val, (int, float)) else 0
            except (TypeError, ValueError):
                frames = 0
            total_frames += frames
            cap.release()
        else:
            logger.error("Failed to open video %s to count frames.", video_path)

    progress_bar = tqdm(total=total_frames, desc="Progress", unit="frame")

    all_results = process_videos(
        video_paths=all_videos,
        model_name=args.model,
        progress_bar=progress_bar,
        inclusion_region=inclusion_region,
        conf_threshold=args.conf,
        target_classes=target_classes,
        threads=thread_count,
        plate_model_name=plate_model_name,
        run_base_model=run_base_model,
    )

    progress_bar.close()

    if not args.no_save:
        logger.info("Saving annotated videos to output directory...")
        save_annotated_videos(
            all_results,
            input_folder,
            output_folder,
        )
    else:
        logger.info("Skipping saving annotated videos as requested.")

    detection_details: Dict[str, List[Dict[str, Union[int, List[int], float]]]] = (
        {}
    )
    for res in all_results:
        relative_key = str(res.video_path.relative_to(input_folder))
        file_detections = []
        for frame_idx, rect, coco_id, conf in res.boxes:
            label = CLASS_ID_MAPPING[coco_id]
            if label not in total_counts:
                total_counts[label] = 0
            total_counts[label] += 1

            file_detections.append(
                {
                    "frame_index": frame_idx,
                    "box": [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
                    "label": coco_id,
                    "confidence": conf,
                }
            )
        detection_details[relative_key] = file_detections

    print("\n--- Summary ---")
    for category, count in total_counts.items():
        print(f"Total {category} detected: {count}")
    print("YOLO video processing complete!\n")

    if args.report:
        logger.info("Generating detection report at %s...", args.report)
        report_data = {
            "metadata": {
                "input_dir": str(input_folder),
                "output_dir": str(output_folder),
                "model_weights": args.model,
                "confidence_threshold": args.conf,
                "target_classes": target_classes,
                "plate_model_weights": plate_model_name,
                "total_videos_processed": len(all_videos),
                "generated_at": datetime.datetime.now().isoformat(),
            },
            "statistics": total_counts,
            "detections": detection_details,
        }

        try:
            with open(args.report, "w") as f:
                json.dump(report_data, f, indent=4)
        except Exception as e:
            logger.error("Failed to save report to %s: %s", args.report, e)


if __name__ == "__main__":
    main()
