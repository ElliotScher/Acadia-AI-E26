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

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLO

from utility.geometryutils import Rectangle

# COCO classes: 0=person, 2=car, 5=bus, 7=truck
DEFAULT_TARGET_CLASSES = [0, 2, 5, 7]

# Initialize Logger
logger = logging.getLogger("yolo_video_detection")


@dataclass
class DetectionResult:
    """
    Holds the YOLO detection results for a single video.

    Args:
        video_path (Path): Path to the source video.
        boxes (List[Tuple[int, Rectangle, str, float]]): List of (frame_idx, rectangle, label, confidence) detections.
    """

    video_path: Path
    boxes: List[
        Tuple[int, Rectangle, str, float]
    ]  # List of (frame_idx, rectangle, label, confidence)


def detect_gpus() -> List[str]:
    """
    Detects all available GPUs on the system.

    Returns:
        List[str]: Hardcoded to return an empty list as GPU support has been disabled.
    """
    return []


def open_video_capture(
    video_path: Union[str, Path], use_gpu: bool = False
) -> cv2.VideoCapture:
    """
    Opens a cv2.VideoCapture object.

    Args:
        video_path (Union[str, Path]): Path to the video file.
        use_gpu (bool): Optional GPU decode flag (ignored).

    Returns:
        cv2.VideoCapture: The opened VideoCapture object.
    """
    return cv2.VideoCapture(str(video_path))


def _frame_worker(
    model_name: str,
    frame_queue: queue.Queue,
    results_by_video: Dict[Path, List[Tuple[int, Rectangle, str, float]]],
    results_lock: threading.Lock,
    progress_bar: tqdm,
    inclusion_region: Optional[Rectangle],
    conf_threshold: float,
    classes_list: List[int],
    error_flag: threading.Event,
    device: str = "cpu",
) -> None:
    """
    Worker thread that pulls frames from the queue, runs YOLO, and records detections.

    Args:
        model_name (str): YOLO weights name or path.
        frame_queue (queue.Queue): Frame task queue.
        results_by_video (Dict[Path, List[Tuple[int, Rectangle, str, float]]]): Shared results accumulator.
        results_lock (threading.Lock): Thread lock for safe writes.
        progress_bar (tqdm): Shared progress bar instance.
        inclusion_region (Optional[Rectangle]): Optional spatial filter.
        conf_threshold (float): Detection confidence threshold.
        classes_list (List[int]): COCO class filter list.
        error_flag (threading.Event): Error notification event.
        device (str): PyTorch device to run inference on.
    """
    try:
        thread_model = YOLO(model_name)
    except Exception as e:
        logger.error(
            "Failed to load YOLO model '%s' in worker thread: %s", model_name, e
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
            results = thread_model.predict(
                source=frame,
                conf=conf_threshold,
                classes=classes_list,
                verbose=False,
                device=device,
            )

            boxes_found = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = thread_model.names[cls]

                    # Standardize transport categories to 'car'
                    if label in ["car", "bus", "truck"]:
                        label = "car"

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
                    boxes_found.append((frame_idx, rect, label, conf))

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
    cores: int = 1,
    gpu_devices: Optional[List[str]] = None,
    use_gpu_decode: bool = False,
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
        cores (int): Number of worker threads to spawn.
        gpu_devices (Optional[List[str]]): List of GPU device strings to distribute workers across.
        use_gpu_decode (bool): Whether to attempt GPU hardware-accelerated video decoding.

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

    results_by_video = {vp: [] for vp in valid_video_paths}
    results_lock = threading.Lock()
    error_flag = threading.Event()

    # Limit queue size to avoid high memory consumption from loading many frames in RAM
    frame_queue = queue.Queue(maxsize=cores * 4)

    # Spawn worker threads
    worker_threads = []
    for i in range(cores):
        device = gpu_devices[i % len(gpu_devices)] if gpu_devices else "cpu"
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

            cap = open_video_capture(video_path, use_gpu=use_gpu_decode)
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

    # Send termination sentinels to all worker threads
    for _ in range(cores):
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
    use_gpu_decode: bool = False,
) -> None:
    """
    Saves detection results to disk with annotations.

    Args:
        results (List[DetectionResult]): YOLO detection outputs.
        input_folder (Path): Source directory for relative path resolution.
        output_folder (Path): Destination directory for saved videos.
        use_gpu_decode (bool): Whether to attempt GPU hardware-accelerated video decoding.
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
        cap = open_video_capture(res.video_path, use_gpu=use_gpu_decode)
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

        cap = open_video_capture(video_path, use_gpu=use_gpu_decode)
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
        boxes_by_frame: Dict[int, List[Tuple[Rectangle, str, float]]] = {}
        for frame_idx, rect, label, conf in boxes:
            if frame_idx not in boxes_by_frame:
                boxes_by_frame[frame_idx] = []
            boxes_by_frame[frame_idx].append((rect, label, conf))

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
                # Draw rectangles on the frame
                for rect, label, conf in boxes_by_frame[frame_idx]:
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


# thread_worker removed as processing is now parallelized at the frame level


def main() -> None:
    """
    Main CLI entry point for YOLO video detection script.
    """
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
        "-c",
        "--cores",
        type=int,
        default=1,
        help="Number of CPU cores to allocate to YOLO detections (default: 1).",
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
        help="Comma-separated list of COCO category IDs to detect (e.g., '0,2,5,7').",
    )
    parser.add_argument(
        "--inclusion-region",
        type=str,
        default=None,
        help="Inclusion region as 'x,y,w,h' in pixels (default: None).",
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

    output_folder.mkdir(parents=True, exist_ok=True)

    if args.cores <= 0:
        logger.error("The number of allocated CPU cores/threads must be at least 1.")
        sys.exit(1)
    thread_count = args.cores

    # Determine GPU availability
    gpu_devices = []
    if not args.no_gpu:
        gpu_devices = detect_gpus()

    use_gpu_decode = len(gpu_devices) > 0

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    logger.info("Allocating %d thread(s) for video processing...", thread_count)
    if gpu_devices:
        logger.info("GPU acceleration enabled. Detected GPU(s): %s", gpu_devices)
    else:
        logger.info("GPU acceleration disabled or not found. Using CPU.")

    target_classes = DEFAULT_TARGET_CLASSES
    if args.classes:
        try:
            target_classes = [int(x.strip()) for x in args.classes.split(",")]
        except ValueError:
            logger.error(
                "Invalid class list format: '%s'. Expected comma-separated integers.",
                args.classes,
            )
            sys.exit(1)

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
        cap = open_video_capture(video_path, use_gpu=use_gpu_decode)
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
        cores=thread_count,
        gpu_devices=gpu_devices,
        use_gpu_decode=use_gpu_decode,
    )

    progress_bar.close()

    if not args.no_save:
        logger.info("Saving annotated videos to output directory...")
        save_annotated_videos(
            all_results,
            input_folder,
            output_folder,
            use_gpu_decode=use_gpu_decode,
        )
    else:
        logger.info("Skipping saving annotated videos as requested.")

    detection_details: Dict[str, List[Dict[str, Union[int, List[int], str, float]]]] = (
        {}
    )
    for res in all_results:
        relative_key = str(res.video_path.relative_to(input_folder))
        file_detections = []
        for frame_idx, rect, label, conf in res.boxes:
            if label not in total_counts:
                total_counts[label] = 0
            total_counts[label] += 1

            file_detections.append(
                {
                    "frame_index": frame_idx,
                    "box": [rect.x, rect.y, rect.x + rect.w, rect.y + rect.h],
                    "label": label,
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
