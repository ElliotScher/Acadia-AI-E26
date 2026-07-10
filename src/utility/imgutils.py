import datetime
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional, Union

import cv2
import numpy as np

# Sidecar filename written by video_plateextractor.py next to its cropped plate
# images, mapping each crop back to a timestamp resolved from its source frame
# before cropping (a tight plate crop no longer contains the burned-in on-screen
# timestamp text get_timestamp's OCR step relies on). Shared here so
# video_plateextractor.py (writer) and plate_dwellprofiler.py (reader) agree on
# the same filename without duplicating the constant.
PLATE_MANIFEST_FILENAME = "plate_manifest.json"

# Bounds an OCR-read year must fall within to be trusted. Tesseract occasionally
# misreads a single digit of the on-screen date (e.g. "2026" -> "5026"), which
# would otherwise produce a wildly implausible timestamp far in the past/future.
_MIN_PLAUSIBLE_OCR_YEAR = 2000
_MAX_PLAUSIBLE_OCR_YEAR = 2099


def _plausible_ocr_year(year: int) -> int:
    """
    Returns `year` unchanged if it falls within a plausible range, otherwise
    substitutes the current year.

    A single misread digit in the on-screen date's year (e.g. "2026" -> "5026")
    otherwise corrupts an entire timestamp even though the rest of the OCR'd
    date/time is trustworthy, so the year is corrected rather than discarding
    the whole reading.

    Args:
        year (int): The OCR-parsed year value.

    Returns:
        int: `year` if plausible, otherwise the current year.
    """
    if _MIN_PLAUSIBLE_OCR_YEAR <= year <= _MAX_PLAUSIBLE_OCR_YEAR:
        return year
    return datetime.datetime.now().year


def _parse_timestamp_from_ocr_text(text: str) -> Optional[float]:
    """
    Searches OCR'd text for a date/time pattern and returns it as an epoch timestamp.

    Tries, in order: YYYY-MM-DD HH:MM:SS (or YYYY/MM/DD), MM/DD/YYYY or DD/MM/YYYY
    HH:MM:SS, and a bare HH:MM:SS fallback (returned as raw seconds-since-midnight,
    not a real epoch time, since no date is available).

    Args:
        text (str): Raw text returned by pytesseract for a single OCR attempt.

    Returns:
        Optional[float]: The parsed timestamp, or None if no pattern matched.
    """
    # 1. YYYY-MM-DD HH:MM:SS (or YYYY/MM/DD)
    pattern1 = r"(\d{4})[-/](\d{2})[-/](\d{2})\s*(\d{2})[:\.](\d{2})[:\.](\d{2})"
    match1 = re.search(pattern1, text)
    if match1:
        year, month, day, hour, minute, second = map(int, match1.groups())
        year = _plausible_ocr_year(year)
        try:
            return datetime.datetime(year, month, day, hour, minute, second).timestamp()
        except ValueError:
            pass

    # 2. MM/DD/YYYY HH:MM:SS or DD/MM/YYYY HH:MM:SS
    pattern2 = r"(\d{2})[-/](\d{2})[-/](\d{4})\s*(\d{2})[:\.](\d{2})[:\.](\d{2})"
    match2 = re.search(pattern2, text)
    if match2:
        val1, val2, year, hour, minute, second = map(int, match2.groups())
        year = _plausible_ocr_year(year)
        # Try MM/DD/YYYY first
        try:
            return datetime.datetime(year, val1, val2, hour, minute, second).timestamp()
        except ValueError:
            # Fall back to DD/MM/YYYY
            try:
                return datetime.datetime(
                    year, val2, val1, hour, minute, second
                ).timestamp()
            except ValueError:
                pass

    # 3. Simple time fallback: HH:MM:SS
    pattern3 = r"(\d{2})[:\.](\d{2})[:\.](\d{2})"
    match3 = re.search(pattern3, text)
    if match3:
        hour, minute, second = map(int, match3.groups())
        return float(hour * 3600 + minute * 60 + second)

    return None


def extract_timestamp_via_ocr(img_path: Path) -> Optional[float]:
    """
    Attempts to extract a timestamp from the image pixels using pytesseract OCR.

    Looks for date/time patterns like YYYY-MM-DD HH:MM:SS or MM/DD/YYYY HH:MM:SS.
    The time is expected to always be within the bottom 10% of the image.

    Args:
        img_path (Path): Path to the input image file.

    Returns:
        Optional[float]: The extracted timestamp in seconds since epoch if successful,
            or None if OCR extraction fails.
    """
    try:
        import pytesseract
        from PIL import Image
        import shutil

    except ImportError:
        return None

    try:
        img = Image.open(img_path)

        # If the image resolution is high (e.g., 4K), resize it down to 1920 width
        # to match Tesseract's expected font scale/DPI and dramatically improve OCR accuracy.
        if img.width > 1920:
            aspect = img.height / img.width
            img = img.resize((1920, int(1920 * aspect)))

        # Preprocessing: convert to grayscale
        gray = img.convert("L")

        # Crop ONLY the bottom 10% of the image as the time is always printed there
        w, h = img.size
        bottom_crop = gray.crop((0, int(h * 0.90), w, h))

        # Binarizing (thresholding to pure black-and-white) helps Tesseract read
        # frames where it otherwise misses the on-screen date/time entirely, but
        # it isn't strictly better on every frame - it can turn an already-clean
        # read into a noisier one. So it's only tried as a fallback crop variant,
        # after the original grayscale crop has failed at every config.
        bottom_arr = np.array(bottom_crop)
        binarized_crop = Image.fromarray((bottom_arr > 140).astype(np.uint8) * 255)

        for crop in (bottom_crop, binarized_crop):
            for config in [None, "--psm 6"]:
                if config:
                    text = pytesseract.image_to_string(crop, config=config)
                else:
                    text = pytesseract.image_to_string(crop)

                ts = _parse_timestamp_from_ocr_text(text)
                if ts is not None:
                    return ts

    except Exception:
        # Gracefully ignore OCR errors
        pass
    return None


_PLATE_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _normalize_plate_text(text: str) -> str:
    """
    Uppercases OCR'd plate text and strips everything but letters and digits.

    Args:
        text (str): Raw plate text returned by the plate OCR model.

    Returns:
        str: The normalized, alphanumeric-only plate text (may be empty).
    """
    return "".join(ch for ch in text.upper() if ch in _PLATE_CHAR_WHITELIST)


_PLATE_OCR_MODEL_NAME = "cct-s-v2-global-model"
_plate_recognizer = None


def _get_plate_recognizer():
    """
    Lazily loads and caches the shared fast-plate-ocr LicensePlateRecognizer.

    Loading it (including a one-time model download to a local cache on first
    use) is too expensive to repeat for every plate crop, so it's loaded once
    per process and reused.

    Returns:
        LicensePlateRecognizer: The cached recognizer instance.
    """
    global _plate_recognizer
    if _plate_recognizer is None:
        from fast_plate_ocr import LicensePlateRecognizer

        _plate_recognizer = LicensePlateRecognizer(_PLATE_OCR_MODEL_NAME)
    return _plate_recognizer


def extract_plate_text_via_ocr(crop: np.ndarray, min_length: int = 3) -> Optional[str]:
    """
    Attempts to read a license plate's alphanumeric text from a BGR crop.

    Uses fast-plate-ocr's purpose-trained plate recognition model rather than
    generic document OCR (e.g. Tesseract) - real plates combine embossed
    fonts, decorative state graphics, and (from a distant/handheld camera)
    low source resolution, which generic OCR handles very poorly even after
    upscaling; a model actually trained on plate crops reads them reliably.

    Args:
        crop (np.ndarray): OpenCV BGR crop of the plate region.
        min_length (int): Minimum number of alphanumeric characters required for a
            reading to be trusted. Shorter results are almost always noise from
            an unreadable plate rather than a genuine (if short) plate. Defaults to 3.

    Returns:
        Optional[str]: The normalized (uppercase, alphanumeric-only) plate text, or
            None if OCR failed to produce a plausible reading.
    """
    if crop is None or crop.size == 0:
        return None

    try:
        recognizer = _get_plate_recognizer()
        results = recognizer.run(crop)
        if not results:
            return None

        plate_text = _normalize_plate_text(results[0].plate)
        if len(plate_text) >= min_length:
            return plate_text
    except Exception:
        # Gracefully ignore OCR errors
        pass

    return None


def get_center_crop(img: np.ndarray, margin_pct: float) -> np.ndarray:
    """
    Extracts the center portion of an image crop to focus on the vehicle's body,
    reducing background noise and foliage.

    Args:
        img (np.ndarray): OpenCV BGR image.
        margin_pct (float): Percentage of margin to crop from all sides.

    Returns:
        np.ndarray: The cropped image region.
    """
    h, w = img.shape[:2]
    dy = int(h * margin_pct)
    dx = int(w * margin_pct)
    crop = img[dy : h - dy, dx : w - dx]
    if crop.size > 0:
        return crop
    return img


def get_hsv_hist(crop: np.ndarray) -> np.ndarray:
    """
    Computes a normalized 3D HSV histogram for a crop.

    Used to compare the color distribution of the entity across images.
    Assumes the input crop has already been center-cropped if desired.

    Args:
        crop (np.ndarray): OpenCV BGR image crop of the entity.

    Returns:
        np.ndarray: A normalized 8x8x8 HSV histogram array.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Calculate 3D histogram: 8 bins for H, 8 for S, 8 for V
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compute_sharpness(img: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian to evaluate image focus/sharpness.

    Args:
        img (np.ndarray): BGR image crop.

    Returns:
        float: Variance of the Laplacian (higher means sharper).
    """
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_entities(img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects green bounding boxes in the image and returns a list containing
    at most the largest box (x, y, w, h).

    Uses color-masking rules to filter out foliage while extracting boxes.

    Args:
        img (np.ndarray): OpenCV BGR image.

    Returns:
        List[Tuple[int, int, int, int]]: A list containing the bounding box
            coordinates (x, y, w, h) if found, otherwise an empty list.
    """
    # Strict green filter in BGR: B < 50, G > 180, R < 50
    mask = (img[:, :, 0] < 50) & (img[:, :, 1] > 180) & (img[:, :, 2] < 50)
    mask = mask.astype(np.uint8) * 255

    # Use RETR_EXTERNAL to find only the outer boundaries of contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    h_img, w_img = img.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out noise (too small) or image borders (too large)
        if 20 < w < w_img * 0.98 and 20 < h < h_img * 0.98:
            raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return []

    # Since images are assumed to have at most one vehicle profile box, return the largest box (by area)
    largest_box = max(raw_boxes, key=lambda b: b[2] * b[3])
    return [largest_box]


def get_timestamp(img_path: Path) -> float:
    """
    Parses a timestamp from the image's pixels using OCR first.

    If OCR fails, falls back to parsing the filename (expected format HH-MM-SS.jpg)
    combined with the parent folder name (expected format YYYY-MM-DD), or the
    file modification time.

    Args:
        img_path (Path): Path to the image file.

    Returns:
        float: The extracted timestamp in seconds since epoch.

    Raises:
        FileNotFoundError: If the image file does not exist.
        OSError: If reading the file modification time fails.
    """
    # 1. Try OCR using pytesseract first
    ocr_ts = extract_timestamp_via_ocr(img_path)
    if ocr_ts is not None:
        return ocr_ts

    # 2. Try filename and parent directory parsing (fast and accurate if formatted)
    stem = img_path.stem
    try:
        parts = stem.split("-")
        if len(parts) >= 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            parent_name = img_path.parent.name
            try:
                date_parts = parent_name.split("-")
                if len(date_parts) == 3 and len(date_parts[0]) == 4:
                    dt = datetime.datetime(
                        int(date_parts[0]),
                        int(date_parts[1]),
                        int(date_parts[2]),
                        h,
                        m,
                        s,
                    )
                    return dt.timestamp()
            except ValueError:
                pass
            return float(h * 3600 + m * 60 + s)
    except (ValueError, IndexError):
        pass

    # 3. Fallback to file modification time
    try:
        return float(img_path.stat().st_mtime)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Image file not found: {img_path}") from e
    except OSError as e:
        raise OSError(f"Failed to read metadata for {img_path}: {e}") from e


def load_video_start_times(path: Union[str, Path]) -> List[datetime.datetime]:
    """
    Loads a user-supplied JSON file overriding video start timestamps, for
    footage whose file mtime is unreliable (e.g. copied or re-encoded) and
    that has no burned-in on-screen clock get_timestamp's OCR step could fall
    back to instead.

    The file is a plain JSON array with exactly one entry per video, given in
    the same order the caller discovers/sorts its videos in (both
    video_entityprofiler.py and video_plateextractor.py process videos sorted
    by full path) - there's no filename keying, so the caller is responsible
    for matching list position to video position and for checking the lengths
    line up (see validate_video_start_times).

    Args:
        path (Union[str, Path]): Path to a JSON file containing an array of
            start timestamps, each either a UNIX epoch number or an ISO 8601
            string (e.g. "2026-07-08T14:30:00").

    Returns:
        List[datetime.datetime]: Each entry resolved to a datetime, in the
            same order as the input JSON array.

    Raises:
        OSError: If the file can't be read.
        ValueError: If the JSON isn't an array, or a value is neither a
            number nor a parseable ISO 8601 string.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array of start times in '{path}', got "
            f"{type(raw).__name__}."
        )

    start_times: List[datetime.datetime] = []
    for i, value in enumerate(raw):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            start_times.append(datetime.datetime.fromtimestamp(float(value)))
        elif isinstance(value, str):
            start_times.append(datetime.datetime.fromisoformat(value))
        else:
            raise ValueError(
                f"Invalid start time at index {i}: {value!r}. Expected a "
                "UNIX epoch number or an ISO 8601 datetime string."
            )

    return start_times


def validate_video_start_times(
    start_times: Optional[List[datetime.datetime]], video_count: int
) -> None:
    """
    Checks that a loaded start-times list has exactly one entry per video,
    since positions (not filenames) are what map each timestamp to a video.

    Args:
        start_times (Optional[List[datetime.datetime]]): Preloaded list from
            load_video_start_times, or None if no override file was given.
        video_count (int): Number of videos the caller found to process.

    Raises:
        ValueError: If start_times is given and its length doesn't match
            video_count.
    """
    if start_times is None:
        return

    if len(start_times) != video_count:
        raise ValueError(
            f"--start-times provided {len(start_times)} timestamp(s) but "
            f"{video_count} video(s) were found; there must be exactly one "
            "timestamp per video, listed in the same sorted-by-path order "
            "the videos are processed in."
        )
