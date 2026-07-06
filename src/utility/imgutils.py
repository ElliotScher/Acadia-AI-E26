import datetime
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np


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
        gray = img.convert('L')
        
        # Crop ONLY the bottom 10% of the image as the time is always printed there
        w, h = img.size
        bottom_crop = gray.crop((0, int(h * 0.90), w, h))
        # Try OCR with default PSM, then fallback to PSM 6
        for config in [None, "--psm 6"]:
            if config:
                text = pytesseract.image_to_string(bottom_crop, config=config)
            else:
                text = pytesseract.image_to_string(bottom_crop)
            
            # Regex search for datetime patterns
            # 1. YYYY-MM-DD HH:MM:SS (or YYYY/MM/DD)
            pattern1 = r'(\d{4})[-/](\d{2})[-/](\d{2})\s*(\d{2})[:\.](\d{2})[:\.](\d{2})'
            match1 = re.search(pattern1, text)
            if match1:
                year, month, day, hour, minute, second = map(int, match1.groups())
                dt_obj = datetime.datetime(year, month, day, hour, minute, second)
                return dt_obj.timestamp()
                
            # 2. MM/DD/YYYY HH:MM:SS or DD/MM/YYYY HH:MM:SS
            pattern2 = r'(\d{2})[-/](\d{2})[-/](\d{4})\s*(\d{2})[:\.](\d{2})[:\.](\d{2})'
            match2 = re.search(pattern2, text)
            if match2:
                val1, val2, year, hour, minute, second = map(int, match2.groups())
                # Try MM/DD/YYYY first
                try:
                    dt_obj = datetime.datetime(year, val1, val2, hour, minute, second)
                    return dt_obj.timestamp()
                except ValueError:
                    # Fall back to DD/MM/YYYY
                    try:
                        dt_obj = datetime.datetime(year, val2, val1, hour, minute, second)
                        return dt_obj.timestamp()
                    except ValueError:
                        pass
                        
            # 3. Simple time fallback: HH:MM:SS
            pattern3 = r'(\d{2})[:\.](\d{2})[:\.](\d{2})'
            match3 = re.search(pattern3, text)
            if match3:
                hour, minute, second = map(int, match3.groups())
                return float(hour * 3600 + minute * 60 + second)
            
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
