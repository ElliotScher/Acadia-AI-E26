import pytest
import numpy as np
import cv2


@pytest.fixture
def dummy_bgr_image():
    """Returns a dummy 100x100 black BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def dummy_green_box_image():
    """Returns a 100x100 black BGR image with a solid green rectangle (30x30)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # OpenCV draws rectangle with (x1, y1), (x2, y2), color, thickness
    cv2.rectangle(img, (25, 25), (55, 55), (0, 255, 0), thickness=-1)
    return img
