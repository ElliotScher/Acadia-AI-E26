import numpy as np
import pytest

from utility import imgutils
from utility.geometryutils import Rectangle


def test_compute_iou():
    # Perfect overlap
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(0, 0, 10, 10)
    assert Rectangle.compute_iou(box1, box2) == pytest.approx(1.0)

    # Partial overlap: half width, identical height
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(5, 0, 10, 10)
    # intersection: 5 width, 10 height = 50 area
    # union: 100 + 100 - 50 = 150 area
    # IoU: 50 / 150 = 0.3333
    assert Rectangle.compute_iou(box1, box2) == pytest.approx(1.0 / 3.0)

    # No overlap
    box1 = Rectangle(0, 0, 10, 10)
    box2 = Rectangle(20, 20, 10, 10)
    assert Rectangle.compute_iou(box1, box2) == 0.0


def test_compute_sharpness():
    # Empty crop
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    assert imgutils.compute_sharpness(empty_crop) == 0.0

    # Constant crop (no edges -> variance of Laplacian is 0)
    flat_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    assert imgutils.compute_sharpness(flat_crop) == 0.0

    # Edge crop (sharp transitions -> positive variance)
    edge_crop = np.zeros((50, 50, 3), dtype=np.uint8)
    edge_crop[:, 25:] = 255
    assert imgutils.compute_sharpness(edge_crop) > 0.0


def test_is_box_excluded_by_zones():
    # Test case 1: no zones
    assert (
        Rectangle.is_box_excluded_by_zones(Rectangle(10, 10, 20, 20), [], []) is False
    )

    # Test case 2: inclusion zone (box inside)
    inclusion_zones = [Rectangle(0, 0, 50, 50)]
    # Box (10, 10, 20, 20) inside inclusion zone (0, 0 to 50, 50) -> not excluded (False)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(10, 10, 20, 20), inclusion_zones, []
        )
        is False
    )

    # Box (60, 60, 20, 20) outside inclusion zone -> excluded (True)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(60, 60, 20, 20), inclusion_zones, []
        )
        is True
    )

    # Test case 3: exclusion zone (box inside)
    exclusion_zones = [Rectangle(0, 0, 50, 50)]
    # Box (10, 10, 20, 20) inside exclusion zone -> excluded (True)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(10, 10, 20, 20), [], exclusion_zones
        )
        is True
    )

    # Box (60, 60, 20, 20) outside exclusion zone -> not excluded (False)
    assert (
        Rectangle.is_box_excluded_by_zones(
            Rectangle(60, 60, 20, 20), [], exclusion_zones
        )
        is False
    )


