import pytest
from src.detection.image_yolo import Rectangle


def test_rectangle_intersects():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(5, 5, 10, 10)
    # Overlapping rectangles
    assert Rectangle.bounding_box_intersects(rect1, rect2) is True
    assert Rectangle.bounding_box_intersects(rect2, rect1) is True


def test_rectangle_no_intersection():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(15, 15, 10, 10)
    # Disjoint rectangles
    assert Rectangle.bounding_box_intersects(rect1, rect2) is False
    assert Rectangle.bounding_box_intersects(rect2, rect1) is False


def test_rectangle_adjacent_no_intersection():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(10, 0, 10, 10)
    # Bordering, but not overlapping (non-inclusive of edge overlap)
    assert Rectangle.bounding_box_intersects(rect1, rect2) is False
