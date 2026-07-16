import pytest
from src.utility.geometryutils import Rectangle


def test_compute_iou_identical_boxes():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(0, 0, 10, 10)
    assert Rectangle.compute_iou(rect1, rect2) == pytest.approx(1.0)


def test_compute_iou_no_overlap():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(20, 20, 10, 10)
    assert Rectangle.compute_iou(rect1, rect2) == 0.0


def test_compute_iou_partial_overlap():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(5, 5, 10, 10)
    # intersection is 5x5=25, union is 100+100-25=175
    assert Rectangle.compute_iou(rect1, rect2) == pytest.approx(25 / 175)


def test_compute_iou_containment():
    outer = Rectangle(0, 0, 10, 10)
    inner = Rectangle(2, 2, 4, 4)
    # intersection = inner area = 16, union = 100
    assert Rectangle.compute_iou(outer, inner) == pytest.approx(16 / 100)


def test_compute_iou_zero_area_boxes():
    rect1 = Rectangle(0, 0, 0, 0)
    rect2 = Rectangle(0, 0, 0, 0)
    # union area is 0, should not divide by zero
    assert Rectangle.compute_iou(rect1, rect2) == 0.0


def test_compute_iou_symmetric():
    rect1 = Rectangle(0, 0, 10, 10)
    rect2 = Rectangle(5, 5, 10, 10)
    assert Rectangle.compute_iou(rect1, rect2) == Rectangle.compute_iou(rect2, rect1)


def test_is_box_excluded_by_zones_no_zones():
    box = Rectangle(0, 0, 10, 10)
    assert Rectangle.is_box_excluded_by_zones(box, [], []) is False


def test_is_box_excluded_by_zones_hits_exclusion():
    box = Rectangle(0, 0, 10, 10)
    exclusion = Rectangle(5, 5, 10, 10)
    assert Rectangle.is_box_excluded_by_zones(box, [], [exclusion]) is True


def test_is_box_excluded_by_zones_misses_exclusion():
    box = Rectangle(0, 0, 10, 10)
    exclusion = Rectangle(50, 50, 10, 10)
    assert Rectangle.is_box_excluded_by_zones(box, [], [exclusion]) is False


def test_is_box_excluded_by_zones_inside_inclusion():
    box = Rectangle(0, 0, 10, 10)
    inclusion = Rectangle(5, 5, 10, 10)
    assert Rectangle.is_box_excluded_by_zones(box, [inclusion], []) is False


def test_is_box_excluded_by_zones_outside_all_inclusions():
    box = Rectangle(0, 0, 10, 10)
    inclusion = Rectangle(50, 50, 10, 10)
    assert Rectangle.is_box_excluded_by_zones(box, [inclusion], []) is True


def test_is_box_excluded_by_zones_inside_one_of_several_inclusions():
    box = Rectangle(0, 0, 10, 10)
    far_inclusion = Rectangle(100, 100, 10, 10)
    near_inclusion = Rectangle(5, 5, 10, 10)
    assert (
        Rectangle.is_box_excluded_by_zones(box, [far_inclusion, near_inclusion], [])
        is False
    )


def test_is_box_excluded_by_zones_exclusion_takes_priority_over_inclusion():
    box = Rectangle(0, 0, 10, 10)
    inclusion = Rectangle(0, 0, 10, 10)
    exclusion = Rectangle(0, 0, 10, 10)
    assert (
        Rectangle.is_box_excluded_by_zones(box, [inclusion], [exclusion]) is True
    )
