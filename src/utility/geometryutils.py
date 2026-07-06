from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class Rectangle:
    """
    Represents a 2D bounding box with pixel coordinates.

    Args:
        x (int): Horizontal pixel offset of the top-left corner.
        y (int): Vertical pixel offset of the top-left corner.
        w (int): Bounding box width in pixels.
        h (int): Bounding box height in pixels.
    """

    x: int
    y: int
    w: int
    h: int

    @staticmethod
    def bounding_box_intersects(rect1: "Rectangle", rect2: "Rectangle") -> bool:
        """
        Checks if two rectangles overlap horizontally and vertically.

        Args:
            rect1 (Rectangle): First bounding box.
            rect2 (Rectangle): Second bounding box.

        Returns:
            bool: True if the two rectangles overlap, False otherwise.
        """
        overlap_x = not (rect1.x + rect1.w <= rect2.x or rect2.x + rect2.w <= rect1.x)
        overlap_y = not (rect1.y + rect1.h <= rect2.y or rect2.y + rect2.h <= rect1.y)
        return overlap_x and overlap_y

    @staticmethod
    def compute_iou(rect1: "Rectangle", rect2: "Rectangle") -> float:
        """
        Computes the Intersection-over-Union (IoU) of two Rectangle bounding boxes.

        Args:
            rect1 (Rectangle): First bounding box.
            rect2 (Rectangle): Second bounding box.

        Returns:
            float: Intersection-over-Union ratio in range [0.0, 1.0].
        """
        xi1 = max(rect1.x, rect2.x)
        yi1 = max(rect1.y, rect2.y)
        xi2 = min(rect1.x + rect1.w, rect2.x + rect2.w)
        yi2 = min(rect1.y + rect1.h, rect2.y + rect2.h)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h

        rect1_area = rect1.w * rect1.h
        rect2_area = rect2.w * rect2.h
        union_area = rect1_area + rect2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def is_box_excluded_by_zones(
        box: "Rectangle",
        inclusion_zones: List["Rectangle"],
        exclusion_zones: List["Rectangle"],
    ) -> bool:
        """
        Checks if a box is excluded by the defined inclusion/exclusion rectangular zones.

        Args:
            box (Rectangle): Bounding box in pixel coordinates.
            inclusion_zones (List[Rectangle]): List of allowed inclusion zones (pixel coordinates).
            exclusion_zones (List[Rectangle]): List of forbidden exclusion zones (pixel coordinates).

        Returns:
            bool: True if box is excluded, False otherwise.
        """
        # 1. If it hits any exclusion zone, it's excluded
        for zone in exclusion_zones:
            if Rectangle.bounding_box_intersects(box, zone):
                return True

        # 2. If inclusion zones exist, it must hit at least one inclusion zone
        if inclusion_zones:
            inside = False
            for zone in inclusion_zones:
                if Rectangle.bounding_box_intersects(box, zone):
                    inside = True
                    break
            if not inside:
                return True

        return False
