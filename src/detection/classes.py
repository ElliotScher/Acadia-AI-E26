from typing import Dict, List

TARGET_CLASSES: List[int] = [0, 1, 2, 3, 5, 7]

# Buses (5) and trucks (7) are tracked as ordinary vehicle traffic rather
# than modeled as their own category, so every detector/report in the
# pipeline folds them into "car" (2) - shared here so video_yolo.py's report
# and any downstream consumer agree on the same convention. Merging is keyed
# on the numeric COCO id rather than a resolved string name: the id is the
# canonical value a YOLO model actually predicts, while a name (whether from
# CLASS_ID_MAPPING or a model's own .names) is just a display lookup on top
# of it and shouldn't be re-parsed to decide classification.
_VEHICLE_MERGE_CLASS_IDS = {5, 7}
MERGED_VEHICLE_CLASS_ID = 2


def merge_vehicle_class_id(class_id: int) -> int:
    """
    Folds the "bus" (5) and "truck" (7) COCO class ids into "car" (2); every
    other id passes through unchanged.

    Args:
        class_id (int): A raw COCO class id.

    Returns:
        int: The merged class id.
    """
    return (
        MERGED_VEHICLE_CLASS_ID
        if class_id in _VEHICLE_MERGE_CLASS_IDS
        else class_id
    )

CLASS_ID_MAPPING: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}