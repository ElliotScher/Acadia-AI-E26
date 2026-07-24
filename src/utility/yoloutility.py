from ultralytics import YOLO


def load_model(model_name: str) -> YOLO:
    """
    Loads and returns a YOLO model from the given path or name.

    Args:
        model_name (str): YOLO weights name.
    """
    return YOLO(model_name)