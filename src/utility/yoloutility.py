import sys
from pathlib import Path

from ultralytics import YOLO


def _models_dir() -> Path:
    """
    Base directory containing this project's bundled `models/` tree - the
    PyInstaller extraction dir when frozen (see main.spec's `datas`),
    otherwise the repo root in a normal dev checkout.
    """
    base = (
        Path(sys._MEIPASS)  # type: ignore[attr-defined]
        if hasattr(sys, "_MEIPASS")
        else Path(__file__).resolve().parents[2]
    )
    return base / "models"


def load_model(model_name: str) -> YOLO:
    """
    Loads and returns a YOLO model from the given path or name.

    Resolves a bare weights filename (e.g. "yolo26s.pt") against this
    project's bundled models/ tree first, so the app always uses the
    checkpoint shipped with it instead of Ultralytics silently downloading
    one from GitHub on first use. Paths that already point to an existing
    file (absolute, or relative to the current working directory) are left
    untouched.

    Args:
        model_name (str): YOLO weights name or path.
    """
    path = Path(model_name)
    if not path.is_absolute() and not path.exists():
        bundled = next(_models_dir().rglob(path.name), None)
        if bundled is not None:
            model_name = str(bundled)

    return YOLO(model_name)
