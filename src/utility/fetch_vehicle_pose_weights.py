"""
Vehicle Pose Weights Fetcher

Downloads the pretrained vehicle-pose checkpoint used by
detection.direction.vehicle_direction from its original source - a Google
Drive folder maintained by https://github.com/Habib0905/Vehicle-Pose-Estimation
(unlicensed research code; not vendored into this repo - see that module's
docstring for why we depend on it anyway) - to a local path, if it isn't
already there. Mirrors how Ultralytics' own pretrained weights (e.g.
yolo26s-pose.pt) are fetched on first use rather than committed to git; the
difference is this checkpoint isn't on any registry Ultralytics knows about,
so this project has to do that fetch-and-cache step itself.
"""

import argparse
from pathlib import Path

import gdown

# From github.com/Habib0905/Vehicle-Pose-Estimation's weights.txt, which
# points at https://drive.google.com/drive/folders/17u0B0aKTYkY8I72gQLl2tsEyvQzDavJp
# - last.pt specifically (the repo's own inference notebook uses last.pt,
# not best.pt).
_WEIGHTS_FILE_ID = "135Yj2fSR9FP1FZhScHdEOlBb3DeSRNA6"
_WEIGHTS_URL = f"https://drive.google.com/uc?id={_WEIGHTS_FILE_ID}"

# Repo-relative local cache path: src/utility/fetch_vehicle_pose_weights.py ->
# parents[2] is the repo root.
DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "vehicle_direction" / "last.pt"
)


def fetch_vehicle_pose_weights(
    dest: Path = DEFAULT_WEIGHTS_PATH, force: bool = False
) -> Path:
    """
    Downloads the vehicle-pose checkpoint to dest if it isn't already there.

    Args:
        dest (Path): Local path to save the weights to. Defaults to
            DEFAULT_WEIGHTS_PATH.
        force (bool): Re-download even if dest already exists. Defaults to
            False.

    Returns:
        Path: dest, once the weights are confirmed present on disk.

    Raises:
        RuntimeError: If the download completes without producing a file at
            dest (e.g. Google Drive served an interstitial page instead of
            the weights).
    """
    if dest.exists() and not force:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(_WEIGHTS_URL, str(dest), quiet=False)

    if not dest.exists():
        raise RuntimeError(f"Failed to download vehicle-pose weights to {dest}")

    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Downloads the pretrained vehicle-pose checkpoint used by "
        "detection.direction.vehicle_direction, if it isn't already cached "
        "locally."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"Where to save the weights (default: {DEFAULT_WEIGHTS_PATH}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists at the destination.",
    )
    args = parser.parse_args()

    dest = fetch_vehicle_pose_weights(Path(args.output), force=args.force)
    print(f"Vehicle-pose weights available at {dest}")


if __name__ == "__main__":
    main()
