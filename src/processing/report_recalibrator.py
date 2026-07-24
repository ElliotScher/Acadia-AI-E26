"""
Report Recalibrator

Recalibrates absolute_speed across an existing video_entityprofiler.py JSON
report using a chosen reference entity, without re-running video processing.
relative_speed is already present in the report for every entity, and it's
all calibrate_absolute_speeds needs to derive absolute_speed - so
recalibrating (e.g. because the wrong entity_id was used, or a better
reference speed became available after the fact) is a pure post-processing
step over the report file.

left- and right-traveling entities are calibrated independently, each
against its own reference entity/speed (--left-reference-* /
--right-reference-*), since a camera's pixel-to-real-world relationship
generally isn't the same for both directions of travel (e.g. perspective
foreshortening compresses more pixels per real-world foot in a lane
receding from the camera than one approaching it). At least one direction
must be given; both may be given in the same run. A histogram is saved per
calibrated direction rather than one combined histogram, so two different
scale factors are never blended into the same chart.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.detection.classes import CLASS_ID_MAPPING
from src.utility.loggingutils import setup_logging_and_paths

# Initialize Logger
logger = logging.getLogger("report_recalibrator")

HISTOGRAM_BAR_COLOR = "#2a78d6"
HISTOGRAM_GRIDLINE_COLOR = "#e1e0d9"
HISTOGRAM_AXIS_COLOR = "#c3c2b7"
HISTOGRAM_MUTED_TEXT_COLOR = "#898781"
HISTOGRAM_PRIMARY_TEXT_COLOR = "#0b0b0b"
HISTOGRAM_SURFACE_COLOR = "#fcfcfb"


def calibrate_report(
    report: Dict[str, Any],
    reference_entity_id: int,
    reference_speed: float,
    reference_video: Optional[str] = None,
    entity_type: Optional[int] = None,
    direction: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Recomputes absolute_speed for every entity in a video_entityprofiler.py
    report, scaled against one reference entity's known real-world speed.

    Mirrors video_entityprofiler.calibrate_absolute_speeds, but operates on
    the report's plain "individual_entities" dicts instead of
    VideoEntityRecord objects, since the report doesn't retain the frame
    image data those objects carry.

    Args:
        report (Dict[str, Any]): Parsed JSON report dict, as produced by
            video_entityprofiler.py's --report flag. Must contain an
            "individual_entities" list. Mutated in place.
        reference_entity_id (int): entity_id of the record to calibrate
            against. entity_id is only unique within a single video, so pass
            reference_video too whenever the report spans more than one video.
        reference_speed (float): The reference entity's known real-world
            speed, in whatever unit every entity's absolute_speed should end
            up in (e.g. mph, km/h).
        reference_video (Optional[str]): Filename of the video the reference
            entity came from, to disambiguate when multiple videos produced
            the same entity_id. Matched by filename alone. Defaults to None.
        entity_type (Optional[int]): If given, restricts both the reference
            lookup and the rescale to entities whose "entity_type" (the COCO
            class id video_entityprofiler.py assigned via its --yolo-report)
            equals this value - e.g. calibrating with a known bicycle speed
            (class id 1) only rescales other bicycles, leaving cars/etc.
            untouched. Defaults to None (every entity, regardless of type).
        direction (Optional[str]): If given ("left" or "right"), restricts
            both the reference lookup and the rescale to entities traveling
            in this direction - entities traveling the other direction are
            left untouched. Call this twice, once per direction (each with
            its own reference_entity_id/reference_speed), to recalibrate
            left- and right-traveling traffic against two independent
            real-world reference speeds instead of one shared scale factor,
            since a camera's pixel-to-real-world relationship generally
            isn't the same for both directions. Matched case-insensitively
            against each entity's "direction" field (video_entityprofiler.py
            always writes it lowercase). Defaults to None (every entity,
            regardless of direction).

    Returns:
        Dict[str, Any]: The same report dict, with every matching entity's
            absolute_speed recomputed and metadata updated to reflect this
            calibration - under direction-prefixed keys
            (metadata.left_reference_entity_id, etc.) when direction is
            given, or the bare keys otherwise.

    Raises:
        ValueError: If the report has no individual_entities, no entity
            (matching direction, if given) matches reference_entity_id (and
            reference_video/entity_type, if given), more than one entity
            matches and reference_video wasn't given to disambiguate, or the
            matched reference has a relative_speed of 0 (a
            stationary/degenerate reference can't be used to derive a scale
            factor).
    """
    entities = report.get("individual_entities")
    if not entities:
        raise ValueError("Report has no 'individual_entities' to calibrate.")

    target_entities = entities
    if entity_type is not None:
        target_entities = [e for e in target_entities if e.get("entity_type") == entity_type]
    if direction is not None:
        direction_lower = direction.lower()
        target_entities = [
            e for e in target_entities if str(e.get("direction")).lower() == direction_lower
        ]

    candidates = [e for e in target_entities if e["entity_id"] == reference_entity_id]

    if not candidates:
        raise ValueError(
            f"No entity with entity_id {reference_entity_id} found"
            + (f" with entity_type {entity_type}" if entity_type is not None else "")
            + (f" traveling '{direction}'" if direction is not None else "")
            + "."
        )

    if reference_video is not None:
        ref_name = Path(reference_video).name
        video_matched = [e for e in candidates if Path(e["video"]).name == ref_name]
        if not video_matched:
            found_videos = sorted({str(e["video"]) for e in candidates})
            raise ValueError(
                f"Entity_id {reference_entity_id} was found, but not in a video "
                f"named '{ref_name}' (matched by filename, not full path) - it "
                f"appears in: {', '.join(found_videos)}."
            )
        candidates = video_matched

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple entities with entity_id {reference_entity_id} found across "
            "different videos - pass reference_video to disambiguate."
        )

    reference = candidates[0]
    if reference["relative_speed"] <= 0:
        raise ValueError(
            f"Entity {reference_entity_id} has a relative speed of 0 and can't "
            "be used as a calibration reference."
        )

    scale = reference_speed / reference["relative_speed"]
    for e in target_entities:
        e["absolute_speed"] = e["relative_speed"] * scale

    report.setdefault("metadata", {})
    prefix = f"{direction}_" if direction is not None else ""
    report["metadata"][f"{prefix}reference_entity_id"] = reference_entity_id
    report["metadata"][f"{prefix}reference_speed"] = reference_speed
    report["metadata"][f"{prefix}reference_video"] = reference_video
    report["metadata"][f"{prefix}entity_type_filter"] = entity_type

    return report


def plot_speed_histogram(
    report: Dict[str, Any],
    output_path: Union[str, Path],
    entity_type: Optional[int] = None,
    direction: Optional[str] = None,
) -> None:
    """
    Saves a histogram of every entity's absolute_speed to a PNG file.

    Args:
        report (Dict[str, Any]): A calibrated report dict, as returned by
            calibrate_report - must contain "individual_entities" with
            absolute_speed populated.
        output_path (Union[str, Path]): Filepath to save the histogram PNG to.
        entity_type (Optional[int]): If given, only entities whose
            "entity_type" (a COCO class id) equals this value are plotted.
            Defaults to None (every entity).
        direction (Optional[str]): If given, only entities whose "direction"
            equals this value (matched case-insensitively) are plotted -
            pass this when the report was calibrated with two different
            per-direction reference speeds, since combining both directions'
            absolute_speed into one histogram would blend two different
            scale factors together. Defaults to None (every entity,
            regardless of direction).

    Raises:
        ValueError: If no entity (after any entity_type/direction filtering)
            has a non-null absolute_speed.
    """
    entities = report.get("individual_entities", [])
    if entity_type is not None:
        entities = [e for e in entities if e.get("entity_type") == entity_type]
    if direction is not None:
        direction_lower = direction.lower()
        entities = [e for e in entities if str(e.get("direction")).lower() == direction_lower]

    speeds: List[float] = [
        e["absolute_speed"] for e in entities if e.get("absolute_speed") is not None
    ]
    if not speeds:
        raise ValueError("Report has no entities with an absolute_speed to plot.")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=HISTOGRAM_SURFACE_COLOR)
    ax.set_facecolor(HISTOGRAM_SURFACE_COLOR)

    ax.hist(
        speeds,
        bins="auto",
        color=HISTOGRAM_BAR_COLOR,
        edgecolor=HISTOGRAM_SURFACE_COLOR,
        linewidth=2,
    )

    if entity_type is None:
        title = "Entity Speed Distribution"
    else:
        type_label = CLASS_ID_MAPPING.get(entity_type, str(entity_type))
        title = f"{type_label.capitalize()} Speed Distribution"
    if direction is not None:
        title += f" ({direction.capitalize()})"
    ax.set_title(title, color=HISTOGRAM_PRIMARY_TEXT_COLOR, fontsize=14)
    ax.set_xlabel("Absolute Speed", color=HISTOGRAM_MUTED_TEXT_COLOR)
    ax.set_ylabel("Entities", color=HISTOGRAM_MUTED_TEXT_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(HISTOGRAM_AXIS_COLOR)
    ax.tick_params(colors=HISTOGRAM_MUTED_TEXT_COLOR)
    ax.yaxis.grid(True, color=HISTOGRAM_GRIDLINE_COLOR, linewidth=1)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=HISTOGRAM_SURFACE_COLOR)
    plt.close(fig)


def _histogram_path(args: argparse.Namespace, output_path: Path, direction: str) -> Path:
    """
    Resolves the output path for one direction's speed histogram, inserting
    the direction before the file extension so multiple directions never
    collide on the same file.

    Args:
        args (argparse.Namespace): Parsed CLI args - uses histogram if
            given, else output_path to build the default name.
        output_path (Path): Path the recalibrated report was written to.
        direction (str): Direction this histogram is for (e.g. "left").

    Returns:
        Path: '<histogram stem>_<direction><histogram suffix>' if
            --histogram was given, else
            '<output_path stem>_<direction>_histogram.png'.
    """
    if args.histogram:
        base = Path(args.histogram)
        return base.with_name(f"{base.stem}_{direction}{base.suffix}")
    return output_path.with_name(f"{output_path.stem}_{direction}_histogram.png")


def main() -> None:
    """
    Main CLI entry point for the report recalibrator.
    """
    parser = argparse.ArgumentParser(
        description="Recalibrate absolute_speed across an existing "
        "video_entityprofiler.py JSON report using a new reference entity, "
        "without re-running video processing."
    )
    parser.add_argument(
        "report",
        type=str,
        help="Path to an existing video_entityprofiler.py JSON report (from --report).",
    )
    parser.add_argument(
        "--left-reference-entity-id",
        type=int,
        default=None,
        help="entity_id of one left-traveling tracked vehicle whose actual "
        "real-world speed is known - calibrated independently from "
        "--right-reference-* below, since a camera's pixel-to-real-world "
        "relationship isn't generally the same for both directions of "
        "travel. Requires --left-reference-speed. At least one of "
        "--left-reference-entity-id/--right-reference-entity-id is required.",
    )
    parser.add_argument(
        "--left-reference-speed",
        type=float,
        default=None,
        help="The left reference entity's actual real-world speed (e.g. "
        "mph) - used with --left-reference-entity-id. Whatever unit is "
        "given here is the unit every left-traveling entity's "
        "absolute_speed ends up in.",
    )
    parser.add_argument(
        "--left-reference-video",
        type=str,
        default=None,
        help="Filename of the video --left-reference-entity-id came from, to "
        "disambiguate when multiple videos produced that entity_id.",
    )
    parser.add_argument(
        "--right-reference-entity-id",
        type=int,
        default=None,
        help="Same as --left-reference-entity-id, but for right-traveling "
        "entities - calibrated independently with its own scale factor. "
        "Requires --right-reference-speed.",
    )
    parser.add_argument(
        "--right-reference-speed",
        type=float,
        default=None,
        help="The right reference entity's actual real-world speed - used "
        "with --right-reference-entity-id.",
    )
    parser.add_argument(
        "--right-reference-video",
        type=str,
        default=None,
        help="Filename of the video --right-reference-entity-id came from, "
        "to disambiguate when multiple videos produced that entity_id.",
    )
    parser.add_argument(
        "--entity-type",
        type=int,
        default=None,
        help="If given (e.g. 2 for car, 1 for bicycle - see "
        "src/detection/classes.py's CLASS_ID_MAPPING), restricts both the "
        "reference lookup and the rescale to entities of this COCO class id "
        "- other entities' absolute_speed is left untouched, and the "
        "histogram (if generated) only plots this type. Defaults to None "
        "(every entity).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to write the recalibrated report to. Defaults to "
        "overwriting the input report in place.",
    )
    parser.add_argument(
        "-H",
        "--histogram",
        type=str,
        default=None,
        help="Path to save each calibrated direction's histogram PNG to - "
        "the direction is inserted before the file extension (e.g. "
        "'out_left.png', 'out_right.png' for 'out.png'). Defaults to "
        "'<output>_<direction>_histogram.png'.",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip generating the speed histogram.",
    )

    args, _, _ = setup_logging_and_paths(parser, logger)

    if args.left_reference_entity_id is None and args.right_reference_entity_id is None:
        logger.error(
            "Provide at least one of --left-reference-entity-id or "
            "--right-reference-entity-id."
        )
        sys.exit(1)
    if args.left_reference_entity_id is not None and args.left_reference_speed is None:
        logger.error("--left-reference-entity-id requires --left-reference-speed.")
        sys.exit(1)
    if args.right_reference_entity_id is not None and args.right_reference_speed is None:
        logger.error("--right-reference-entity-id requires --right-reference-speed.")
        sys.exit(1)

    report_path = Path(args.report)
    if not report_path.is_file():
        logger.error("Report file '%s' does not exist.", report_path)
        sys.exit(1)

    with open(report_path, "r") as f:
        report = json.load(f)

    reference_args = {
        "left": (
            args.left_reference_entity_id,
            args.left_reference_speed,
            args.left_reference_video,
        ),
        "right": (
            args.right_reference_entity_id,
            args.right_reference_speed,
            args.right_reference_video,
        ),
    }
    calibrated_directions = []
    for direction, (entity_id, speed, video) in reference_args.items():
        if entity_id is None:
            continue
        try:
            calibrate_report(
                report,
                reference_entity_id=entity_id,
                reference_speed=speed,
                reference_video=video,
                entity_type=args.entity_type,
                direction=direction,
            )
            calibrated_directions.append(direction)
        except ValueError as e:
            logger.error("Could not calibrate '%s' entities: %s", direction, e)
            sys.exit(1)

    output_path = Path(args.output) if args.output else report_path
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    for direction in calibrated_directions:
        entity_id, speed, _ = reference_args[direction]
        calibrated_count = sum(
            1
            for e in report["individual_entities"]
            if str(e.get("direction")).lower() == direction
            and (args.entity_type is None or e.get("entity_type") == args.entity_type)
        )
        logger.info(
            "Recalibrated %d '%s' entities using entity_id %d at %.2f as the reference.",
            calibrated_count,
            direction,
            entity_id,
            speed,
        )
    logger.info("Wrote %s.", output_path)

    if not args.no_histogram:
        for direction in calibrated_directions:
            histogram_path = _histogram_path(args, output_path, direction)
            try:
                plot_speed_histogram(
                    report, histogram_path, entity_type=args.entity_type, direction=direction
                )
                logger.info("Saved '%s' speed histogram to %s.", direction, histogram_path)
            except ValueError as e:
                logger.error(
                    "Could not generate '%s' speed histogram: %s", direction, e
                )


if __name__ == "__main__":
    main()
