"""
Report Recalibrator

Recalibrates absolute_speed across an existing video_entityprofiler.py JSON
report using a chosen reference entity, without re-running video processing.
relative_speed is already present in the report for every entity, and it's
all calibrate_absolute_speeds needs to derive absolute_speed - so
recalibrating (e.g. because the wrong entity_id was used, or a better
reference speed became available after the fact) is a pure post-processing
step over the report file.
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
    entity_type: Optional[str] = None,
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
        entity_type (Optional[str]): If given, restricts both the reference
            lookup and the rescale to entities whose "entity_type" (set by
            video_entityprofiler.py's --classify-model) equals this value -
            e.g. calibrating with a known bicycle speed only rescales other
            bicycles, leaving cars/etc. untouched. Defaults to None (every
            entity, regardless of type).

    Returns:
        Dict[str, Any]: The same report dict, with every matching entity's
            absolute_speed recomputed and metadata.reference_entity_id/
            reference_speed/reference_video/entity_type_filter updated to
            reflect this calibration.

    Raises:
        ValueError: If the report has no individual_entities, no entity
            matches reference_entity_id (and reference_video/entity_type, if
            given), more than one entity matches and reference_video wasn't
            given to disambiguate, or the matched reference has a
            relative_speed of 0 (a stationary/degenerate reference can't be
            used to derive a scale factor).
    """
    entities = report.get("individual_entities")
    if not entities:
        raise ValueError("Report has no 'individual_entities' to calibrate.")

    target_entities = (
        [e for e in entities if e.get("entity_type") == entity_type]
        if entity_type is not None
        else entities
    )

    candidates = [e for e in target_entities if e["entity_id"] == reference_entity_id]

    if reference_video is not None:
        ref_name = Path(reference_video).name
        candidates = [e for e in candidates if Path(e["video"]).name == ref_name]

    if not candidates:
        raise ValueError(
            f"No entity with entity_id {reference_entity_id} found"
            + (f" in video '{reference_video}'" if reference_video else "")
            + (f" with entity_type '{entity_type}'" if entity_type else "")
            + "."
        )
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
    report["metadata"]["reference_entity_id"] = reference_entity_id
    report["metadata"]["reference_speed"] = reference_speed
    report["metadata"]["reference_video"] = reference_video
    report["metadata"]["entity_type_filter"] = entity_type

    return report


def plot_speed_histogram(
    report: Dict[str, Any],
    output_path: Union[str, Path],
    entity_type: Optional[str] = None,
) -> None:
    """
    Saves a histogram of every entity's absolute_speed to a PNG file.

    Args:
        report (Dict[str, Any]): A calibrated report dict, as returned by
            calibrate_report - must contain "individual_entities" with
            absolute_speed populated.
        output_path (Union[str, Path]): Filepath to save the histogram PNG to.
        entity_type (Optional[str]): If given, only entities whose
            "entity_type" equals this value are plotted. Defaults to None
            (every entity).

    Raises:
        ValueError: If no entity (after any entity_type filtering) has a
            non-null absolute_speed.
    """
    entities = report.get("individual_entities", [])
    if entity_type is not None:
        entities = [e for e in entities if e.get("entity_type") == entity_type]

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

    title = (
        f"{entity_type.capitalize()} Speed Distribution"
        if entity_type
        else "Entity Speed Distribution"
    )
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
        "--reference-entity-id",
        type=int,
        required=True,
        help="entity_id of one tracked vehicle whose actual real-world speed is known.",
    )
    parser.add_argument(
        "--reference-speed",
        type=float,
        required=True,
        help="The reference entity's actual real-world speed (e.g. mph). "
        "Whatever unit is given here is the unit every entity's "
        "absolute_speed ends up in.",
    )
    parser.add_argument(
        "--reference-video",
        type=str,
        default=None,
        help="Filename of the video --reference-entity-id came from, to "
        "disambiguate when multiple videos produced that entity_id. Only "
        "needed if --reference-entity-id is otherwise ambiguous.",
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        default=None,
        help="If given (e.g. 'bicycle'), restricts both the reference lookup "
        "and the rescale to entities of this type (set by "
        "video_entityprofiler.py's --classify-model) - other entities' "
        "absolute_speed is left untouched, and the histogram (if generated) "
        "only plots this type. Defaults to None (every entity).",
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
        help="Path to save a histogram PNG of every entity's absolute_speed "
        "to. Defaults to '<output>_histogram.png' next to the output report.",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Skip generating the speed histogram.",
    )

    args, _, _ = setup_logging_and_paths(parser, logger)

    report_path = Path(args.report)
    if not report_path.is_file():
        logger.error("Report file '%s' does not exist.", report_path)
        sys.exit(1)

    with open(report_path, "r") as f:
        report = json.load(f)

    try:
        calibrate_report(
            report,
            reference_entity_id=args.reference_entity_id,
            reference_speed=args.reference_speed,
            reference_video=args.reference_video,
            entity_type=args.entity_type,
        )
    except ValueError as e:
        logger.error("Could not calibrate report: %s", e)
        sys.exit(1)

    output_path = Path(args.output) if args.output else report_path
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)

    calibrated_count = (
        sum(
            1
            for e in report["individual_entities"]
            if e.get("entity_type") == args.entity_type
        )
        if args.entity_type
        else len(report["individual_entities"])
    )
    logger.info(
        "Recalibrated %d entities using entity_id %d at %.2f as the reference. "
        "Wrote %s.",
        calibrated_count,
        args.reference_entity_id,
        args.reference_speed,
        output_path,
    )

    if not args.no_histogram:
        histogram_path = (
            Path(args.histogram)
            if args.histogram
            else output_path.with_name(output_path.stem + "_histogram.png")
        )
        plot_speed_histogram(report, histogram_path, entity_type=args.entity_type)
        logger.info("Saved speed histogram to %s.", histogram_path)


if __name__ == "__main__":
    main()
