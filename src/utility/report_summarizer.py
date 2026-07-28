"""
Report Summarizer

Ingests a video_entityprofiler.py (or report_recalibrator.py) JSON report and
extracts the summary statistics that are otherwise buried in its flat
"individual_entities" list: total unique entities, direction split (left vs
right), a breakdown by entity type (see src/detection/classes.py's
CLASS_ID_MAPPING) crossed with direction, a per-video breakdown, and
relative_speed/absolute_speed statistics - overall and per type.

video_entityprofiler.py's own CLI already prints a couple of these numbers
(total entities, left/right counts) but doesn't retain that breakdown
anywhere, doesn't cross it with entity type, and can't be pointed at a report
after the fact. This is a pure post-processing step over the report file, no
different in spirit from report_recalibrator.py.
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.detection.classes import CLASS_ID_MAPPING

logger = logging.getLogger("report_summarizer")


def _entity_type_label(entity_type: Optional[int]) -> str:
    if entity_type is None:
        return "unknown"
    return CLASS_ID_MAPPING.get(entity_type, f"unknown ({entity_type})")


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return statistics.median(values) if values else None


@dataclass
class GroupStats:
    """
    Entity counts and speed statistics for one slice of a report (e.g. all
    entities of one type, or all entities from one video).

    Args:
        label (str): Human-readable identifier for this group (an entity
            type's display name, or a video filename).
        count (int): Number of entities in this group.
        left_count (int): Number traveling "left".
        right_count (int): Number traveling "right".
        relative_speed_mean (Optional[float]): Mean relative_speed across the
            group. None if the group is empty.
        relative_speed_median (Optional[float]): Median relative_speed across
            the group. None if the group is empty.
        absolute_speed_mean (Optional[float]): Mean absolute_speed across
            entities in the group that have one. None if none do.
        absolute_speed_median (Optional[float]): Median absolute_speed across
            entities in the group that have one. None if none do.
    """

    label: str
    count: int
    left_count: int
    right_count: int
    relative_speed_mean: Optional[float]
    relative_speed_median: Optional[float]
    absolute_speed_mean: Optional[float]
    absolute_speed_median: Optional[float]


@dataclass
class ReportSummary:
    """
    Every summary statistic extracted from a video_entityprofiler.py report.

    Args:
        total_entities (int): Total number of entities across every video.
        total_videos (int): Number of distinct videos entities were drawn
            from (from the "video" field on each entity, not metadata - so
            this still reflects reality if a video produced zero entities
            and was filtered out upstream).
        left_count (int): Total entities traveling "left".
        right_count (int): Total entities traveling "right".
        calibrated_count (int): Number of entities with a non-null
            absolute_speed (i.e. the report went through calibration).
        by_type (List[GroupStats]): One GroupStats per distinct entity_type
            (including "unknown" for entities with a null entity_type),
            sorted by count descending.
        by_video (List[GroupStats]): One GroupStats per distinct video,
            sorted by count descending.
        relative_speed_mean (Optional[float]): Mean relative_speed across
            every entity. None if the report has no entities.
        relative_speed_median (Optional[float]): Median relative_speed across
            every entity. None if the report has no entities.
        absolute_speed_mean (Optional[float]): Mean absolute_speed across
            every calibrated entity. None if none are calibrated.
        absolute_speed_median (Optional[float]): Median absolute_speed across
            every calibrated entity. None if none are calibrated.
    """

    total_entities: int
    total_videos: int
    left_count: int
    right_count: int
    calibrated_count: int
    by_type: List[GroupStats]
    by_video: List[GroupStats]
    relative_speed_mean: Optional[float]
    relative_speed_median: Optional[float]
    absolute_speed_mean: Optional[float]
    absolute_speed_median: Optional[float]


def _group_stats(label: str, entities: List[Dict[str, Any]]) -> GroupStats:
    relative_speeds = [e["relative_speed"] for e in entities]
    absolute_speeds = [
        e["absolute_speed"] for e in entities if e.get("absolute_speed") is not None
    ]
    return GroupStats(
        label=label,
        count=len(entities),
        left_count=sum(1 for e in entities if e.get("direction") == "left"),
        right_count=sum(1 for e in entities if e.get("direction") == "right"),
        relative_speed_mean=_mean(relative_speeds),
        relative_speed_median=_median(relative_speeds),
        absolute_speed_mean=_mean(absolute_speeds),
        absolute_speed_median=_median(absolute_speeds),
    )


def summarize_entities(entities: List[Dict[str, Any]]) -> ReportSummary:
    """
    Computes every summary statistic over a report's "individual_entities" list.

    Args:
        entities (List[Dict[str, Any]]): The report's "individual_entities"
            value - one dict per tracked entity, as produced by
            video_entityprofiler.py's --report flag.

    Returns:
        ReportSummary: Every extracted statistic.

    Raises:
        ValueError: If entities is empty.
    """
    if not entities:
        raise ValueError("Report has no 'individual_entities' to summarize.")

    by_type_entities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_video_entities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entities:
        by_type_entities[_entity_type_label(e.get("entity_type"))].append(e)
        by_video_entities[e.get("video", "unknown")].append(e)

    by_type = sorted(
        (_group_stats(label, group) for label, group in by_type_entities.items()),
        key=lambda g: g.count,
        reverse=True,
    )
    by_video = sorted(
        (_group_stats(label, group) for label, group in by_video_entities.items()),
        key=lambda g: g.count,
        reverse=True,
    )

    relative_speeds = [e["relative_speed"] for e in entities]
    absolute_speeds = [
        e["absolute_speed"] for e in entities if e.get("absolute_speed") is not None
    ]

    return ReportSummary(
        total_entities=len(entities),
        total_videos=len(by_video_entities),
        left_count=sum(1 for e in entities if e.get("direction") == "left"),
        right_count=sum(1 for e in entities if e.get("direction") == "right"),
        calibrated_count=len(absolute_speeds),
        by_type=by_type,
        by_video=by_video,
        relative_speed_mean=_mean(relative_speeds),
        relative_speed_median=_median(relative_speeds),
        absolute_speed_mean=_mean(absolute_speeds),
        absolute_speed_median=_median(absolute_speeds),
    )


def summarize_report(report: Dict[str, Any]) -> ReportSummary:
    """
    Computes every summary statistic over a parsed report dict.

    Args:
        report (Dict[str, Any]): Parsed JSON report dict, as produced by
            video_entityprofiler.py's --report flag (or later recalibrated by
            report_recalibrator.py). Must contain "individual_entities".

    Returns:
        ReportSummary: Every extracted statistic.

    Raises:
        ValueError: If the report has no "individual_entities".
    """
    entities = report.get("individual_entities")
    if not entities:
        raise ValueError("Report has no 'individual_entities' to summarize.")
    return summarize_entities(entities)


def load_and_summarize(report_path: "str | Path") -> ReportSummary:
    """
    Loads a report JSON file and computes every summary statistic over it.

    Args:
        report_path (str | Path): Path to a video_entityprofiler.py (or
            report_recalibrator.py) JSON report.

    Returns:
        ReportSummary: Every extracted statistic.

    Raises:
        ValueError: If the report has no "individual_entities".
        OSError: If the file can't be read.
        json.JSONDecodeError: If the file isn't valid JSON.
    """
    with open(report_path, "r") as f:
        report = json.load(f)
    return summarize_report(report)


def _format_speed_suffix(g: GroupStats) -> str:
    parts = []
    if g.relative_speed_mean is not None:
        parts.append(
            f"rel_speed mean {g.relative_speed_mean:.3f} / median {g.relative_speed_median:.3f}"
        )
    if g.absolute_speed_mean is not None:
        parts.append(
            f"abs_speed mean {g.absolute_speed_mean:.2f} / median {g.absolute_speed_median:.2f}"
        )
    return f" ({', '.join(parts)})" if parts else ""


def format_summary(summary: ReportSummary) -> str:
    """
    Formats a ReportSummary as a human-readable multi-line report.

    Args:
        summary (ReportSummary): Summary to format, as returned by
            summarize_report/summarize_entities/load_and_summarize.

    Returns:
        str: Multi-line summary text.
    """
    lines = [
        "--- Report Summary ---",
        f"Total unique entities: {summary.total_entities}",
        f"Total videos: {summary.total_videos}",
        f"Traveling left: {summary.left_count}",
        f"Traveling right: {summary.right_count}",
        f"Calibrated (absolute_speed populated): {summary.calibrated_count}/{summary.total_entities}",
    ]

    if summary.relative_speed_mean is not None:
        lines.append(
            f"Overall relative_speed: mean {summary.relative_speed_mean:.3f}, "
            f"median {summary.relative_speed_median:.3f}"
        )
    if summary.absolute_speed_mean is not None:
        lines.append(
            f"Overall absolute_speed: mean {summary.absolute_speed_mean:.2f}, "
            f"median {summary.absolute_speed_median:.2f}"
        )

    lines.append("\nBy entity type:")
    for g in summary.by_type:
        lines.append(
            f"  {g.label:<15} {g.count:>5}  (left {g.left_count}, right {g.right_count})"
            f"{_format_speed_suffix(g)}"
        )

    lines.append("\nBy video:")
    for g in summary.by_video:
        lines.append(
            f"  {g.label:<30} {g.count:>5}  (left {g.left_count}, right {g.right_count})"
        )

    return "\n".join(lines)


def summary_to_dict(summary: ReportSummary) -> Dict[str, Any]:
    """
    Converts a ReportSummary into a plain JSON-serializable dict.

    Args:
        summary (ReportSummary): Summary to convert.

    Returns:
        Dict[str, Any]: Dict shaped for json.dump, with by_type/by_video as
            lists of plain dicts keyed by group label.
    """
    return {
        "total_entities": summary.total_entities,
        "total_videos": summary.total_videos,
        "left_count": summary.left_count,
        "right_count": summary.right_count,
        "calibrated_count": summary.calibrated_count,
        "relative_speed_mean": summary.relative_speed_mean,
        "relative_speed_median": summary.relative_speed_median,
        "absolute_speed_mean": summary.absolute_speed_mean,
        "absolute_speed_median": summary.absolute_speed_median,
        "by_type": [vars(g) for g in summary.by_type],
        "by_video": [vars(g) for g in summary.by_video],
    }


def main() -> None:
    """
    Main CLI entry point for the report summarizer.
    """
    parser = argparse.ArgumentParser(
        description="Extract summary statistics (total entities, direction "
        "split, entity type and per-video breakdowns, speed stats) from a "
        "video_entityprofiler.py JSON report."
    )
    parser.add_argument(
        "report",
        type=str,
        help="Path to a video_entityprofiler.py (or report_recalibrator.py) JSON report.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to also write the summary as JSON. Defaults to None (print only).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    report_path = Path(args.report)
    if not report_path.is_file():
        logger.error("Report file '%s' does not exist.", report_path)
        sys.exit(1)

    try:
        summary = load_and_summarize(report_path)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Could not load '%s': %s", report_path, e)
        sys.exit(1)
    except ValueError as e:
        logger.error("Could not summarize '%s': %s", report_path, e)
        sys.exit(1)

    print(format_summary(summary))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary_to_dict(summary), f, indent=4)
        logger.info("Wrote summary JSON to %s.", args.output)


if __name__ == "__main__":
    main()
