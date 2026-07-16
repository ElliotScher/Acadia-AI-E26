"""
Plate OCR Benchmark

Benchmarks imgutils.extract_plate_text_via_ocr's accuracy against a
ground-truth JSON produced by plate_ground_truth_labeler.py: runs OCR on
every image with a known plate text - skipping images the ground truth marks
unreadable ("") or that were never given a final label (null) - and reports
two figures against the hand-confirmed text: the exact plate match rate and
the character-level accuracy (1 - edit distance / ground-truth length,
aggregated across every evaluated plate), plus a breakdown of how far off
(in edit distance) the mismatched plates were. All of this is also broken
out per source video, so a video with a bad angle/lighting doesn't hide
behind a good aggregate score.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2

from src.processing.plate_dwellprofiler import levenshtein_distance
from src.utility.htmlreport import html_heading, html_table, wrap_html_document
from src.utility.imgutils import PLATE_MANIFEST_FILENAME, extract_plate_text_via_ocr

# levenshtein_distance is re-exported from plate_dwellprofiler.py (the
# canonical definition, since it now also drives that module's fuzzy plate
# matching) so existing imports of it from this module keep working.


def _load_raw_ground_truth(path: Path) -> Dict[str, Optional[str]]:
    with open(path, "r") as f:
        return json.load(f)


def load_ground_truth(path: Path) -> Dict[str, str]:
    """
    Loads a ground-truth JSON and keeps only entries with a known plate text.

    Args:
        path (Path): Path to the ground-truth JSON produced by
            plate_ground_truth_labeler.py.

    Returns:
        Dict[str, str]: Relative image path -> ground-truth plate text,
            excluding entries marked unreadable ("") or not yet finished
            (null) - there's no known-correct text to score OCR against for
            either.
    """
    raw = _load_raw_ground_truth(path)
    ground_truth: Dict[str, str] = {}
    for rel_path, text in raw.items():
        if text:
            ground_truth[rel_path] = text
    return ground_truth


def load_source_videos(folder: Path) -> Dict[str, str]:
    """
    Loads video_plateextractor.py's plate_manifest.json from a directory, if
    present, mapping each crop's relative path to its source video's filename.

    Args:
        folder (Path): Directory that may contain a plate_manifest.json.

    Returns:
        Dict[str, str]: Relative crop path -> source video filename. Empty if
            no manifest is found or an entry has no recorded source video.
    """
    manifest_path = folder / PLATE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    return {
        rel_path: Path(entry["source_video"]).name
        for rel_path, entry in data.items()
        if entry.get("source_video")
    }


def video_group_for(rel_path: str, source_videos: Dict[str, str]) -> str:
    """
    Determines which video a crop belongs to, for grouping benchmark results.

    Args:
        rel_path (str): Crop's path relative to the input directory.
        source_videos (Dict[str, str]): Relative crop path -> source video
            filename, from load_source_videos.

    Returns:
        str: The source video's filename if known from the manifest;
            otherwise the crop's parent directory (crops are saved one
            subdirectory per video), or "(unknown video)" if the crop sits
            directly in the input directory with no manifest entry.
    """
    if rel_path in source_videos:
        return source_videos[rel_path]
    parent = Path(rel_path).parent.as_posix()
    return parent if parent != "." else "(unknown video)"


_MISS_BUCKETS: Tuple[str, ...] = ("1", "2", "3", "4+")


def summarize_miss_distances(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarizes how far off the mismatched plates were, in edit distance.

    Aggregate character accuracy is a single length-weighted average across
    every plate, so it can't distinguish "every plate is off by 1-2
    characters" from "a few plates are read as complete garbage while the
    rest are perfect" - both can produce the same aggregate number. This
    looks only at the plates OCR actually got wrong and reports how far off
    each one was, which is the thing you'd need to defend a claim like
    "most misses are off by just 1 character".

    Args:
        results (List[Dict[str, Any]]): Per-image benchmark results, each
            with "exact_match" and "char_errors" keys.

    Returns:
        Dict[str, Any]: Miss count, mean/median edit distance among misses,
            and a histogram (count and share of misses) bucketed by edit
            distance: 1, 2, 3, or 4+ characters off. Exact matches (edit
            distance 0) are excluded entirely, since there's nothing to
            measure for a plate OCR got right.
    """
    miss_distances = [r["char_errors"] for r in results if not r["exact_match"]]
    num_misses = len(miss_distances)

    counts: Dict[str, int] = {bucket: 0 for bucket in _MISS_BUCKETS}
    for distance in miss_distances:
        bucket = str(distance) if distance <= 3 else "4+"
        counts[bucket] += 1

    histogram = {
        bucket: {
            "count": count,
            "share": count / num_misses if num_misses else 0.0,
        }
        for bucket, count in counts.items()
    }

    return {
        "num_misses": num_misses,
        "mean_edit_distance": (
            statistics.mean(miss_distances) if miss_distances else 0.0
        ),
        "median_edit_distance": (
            statistics.median(miss_distances) if miss_distances else 0.0
        ),
        "histogram": histogram,
    }


_BANNER_WIDTH = 46
_LABEL_WIDTH = 26


def _stat_line(label: str, value: str) -> str:
    return f"{label:<{_LABEL_WIDTH}}: {value}"


def _print_summary(
    ground_truth_entries: int,
    total_thrown_out: int,
    thrown_out_unusable: int,
    skipped_missing: int,
    total: int,
    unique_plates_seen: int,
    exact_matches: int,
    match_rate: float,
    char_accuracy: float,
    miss_distribution: Dict[str, Any],
    by_video: List[Dict[str, Any]],
) -> None:
    print()
    print("=" * _BANNER_WIDTH)
    print("PLATE OCR BENCHMARK".center(_BANNER_WIDTH))
    print("=" * _BANNER_WIDTH)
    print(_stat_line("Ground truth entries", str(ground_truth_entries)))
    print(_stat_line("Images thrown out", str(total_thrown_out)))
    if thrown_out_unusable:
        print(_stat_line("  - unreadable/unfinished", str(thrown_out_unusable)))
    if skipped_missing:
        print(_stat_line("  - image file missing", str(skipped_missing)))
    print(_stat_line("Images evaluated", str(total)))
    print(_stat_line("Individual plates seen", str(unique_plates_seen)))
    print("-" * _BANNER_WIDTH)
    print(
        _stat_line(
            "Exact plate match", f"{match_rate * 100:.2f}% ({exact_matches}/{total})"
        )
    )
    print(_stat_line("Character accuracy", f"{char_accuracy * 100:.2f}%"))
    print("=" * _BANNER_WIDTH)

    num_misses = miss_distribution["num_misses"]
    if num_misses:
        print()
        print("MISSES BY EDIT DISTANCE".center(_BANNER_WIDTH, "-"))
        print(_stat_line("Misses (non-exact matches)", str(num_misses)))
        print(
            _stat_line(
                "Mean / median edit distance",
                f"{miss_distribution['mean_edit_distance']:.2f} / "
                f"{miss_distribution['median_edit_distance']:.2f}",
            )
        )
        for bucket in _MISS_BUCKETS:
            entry = miss_distribution["histogram"][bucket]
            label = f"  {bucket} char{'s' if bucket != '1' else ''} off"
            print(
                _stat_line(label, f"{entry['count']:>3}  ({entry['share'] * 100:.2f}%)")
            )

    if by_video:
        print()
        print("BY VIDEO".center(_BANNER_WIDTH, "-"))
        video_w = max(len("Video"), max(len(row["video"]) for row in by_video))
        miss_w = max(
            len("Miss 4+"),
            max(
                len(_miss_cell_text(row["miss_distance_distribution"]["histogram"][b]))
                for row in by_video
                for b in _MISS_BUCKETS
            ),
        )
        header = (
            f"{'Video':<{video_w}}  {'Images':>7}  {'Thrown':>7}  "
            f"{'Plates':>7}  {'Match':>8}  {'CharAcc':>8}  "
            f"{'Miss 1':>{miss_w}}  {'Miss 2':>{miss_w}}  "
            f"{'Miss 3':>{miss_w}}  {'Miss 4+':>{miss_w}}"
        )
        print(header)
        print("-" * len(header))
        for row in by_video:
            miss_hist = row["miss_distance_distribution"]["histogram"]
            print(
                f"{row['video']:<{video_w}}  {row['images_evaluated']:>7}  "
                f"{row['images_thrown_out']:>7}  "
                f"{row['unique_plates_seen']:>7}  "
                f"{row['match_rate'] * 100:>7.2f}%  "
                f"{row['char_accuracy'] * 100:>7.2f}%  "
                f"{_miss_cell_text(miss_hist['1']):>{miss_w}}  "
                f"{_miss_cell_text(miss_hist['2']):>{miss_w}}  "
                f"{_miss_cell_text(miss_hist['3']):>{miss_w}}  "
                f"{_miss_cell_text(miss_hist['4+']):>{miss_w}}"
            )
    print()


_MISS_BUCKET_LABELS = {
    "1": "1 char off",
    "2": "2 chars off",
    "3": "3 chars off",
    "4+": "4+ chars off",
}


def _miss_cell_text(histogram_entry: Dict[str, Any]) -> str:
    """
    Formats one miss-distance bucket as "count (share%)" - the share is
    relative to that group's own miss count, so a video's bucket shares sum
    to 100% of just that video's misses, not the overall total.
    """
    return f"{histogram_entry['count']} ({histogram_entry['share'] * 100:.2f}%)"


def render_html_report(summary_report: Dict[str, Any]) -> str:
    """
    Renders a benchmark summary_report as a standalone HTML page of tables,
    meant to be opened in a browser and copy-pasted (select-all, copy)
    directly into an Outlook email.

    Args:
        summary_report (Dict[str, Any]): The dict returned by
            run_plate_ocr_benchmark.

    Returns:
        str: A complete HTML document.
    """
    stats = summary_report["statistics"]
    miss = summary_report["miss_distance_distribution"]
    by_video = summary_report["by_video"]

    overview_table = html_table(
        ["Metric", "Value"],
        [
            ["Ground truth entries", stats["ground_truth_entries"]],
            ["Images thrown out", stats["images_thrown_out"]],
            [
                "  - Unreadable/unfinished",
                stats["thrown_out_unusable_ground_truth"],
            ],
            ["  - Image file missing", stats["skipped_missing_images"]],
            ["Images evaluated", stats["images_evaluated"]],
            ["Individual plates seen", stats["unique_plates_seen"]],
            [
                "Exact plate match",
                f"{stats['match_rate'] * 100:.2f}% "
                f"({stats['exact_matches']}/{stats['images_evaluated']})",
            ],
            ["Character accuracy", f"{stats['char_accuracy'] * 100:.2f}%"],
        ],
    )

    miss_overview_table = html_table(
        ["Metric", "Value"],
        [
            ["Misses (non-exact matches)", miss["num_misses"]],
            [
                "Mean / median edit distance",
                f"{miss['mean_edit_distance']:.2f} / {miss['median_edit_distance']:.2f}",
            ],
        ],
    )

    miss_histogram_table = html_table(
        ["Edit Distance", "Count", "Share of Misses"],
        [
            [
                _MISS_BUCKET_LABELS[bucket],
                miss["histogram"][bucket]["count"],
                f"{miss['histogram'][bucket]['share'] * 100:.2f}%",
            ]
            for bucket in _MISS_BUCKETS
        ],
    )

    sections = [
        html_heading("Plate OCR Benchmark", level=2),
        overview_table,
        html_heading("Misses by Edit Distance", level=3),
        miss_overview_table,
        "<br>",
        miss_histogram_table,
    ]

    if by_video:
        video_table = html_table(
            [
                "Video",
                "Images",
                "Thrown",
                "Plates",
                "Match",
                "Char Acc",
                "Miss 1",
                "Miss 2",
                "Miss 3",
                "Miss 4+",
            ],
            [
                [
                    row["video"],
                    row["images_evaluated"],
                    row["images_thrown_out"],
                    row["unique_plates_seen"],
                    f"{row['match_rate'] * 100:.2f}%",
                    f"{row['char_accuracy'] * 100:.2f}%",
                    _miss_cell_text(
                        row["miss_distance_distribution"]["histogram"]["1"]
                    ),
                    _miss_cell_text(
                        row["miss_distance_distribution"]["histogram"]["2"]
                    ),
                    _miss_cell_text(
                        row["miss_distance_distribution"]["histogram"]["3"]
                    ),
                    _miss_cell_text(
                        row["miss_distance_distribution"]["histogram"]["4+"]
                    ),
                ]
                for row in by_video
            ],
        )
        sections.append(html_heading("By Video", level=3))
        sections.append(video_table)

    return wrap_html_document(sections)


def run_plate_ocr_benchmark(
    ground_truth_path: Union[str, Path],
    input_dir: Optional[Union[str, Path]] = None,
    report: Optional[Union[str, Path]] = None,
    html_report: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Runs OCR against every ground-truthed plate crop and scores accuracy.

    Args:
        ground_truth_path (Union[str, Path]): Path to the ground-truth JSON.
        input_dir (Optional[Union[str, Path]]): Directory the ground truth's
            relative image paths are resolved against. Defaults to the
            ground truth file's parent directory, matching where
            plate_ground_truth_labeler.py writes it by default.
        report (Optional[Union[str, Path]]): Filepath to save a detailed
            per-image JSON report. Defaults to None.
        html_report (Optional[Union[str, Path]]): Filepath to save an HTML
            version of the summary, with real <table> elements that survive
            copy-paste into Outlook (open in a browser, select all, copy,
            paste). Defaults to None.

    Returns:
        Dict[str, Any]: Summary report dict containing overall accuracy
            statistics, a per-video breakdown, and per-image detail.
    """
    ground_truth_path = Path(ground_truth_path)
    input_folder = Path(input_dir) if input_dir else ground_truth_path.parent

    raw_ground_truth = _load_raw_ground_truth(ground_truth_path)
    source_videos = load_source_videos(input_folder)

    ground_truth: Dict[str, str] = {}
    results: List[Dict[str, Any]] = []
    exact_matches = 0
    total_char_errors = 0
    total_gt_chars = 0
    skipped_missing = 0
    thrown_out_unusable = 0

    # Per-video accumulators, keyed by video_group_for()'s group name. Every
    # raw entry gets a video assigned - including ones thrown out for having
    # no usable ground truth text - so "thrown out" is accurate per video too,
    # not just for images that made it to the OCR step.
    video_plate_texts: Dict[str, set] = defaultdict(set)
    video_exact_matches: Dict[str, int] = defaultdict(int)
    video_char_errors: Dict[str, int] = defaultdict(int)
    video_gt_chars: Dict[str, int] = defaultdict(int)
    video_images: Dict[str, int] = defaultdict(int)
    video_skipped: Dict[str, int] = defaultdict(int)
    video_thrown_out_unusable: Dict[str, int] = defaultdict(int)
    video_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rel_path, text in sorted(raw_ground_truth.items()):
        video = video_group_for(rel_path, source_videos)

        if not text:
            # Marked unreadable ("") or never finished (null) - thrown out
            # before OCR even runs, since there's no known-correct text to
            # score against.
            thrown_out_unusable += 1
            video_thrown_out_unusable[video] += 1
            continue

        true_text = text
        ground_truth[rel_path] = true_text

        img_path = input_folder / rel_path
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_missing += 1
            video_skipped[video] += 1
            continue

        predicted = extract_plate_text_via_ocr(img) or ""
        is_match = predicted == true_text
        char_errors = levenshtein_distance(predicted, true_text)

        exact_matches += int(is_match)
        total_char_errors += char_errors
        total_gt_chars += len(true_text)

        video_images[video] += 1
        video_plate_texts[video].add(true_text)
        video_exact_matches[video] += int(is_match)
        video_char_errors[video] += char_errors
        video_gt_chars[video] += len(true_text)

        result = {
            "image": rel_path,
            "video": video,
            "ground_truth": true_text,
            "predicted": predicted,
            "exact_match": is_match,
            "char_errors": char_errors,
        }
        results.append(result)
        video_results[video].append(result)

    unique_plates_seen = len(set(ground_truth.values()))
    total = len(results)
    match_rate = exact_matches / total if total else 0.0
    char_accuracy = 1 - (total_char_errors / total_gt_chars) if total_gt_chars else 0.0
    total_thrown_out = thrown_out_unusable + skipped_missing
    miss_distribution = summarize_miss_distances(results)

    by_video: List[Dict[str, Any]] = []
    all_videos = set(video_images) | set(video_skipped) | set(video_thrown_out_unusable)
    for video in sorted(all_videos):
        v_total = video_images[video]
        v_matches = video_exact_matches[video]
        v_gt_chars = video_gt_chars[video]
        v_thrown_out_unusable = video_thrown_out_unusable[video]
        v_skipped_missing = video_skipped[video]
        by_video.append(
            {
                "video": video,
                "images_evaluated": v_total,
                "unique_plates_seen": len(video_plate_texts[video]),
                "images_thrown_out": v_thrown_out_unusable + v_skipped_missing,
                "thrown_out_unusable_ground_truth": v_thrown_out_unusable,
                "skipped_missing_images": v_skipped_missing,
                "exact_matches": v_matches,
                "match_rate": v_matches / v_total if v_total else 0.0,
                "char_accuracy": (
                    1 - (video_char_errors[video] / v_gt_chars) if v_gt_chars else 0.0
                ),
                "miss_distance_distribution": summarize_miss_distances(
                    video_results[video]
                ),
            }
        )

    summary_report: Dict[str, Any] = {
        "metadata": {
            "ground_truth_path": str(ground_truth_path),
            "input_dir": str(input_folder),
        },
        "statistics": {
            "ground_truth_entries": len(raw_ground_truth),
            "images_thrown_out": total_thrown_out,
            "thrown_out_unusable_ground_truth": thrown_out_unusable,
            "images_evaluated": total,
            "unique_plates_seen": unique_plates_seen,
            "skipped_missing_images": skipped_missing,
            "exact_matches": exact_matches,
            "match_rate": match_rate,
            "char_accuracy": char_accuracy,
        },
        "miss_distance_distribution": miss_distribution,
        "by_video": by_video,
        "results": results,
    }

    _print_summary(
        ground_truth_entries=len(raw_ground_truth),
        total_thrown_out=total_thrown_out,
        thrown_out_unusable=thrown_out_unusable,
        skipped_missing=skipped_missing,
        total=total,
        unique_plates_seen=unique_plates_seen,
        exact_matches=exact_matches,
        match_rate=match_rate,
        char_accuracy=char_accuracy,
        miss_distribution=miss_distribution,
        by_video=by_video,
    )

    if report:
        with open(report, "w") as f:
            json.dump(summary_report, f, indent=4)

    if html_report:
        with open(html_report, "w") as f:
            f.write(render_html_report(summary_report))

    return summary_report


def main() -> None:
    """
    Main CLI entry point for the plate OCR benchmark script.

    Raises:
        SystemExit: If the ground-truth file is missing/invalid.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark plate OCR accuracy against a ground-truth JSON "
        "produced by plate_ground_truth_labeler.py: reports the exact plate "
        "match rate and character-level accuracy over every image with a "
        "known (non-unreadable) ground-truth plate text."
    )
    parser.add_argument(
        "ground_truth",
        type=str,
        help="Path to the ground-truth JSON file.",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        default=None,
        help="Directory the ground truth's relative image paths are resolved "
        "against. Defaults to the ground truth file's parent directory.",
    )
    parser.add_argument(
        "-r",
        "--report",
        type=str,
        default=None,
        help="Path to save a detailed per-image JSON report.",
    )
    parser.add_argument(
        "-w",
        "--html",
        type=str,
        default=None,
        help="Path to save an HTML version of the summary. Open it in a "
        "browser and copy/paste the tables directly into an Outlook email.",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    if not ground_truth_path.is_file():
        parser.error(f"{ground_truth_path} is not a file.")

    run_plate_ocr_benchmark(
        ground_truth_path=ground_truth_path,
        input_dir=args.input_dir,
        report=args.report,
        html_report=args.html,
    )


if __name__ == "__main__":
    main()
