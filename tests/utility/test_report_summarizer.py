import json

import pytest

from src.utility.report_summarizer import (
    format_summary,
    load_and_summarize,
    summarize_entities,
    summarize_report,
    summary_to_dict,
)


def _entity(
    video="clip.mp4",
    entity_id=1,
    direction="left",
    entity_type=2,
    relative_speed=0.5,
    absolute_speed=None,
):
    return {
        "video": video,
        "entity_id": entity_id,
        "direction": direction,
        "entity_type": entity_type,
        "relative_speed": relative_speed,
        "absolute_speed": absolute_speed,
    }


def test_summarize_entities_rejects_empty_list():
    with pytest.raises(ValueError, match="individual_entities"):
        summarize_entities([])


def test_summarize_report_rejects_missing_key():
    with pytest.raises(ValueError, match="individual_entities"):
        summarize_report({})


def test_total_counts_and_direction_split():
    entities = [
        _entity(entity_id=1, direction="left"),
        _entity(entity_id=2, direction="left"),
        _entity(entity_id=3, direction="right"),
    ]
    summary = summarize_entities(entities)
    assert summary.total_entities == 3
    assert summary.left_count == 2
    assert summary.right_count == 1


def test_total_videos_counts_distinct_videos():
    entities = [
        _entity(video="a.mp4", entity_id=1),
        _entity(video="a.mp4", entity_id=2),
        _entity(video="b.mp4", entity_id=1),
    ]
    summary = summarize_entities(entities)
    assert summary.total_videos == 2
    by_video = {g.label: g for g in summary.by_video}
    assert by_video["a.mp4"].count == 2
    assert by_video["b.mp4"].count == 1


def test_by_type_groups_and_labels_known_and_unknown_types():
    entities = [
        _entity(entity_id=1, entity_type=2),  # car
        _entity(entity_id=2, entity_type=2),  # car
        _entity(entity_id=3, entity_type=1),  # bicycle
        _entity(entity_id=4, entity_type=None),  # unknown
    ]
    summary = summarize_entities(entities)
    by_type = {g.label: g for g in summary.by_type}
    assert by_type["car"].count == 2
    assert by_type["bicycle"].count == 1
    assert by_type["unknown"].count == 1
    # Sorted by count descending - car (2) should come first.
    assert summary.by_type[0].label == "car"


def test_by_type_direction_breakdown():
    entities = [
        _entity(entity_id=1, entity_type=2, direction="left"),
        _entity(entity_id=2, entity_type=2, direction="right"),
        _entity(entity_id=3, entity_type=2, direction="right"),
    ]
    summary = summarize_entities(entities)
    car_stats = next(g for g in summary.by_type if g.label == "car")
    assert car_stats.left_count == 1
    assert car_stats.right_count == 2


def test_relative_speed_mean_and_median():
    entities = [
        _entity(entity_id=1, relative_speed=0.2),
        _entity(entity_id=2, relative_speed=0.4),
        _entity(entity_id=3, relative_speed=0.6),
    ]
    summary = summarize_entities(entities)
    assert summary.relative_speed_mean == pytest.approx(0.4)
    assert summary.relative_speed_median == pytest.approx(0.4)


def test_absolute_speed_stats_only_include_calibrated_entities():
    entities = [
        _entity(entity_id=1, relative_speed=0.5, absolute_speed=None),
        _entity(entity_id=2, relative_speed=0.5, absolute_speed=25.0),
        _entity(entity_id=3, relative_speed=0.5, absolute_speed=35.0),
    ]
    summary = summarize_entities(entities)
    assert summary.calibrated_count == 2
    assert summary.absolute_speed_mean == pytest.approx(30.0)
    assert summary.absolute_speed_median == pytest.approx(30.0)


def test_absolute_speed_stats_none_when_uncalibrated():
    entities = [_entity(entity_id=1, absolute_speed=None)]
    summary = summarize_entities(entities)
    assert summary.calibrated_count == 0
    assert summary.absolute_speed_mean is None
    assert summary.absolute_speed_median is None


def test_load_and_summarize_reads_report_file(tmp_path):
    report = {
        "metadata": {},
        "individual_entities": [
            _entity(entity_id=1, direction="left"),
            _entity(entity_id=2, direction="right"),
        ],
    }
    report_path = tmp_path / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f)

    summary = load_and_summarize(report_path)
    assert summary.total_entities == 2
    assert summary.left_count == 1
    assert summary.right_count == 1


def test_format_summary_includes_key_fields():
    entities = [
        _entity(entity_id=1, direction="left", entity_type=2),
        _entity(entity_id=2, direction="right", entity_type=1),
    ]
    summary = summarize_entities(entities)
    text = format_summary(summary)
    assert "Total unique entities: 2" in text
    assert "Traveling left: 1" in text
    assert "Traveling right: 1" in text
    assert "car" in text
    assert "bicycle" in text


def test_summary_to_dict_is_json_serializable():
    entities = [_entity(entity_id=1)]
    summary = summarize_entities(entities)
    d = summary_to_dict(summary)
    json.dumps(d)  # Should not raise.
    assert d["total_entities"] == 1
    assert d["by_type"][0]["label"] == "car"
