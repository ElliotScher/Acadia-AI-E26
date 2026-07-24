import pytest

from src.processing.report_recalibrator import calibrate_report


def _entity(entity_id, relative_speed, video="clip.mp4", entity_type=2, direction="right"):
    return {
        "video": video,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "direction": direction,
        "relative_speed": relative_speed,
        "absolute_speed": None,
    }


def _report(entities):
    return {"metadata": {}, "individual_entities": entities}


def test_calibrate_report_scales_linearly():
    report = _report(
        [_entity(1, 0.25), _entity(2, 1.0), _entity(3, 0.5)]
    )

    calibrate_report(report, reference_entity_id=2, reference_speed=60.0)

    entities = {e["entity_id"]: e for e in report["individual_entities"]}
    assert entities[1]["absolute_speed"] == pytest.approx(15.0)
    assert entities[2]["absolute_speed"] == pytest.approx(60.0)
    assert entities[3]["absolute_speed"] == pytest.approx(30.0)
    assert report["metadata"]["reference_entity_id"] == 2
    assert report["metadata"]["reference_speed"] == 60.0


def test_calibrate_report_missing_entity_raises():
    report = _report([_entity(1, 0.5)])
    with pytest.raises(ValueError):
        calibrate_report(report, reference_entity_id=99, reference_speed=60.0)


def test_calibrate_report_video_mismatch_names_the_actual_video():
    # entity_id 1 exists, but not in the video the caller named - this should
    # be distinguishable from "entity_id not found at all", and should name
    # the video it actually appears in so a wrong --reference-video is
    # obvious rather than reading as a missing/wrong entity_id.
    report = _report([_entity(1, 0.5, video="cam1.mp4")])
    with pytest.raises(ValueError, match="cam1.mp4"):
        calibrate_report(
            report, reference_entity_id=1, reference_speed=60.0, reference_video="wrong_name.mp4"
        )


def test_calibrate_report_zero_reference_raises():
    report = _report([_entity(1, 0.0)])
    with pytest.raises(ValueError):
        calibrate_report(report, reference_entity_id=1, reference_speed=60.0)


def test_calibrate_report_no_individual_entities_raises():
    with pytest.raises(ValueError, match="individual_entities"):
        calibrate_report({}, reference_entity_id=1, reference_speed=60.0)


def test_calibrate_report_entity_type_scopes_reference_and_rescale():
    report = _report(
        [
            _entity(1, 0.5, entity_type=2),
            _entity(2, 1.0, entity_type=1),
        ]
    )
    # entity_id 2 only exists among entity_type=1 - lookup should find it
    # when entity_type=1 is given, and only rescale entity_type=1 entities.
    calibrate_report(report, reference_entity_id=2, reference_speed=60.0, entity_type=1)

    entities = {e["entity_id"]: e for e in report["individual_entities"]}
    assert entities[1]["absolute_speed"] is None
    assert entities[2]["absolute_speed"] == pytest.approx(60.0)


def test_calibrate_report_direction_scopes_reference_lookup_and_rescale():
    report = _report(
        [
            _entity(1, 0.5, direction="left"),
            _entity(2, 1.0, direction="left"),
            _entity(3, 0.25, direction="right"),
        ]
    )
    calibrate_report(report, reference_entity_id=2, reference_speed=60.0, direction="left")

    entities = {e["entity_id"]: e for e in report["individual_entities"]}
    assert entities[1]["absolute_speed"] == pytest.approx(30.0)
    assert entities[2]["absolute_speed"] == pytest.approx(60.0)
    assert entities[3]["absolute_speed"] is None


def test_calibrate_report_direction_is_case_insensitive():
    report = _report([_entity(1, 1.0, direction="Left")])
    calibrate_report(report, reference_entity_id=1, reference_speed=60.0, direction="left")
    assert report["individual_entities"][0]["absolute_speed"] == pytest.approx(60.0)


def test_calibrate_report_left_and_right_use_independent_scale_factors():
    report = _report(
        [
            _entity(1, 0.5, direction="left"),
            _entity(2, 1.0, direction="left"),
            _entity(3, 0.25, direction="right"),
            _entity(4, 0.5, direction="right"),
        ]
    )

    calibrate_report(report, reference_entity_id=2, reference_speed=60.0, direction="left")
    calibrate_report(report, reference_entity_id=4, reference_speed=20.0, direction="right")

    entities = {e["entity_id"]: e for e in report["individual_entities"]}
    assert entities[1]["absolute_speed"] == pytest.approx(30.0)
    assert entities[2]["absolute_speed"] == pytest.approx(60.0)
    assert entities[3]["absolute_speed"] == pytest.approx(10.0)
    assert entities[4]["absolute_speed"] == pytest.approx(20.0)


def test_calibrate_report_direction_metadata_does_not_clobber_across_calls():
    report = _report(
        [
            _entity(1, 1.0, direction="left"),
            _entity(2, 1.0, direction="right"),
        ]
    )

    calibrate_report(report, reference_entity_id=1, reference_speed=60.0, direction="left")
    calibrate_report(report, reference_entity_id=2, reference_speed=20.0, direction="right")

    metadata = report["metadata"]
    assert metadata["left_reference_entity_id"] == 1
    assert metadata["left_reference_speed"] == 60.0
    assert metadata["right_reference_entity_id"] == 2
    assert metadata["right_reference_speed"] == 20.0


def test_calibrate_report_direction_missing_entity_raises():
    report = _report([_entity(1, 0.5, direction="right")])
    with pytest.raises(ValueError):
        calibrate_report(report, reference_entity_id=1, reference_speed=60.0, direction="left")
