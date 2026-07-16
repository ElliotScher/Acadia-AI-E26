from src.detection.classes import TARGET_CLASSES, CLASS_ID_MAPPING


def test_target_classes_all_have_mappings():
    for class_id in TARGET_CLASSES:
        assert class_id in CLASS_ID_MAPPING


def test_target_classes_are_unique():
    assert len(TARGET_CLASSES) == len(set(TARGET_CLASSES))


def test_class_id_mapping_keys_are_unique_and_contiguous():
    keys = sorted(CLASS_ID_MAPPING.keys())
    assert keys == list(range(80))


def test_class_id_mapping_values_are_unique():
    values = list(CLASS_ID_MAPPING.values())
    assert len(values) == len(set(values))


def test_target_classes_expected_labels():
    expected_labels = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    actual_labels = {CLASS_ID_MAPPING[c] for c in TARGET_CLASSES}
    assert actual_labels == expected_labels
