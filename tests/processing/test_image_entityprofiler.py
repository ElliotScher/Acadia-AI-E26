import pytest
import numpy as np
from src.processing.image_entityprofiler import (
    ProfileRecord,
    Direction,
    ProfilingMode,
    compute_similarities,
    assign_entity_id,
    track_entities_in_directory,
    match_entry_exit_entities,
    calculate_occupancy_timeline,
    run_entry_exit_profiling,
)


def test_compute_similarities_identical():
    # Setup database with one record
    feat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    record = ProfileRecord(
        entity_id=1,
        feature=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=100.0,
        img_name="image1.jpg",
        box=[0, 0, 10, 10],
    )
    database = [record]

    # Compute similarity with identical feature in a different image and time
    sims = compute_similarities(
        feat=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=200.0,
        img_name="image2.jpg",
        database=database,
    )

    # Identical normalized feature vector cosine similarity = 1.0
    assert sims[1] == pytest.approx(1.0)


def test_compute_similarities_same_image_constraint():
    feat = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    record = ProfileRecord(
        entity_id=1,
        feature=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=100.0,
        img_name="image1.jpg",
        box=[0, 0, 10, 10],
    )
    database = [record]

    # Query is in the exact same image
    sims = compute_similarities(
        feat=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=100.0,
        img_name="image1.jpg",
        database=database,
    )

    # Penalized to -1.0 because two instances of the same entity cannot be in the same image
    assert sims[1] == -1.0


def test_assign_entity_id_new_entity():
    # Feature vectors are orthogonal (cosine similarity = 0.0)
    feat1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    feat2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    record = ProfileRecord(
        entity_id=1,
        feature=feat1,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=100.0,
        img_name="image1.jpg",
    )
    database = [record]

    assigned_id, sim, is_new = assign_entity_id(
        feat=feat2,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=200.0,
        img_name="image2.jpg",
        database=database,
        next_id=2,
        threshold=0.75,
    )

    # Low similarity (0.0) should trigger a new entity creation
    assert is_new is True
    assert assigned_id == 2
    assert sim == pytest.approx(0.0)


def test_assign_entity_id_existing_entity():
    feat = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    record = ProfileRecord(
        entity_id=42,
        feature=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=100.0,
        img_name="image1.jpg",
    )
    database = [record]

    assigned_id, sim, is_new = assign_entity_id(
        feat=feat,
        hsv_hist=None,
        aspect_ratio=None,
        timestamp=200.0,
        img_name="image2.jpg",
        database=database,
        next_id=43,
        threshold=0.75,
    )

    # High similarity (1.0 >= 0.75) should match to existing entity 42
    assert is_new is False
    assert assigned_id == 42
    assert sim == pytest.approx(1.0)



def test_track_entities_in_directory_max_gap():
    from pathlib import Path

    feat1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # Detections of similar object but separated by large timestamp (gap > 10.0)
    results_dict = {
        Path("img1.jpg"): [
            {"box": [10, 10, 5, 5], "feature": feat1, "timestamp": 10.0}
        ],
        Path("img2.jpg"): [
            {"box": [12, 10, 5, 5], "feature": feat1, "timestamp": 25.0}
        ],  # Gap is 15.0 seconds
    }
    image_paths = [Path("img1.jpg"), Path("img2.jpg")]

    # Without max_gap, they match
    db, grouped = track_entities_in_directory(
        results_dict, image_paths, threshold=0.75, max_gap=None
    )
    assert len(grouped) == 1

    # With max_gap = 10.0, they should split into 2 entities
    db, grouped = track_entities_in_directory(
        results_dict, image_paths, threshold=0.75, max_gap=10.0
    )
    assert len(grouped) == 2


def test_match_entry_exit_entities():
    feat1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    feat2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    entry_entities = {
        1: [
            ProfileRecord(
                entity_id=1, feature=feat1, timestamp=10.0, box=[10, 20, 5, 5]
            )
        ],
        2: [
            ProfileRecord(
                entity_id=2, feature=feat2, timestamp=20.0, box=[10, 20, 5, 5]
            )
        ],
    }

    exit_entities = {
        10: [
            ProfileRecord(
                entity_id=10, feature=feat1, timestamp=50.0, box=[30, 20, 5, 5]
            )
        ],
    }

    # Matches exit_10 (feat1) to entry_1 (feat1)
    matches = match_entry_exit_entities(entry_entities, exit_entities, threshold=0.75)
    assert len(matches) == 1
    assert matches[0]["entry_id"] == 1
    assert matches[0]["exit_id"] == 10
    assert matches[0]["dwell_time"] == pytest.approx(40.0)


def test_calculate_occupancy_timeline():
    entry_entities = {
        1: [
            ProfileRecord(
                entity_id=1, feature=np.array([1]), timestamp=10.0, box=[10, 20, 5, 5]
            )
        ],
        2: [
            ProfileRecord(
                entity_id=2, feature=np.array([1]), timestamp=30.0, box=[10, 20, 5, 5]
            )
        ],
    }

    exit_entities = {
        10: [
            ProfileRecord(
                entity_id=10, feature=np.array([1]), timestamp=20.0, box=[30, 20, 5, 5]
            )
        ],
    }

    timeline = calculate_occupancy_timeline(entry_entities, exit_entities)
    assert len(timeline) == 3
    assert timeline[0]["occupancy"] == 1
    assert timeline[1]["occupancy"] == 0
    assert timeline[2]["occupancy"] == 1

    # Test adjustment shift if running occupancy drops below 0
    entry_entities_adj = {
        1: [
            ProfileRecord(
                entity_id=1, feature=np.array([1]), timestamp=20.0, box=[10, 20, 5, 5]
            )
        ],
    }
    exit_entities_adj = {
        10: [
            ProfileRecord(
                entity_id=10, feature=np.array([1]), timestamp=10.0, box=[30, 20, 5, 5]
            )
        ],
    }
    timeline_adj = calculate_occupancy_timeline(entry_entities_adj, exit_entities_adj)
    assert len(timeline_adj) == 2
    assert timeline_adj[0]["occupancy"] == 0
    assert timeline_adj[1]["occupancy"] == 1


def test_run_entry_exit_profiling_api():
    from unittest.mock import patch, MagicMock
    from pathlib import Path
    
    with patch("src.processing.image_entityprofiler.load_feature_extractor") as mock_load, \
         patch("src.processing.image_entityprofiler.extract_features_for_directory") as mock_extract, \
         patch("src.processing.image_entityprofiler.Path.exists", return_value=True), \
         patch("src.processing.image_entityprofiler.Path.is_file", return_value=True), \
         patch("src.processing.image_entityprofiler.Path.rglob", return_value=[Path("img1_car.jpg")]):
         
         mock_load.return_value = (MagicMock(), MagicMock(), MagicMock())
         
         mock_extract.return_value = {
             Path("img1_car.jpg"): [
                 {
                     "box": [10, 20, 5, 5],
                     "feature": np.array([1.0, 0.0]),
                     "hsv_hist": np.array([0.5]),
                     "aspect_ratio": 1.0,
                     "timestamp": 10.0,
                     "img_name": "entity_1_left_car.jpg"
                 }
             ]
         }
         
         # 1. Test with ProfilingMode.DWELL enum
         res_dwell = run_entry_exit_profiling(
             entry_dir="dummy_entry",
             exit_dir="dummy_exit",
             mode=ProfilingMode.DWELL
         )
         assert "dwell_time_matches" in res_dwell
         
         # 2. Test with ProfilingMode.OCCUPANCY enum
         res_occ = run_entry_exit_profiling(
             entry_dir="dummy_entry",
             exit_dir="dummy_exit",
             mode=ProfilingMode.OCCUPANCY
         )
         assert "occupancy_timeline" in res_occ
