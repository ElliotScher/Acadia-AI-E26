from unittest.mock import patch

import pytest

from src.utility.fetch_vehicle_pose_weights import fetch_vehicle_pose_weights


def test_fetch_skips_download_if_already_present(tmp_path):
    dest = tmp_path / "last.pt"
    dest.write_bytes(b"already here")

    with patch("src.utility.fetch_vehicle_pose_weights.gdown.download") as mock_download:
        result = fetch_vehicle_pose_weights(dest)

    mock_download.assert_not_called()
    assert result == dest
    assert dest.read_bytes() == b"already here"


def test_fetch_downloads_when_missing(tmp_path):
    dest = tmp_path / "nested" / "last.pt"

    def fake_download(url, output, quiet):
        from pathlib import Path

        Path(output).write_bytes(b"downloaded weights")

    with patch(
        "src.utility.fetch_vehicle_pose_weights.gdown.download", side_effect=fake_download
    ) as mock_download:
        result = fetch_vehicle_pose_weights(dest)

    mock_download.assert_called_once()
    assert result == dest
    assert dest.exists()
    assert dest.read_bytes() == b"downloaded weights"


def test_fetch_force_redownloads_even_if_present(tmp_path):
    dest = tmp_path / "last.pt"
    dest.write_bytes(b"stale")

    def fake_download(url, output, quiet):
        from pathlib import Path

        Path(output).write_bytes(b"fresh")

    with patch(
        "src.utility.fetch_vehicle_pose_weights.gdown.download", side_effect=fake_download
    ) as mock_download:
        fetch_vehicle_pose_weights(dest, force=True)

    mock_download.assert_called_once()
    assert dest.read_bytes() == b"fresh"


def test_fetch_raises_if_download_produces_no_file(tmp_path):
    dest = tmp_path / "last.pt"

    with patch("src.utility.fetch_vehicle_pose_weights.gdown.download"):
        with pytest.raises(RuntimeError):
            fetch_vehicle_pose_weights(dest)
