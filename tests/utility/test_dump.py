import datetime as dt
import os

from src.utility.dump import dumpcamera


def _set_mtime(path, timestamp: dt.datetime):
    epoch = timestamp.timestamp()
    os.utime(path, (epoch, epoch))


def test_dumpcamera_renames_by_mtime(tmp_path):
    camdir = tmp_path / "cam1"
    camdir.mkdir()
    photo = camdir / "IMG_0001.jpg"
    photo.write_bytes(b"fake-jpg-data")
    _set_mtime(photo, dt.datetime(2026, 5, 1, 8, 30, 15))

    destination = tmp_path / "dest"
    destination.mkdir()

    dumpcamera(str(camdir), "cam1", str(destination))

    expected = destination / "cam1" / "2026-05-01" / "08-30-15.jpg"
    assert expected.exists()
    assert expected.read_bytes() == b"fake-jpg-data"


def test_dumpcamera_recurses_into_subdirectories(tmp_path):
    camdir = tmp_path / "cam2"
    (camdir / "subfolder").mkdir(parents=True)
    photo = camdir / "subfolder" / "IMG_0002.jpg"
    photo.write_bytes(b"nested-photo")
    _set_mtime(photo, dt.datetime(2026, 5, 2, 9, 0, 0))

    destination = tmp_path / "dest"
    destination.mkdir()

    dumpcamera(str(camdir), "cam2", str(destination))

    expected = destination / "cam2" / "2026-05-02" / "09-00-00.jpg"
    assert expected.exists()


def test_dumpcamera_multiple_photos_same_day(tmp_path):
    camdir = tmp_path / "cam3"
    camdir.mkdir()
    photo1 = camdir / "a.jpg"
    photo1.write_bytes(b"first")
    _set_mtime(photo1, dt.datetime(2026, 5, 3, 10, 0, 0))

    photo2 = camdir / "b.jpg"
    photo2.write_bytes(b"second")
    _set_mtime(photo2, dt.datetime(2026, 5, 3, 11, 0, 0))

    destination = tmp_path / "dest"
    destination.mkdir()

    dumpcamera(str(camdir), "cam3", str(destination))

    day_dir = destination / "cam3" / "2026-05-03"
    assert (day_dir / "10-00-00.jpg").read_bytes() == b"first"
    assert (day_dir / "11-00-00.jpg").read_bytes() == b"second"


def test_dumpcamera_preserves_original_file(tmp_path):
    camdir = tmp_path / "cam4"
    camdir.mkdir()
    photo = camdir / "IMG_0003.jpg"
    photo.write_bytes(b"copy-not-move")
    _set_mtime(photo, dt.datetime(2026, 5, 4, 12, 0, 0))

    destination = tmp_path / "dest"
    destination.mkdir()

    dumpcamera(str(camdir), "cam4", str(destination))

    assert photo.exists()
