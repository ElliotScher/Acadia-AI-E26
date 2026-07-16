import argparse
import logging

import pytest

from src.utility.loggingutils import setup_logging_and_paths


def _make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default=None)
    return parser


def test_setup_logging_and_paths_resolves_valid_input_dir(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    monkeypatch.setattr("sys.argv", ["prog", str(input_dir), str(output_dir)])

    parser = _make_parser()
    logger = logging.getLogger("test_setup_logging_valid")
    args, resolved_input, resolved_output = setup_logging_and_paths(parser, logger)

    assert resolved_input == input_dir.resolve()
    assert resolved_output == output_dir.resolve()
    assert args.debug is False


def test_setup_logging_and_paths_missing_input_dir_exits(tmp_path, monkeypatch):
    missing_dir = tmp_path / "does-not-exist"

    monkeypatch.setattr("sys.argv", ["prog", str(missing_dir)])

    parser = _make_parser()
    logger = logging.getLogger("test_setup_logging_missing")

    with pytest.raises(SystemExit) as exc_info:
        setup_logging_and_paths(parser, logger)
    assert exc_info.value.code == 1


def test_setup_logging_and_paths_no_input_dir_attribute(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])

    parser = argparse.ArgumentParser()
    logger = logging.getLogger("test_setup_logging_no_attr")

    args, resolved_input, resolved_output = setup_logging_and_paths(parser, logger)

    assert resolved_input is None
    assert resolved_output is None


def test_setup_logging_and_paths_debug_flag_sets_debug_level(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    monkeypatch.setattr("sys.argv", ["prog", str(input_dir), "--debug"])

    parser = _make_parser()
    logger = logging.getLogger("test_setup_logging_debug")
    args, _, _ = setup_logging_and_paths(parser, logger)

    assert args.debug is True
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_and_paths_default_level_is_info(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()

    monkeypatch.setattr("sys.argv", ["prog", str(input_dir)])

    parser = _make_parser()
    logger = logging.getLogger("test_setup_logging_info")
    setup_logging_and_paths(parser, logger)

    assert logging.getLogger().level == logging.INFO
