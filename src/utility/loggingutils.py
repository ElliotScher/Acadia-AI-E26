import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple


def setup_logging_and_paths(
    parser: argparse.ArgumentParser, logger: logging.Logger
) -> Tuple[argparse.Namespace, Optional[Path], Optional[Path]]:
    """
    Configures debug logging and resolves/validates input/output directories.

    Args:
        parser (argparse.ArgumentParser): The parser instance to add --debug to.
        logger (logging.Logger): Logger instance for error reporting.

    Returns:
        Tuple[argparse.Namespace, Optional[Path], Optional[Path]]: A tuple containing the parsed arguments,
            resolved input directory path, and resolved output directory path.

    Raises:
        SystemExit: If the resolved input directory does not exist or is not a directory.
    """
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging.",
    )
    args = parser.parse_args()

    # Configure Logging level based on flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    input_folder = (
        Path(args.input_dir).resolve()
        if getattr(args, "input_dir", None) is not None
        else None
    )
    output_folder = (
        Path(args.output_dir).resolve()
        if getattr(args, "output_dir", None) is not None
        else None
    )

    if input_folder is not None and not input_folder.is_dir():
        logger.error(
            "Input directory '%s' does not exist or is not a directory.", input_folder
        )
        sys.exit(1)

    return args, input_folder, output_folder
