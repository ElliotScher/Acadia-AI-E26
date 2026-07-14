"""
Plate Ground Truth Labeler

A simple PySide6 GUI for building a ground-truth plate-text label for a
directory of plate crop images (recursed into subdirectories), for use when
benchmarking src/processing/plate_dwellprofiler.py's OCR accuracy.

Two screens, in order:

1. Triage: every image is shown once with a "Readable"/"Unreadable" choice,
   so images no human could read anyway (too blurry, wrong angle, occluded)
   are weeded out before OCR ever runs on them - there's no point asking OCR
   or a human to transcribe something nobody can actually read.
2. Labeling: OCR (imgutils.extract_plate_text_via_ocr, the same function the
   profiler uses) runs over every image that survived triage, then each is
   shown with the OCR reading pre-filled as the label: press Enter to accept
   the reading as correct, edit the text first if OCR got it wrong, or clear
   it before pressing Enter if it turns out not to be readable after all.
   This doubles as a live OCR accuracy check, since a label that differs from
   what OCR read is a miss.

Produces a JSON file mapping each image's path (relative to the input
directory) to its ground-truth plate text ("" for images triage or labeling
marked unreadable). Progress is saved after every decision, so a session can
be closed and resumed later: relaunching against the same --output file
preloads existing decisions, skips triage for images already triaged, and
resumes labeling at the first image still awaiting a label.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.utility.imgutils import extract_plate_text_via_ocr

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Mirrors imgutils._normalize_plate_text's whitelist, so a hand-typed label
# and an OCR reading of the same plate are directly string-comparable.
_PLATE_CHAR_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def normalize_plate_text(text: str) -> str:
    """
    Uppercases hand-typed plate text and strips everything but letters and digits.

    Args:
        text (str): Raw text typed into the label field.

    Returns:
        str: The normalized, alphanumeric-only plate text (may be empty).
    """
    return "".join(ch for ch in text.upper() if ch in _PLATE_CHAR_WHITELIST)


def find_images(folder: Path) -> List[Path]:
    """
    Finds all image files in a directory, recursively.

    Args:
        folder (Path): Directory path to search.

    Returns:
        List[Path]: Sorted list of matching image file paths.
    """
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


class PlateLabelerWindow(QMainWindow):
    """
    A two-screen app: triage each image as readable/unreadable, then
    confirm/correct OCR's reading of the ones marked readable.

    self.labels values carry three states, since triage and labeling both
    write into the same dict:
        - key absent: never triaged, still needs a readable/unreadable call.
        - "" : confirmed unreadable (by triage or by a blank labeling submit).
        - None: triaged readable, but not yet given a confirmed label.
        - any other string: the confirmed, human-verified plate text.
    """

    def __init__(self, input_dir: Path, images: List[Path], output_path: Path):
        """
        Args:
            input_dir (Path): Root directory the images were found under; labels
                are keyed by each image's path relative to this root.
            images (List[Path]): Image paths to process, in display order.
            output_path (Path): JSON file to load existing labels from (if
                present) and save labels to after every decision.
        """
        super().__init__()
        self.input_dir = input_dir
        self.images = images
        self.output_path = output_path
        self.labels: Dict[str, Optional[str]] = self._load_existing_labels()
        self._current_pixmap: Optional[QPixmap] = None
        self._active_image_label: Optional[QLabel] = None
        self._ocr_cache: Dict[str, Optional[str]] = {}
        self.index = 0

        self.setWindowTitle("Plate Ground Truth Labeler")
        self.resize(900, 700)
        self._build_ui()

        self.triage_images = [
            p for p in self.images if self._rel_key(p) not in self.labels
        ]
        self.triage_index = 0
        if self.triage_images:
            self.stack.setCurrentWidget(self.triage_page)
            self.triage_page.setFocus()
            self._show_triage_image()
        else:
            self._start_labeling_stage()

    # ---------------- persistence ----------------

    def _load_existing_labels(self) -> Dict[str, Optional[str]]:
        if not self.output_path.exists():
            return {}
        try:
            with open(self.output_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        with open(self.output_path, "w") as f:
            json.dump(self.labels, f, indent=2, sort_keys=True)

    def _rel_key(self, path: Path) -> str:
        return path.relative_to(self.input_dir).as_posix()

    # ---------------- OCR ----------------

    def _get_ocr_prediction(self, path: Path) -> Optional[str]:
        key = self._rel_key(path)
        if key not in self._ocr_cache:
            img = cv2.imread(str(path))
            self._ocr_cache[key] = (
                extract_plate_text_via_ocr(img) if img is not None else None
            )
        return self._ocr_cache[key]

    def _run_ocr_pass(self, images: List[Path]) -> None:
        """
        Runs OCR over every given image up front (cancelable), so the reading
        is already cached and ready to pre-fill by the time each is shown.
        Any image left uncached by canceling early is simply OCR'd lazily the
        first time it's actually displayed.
        """
        progress = QProgressDialog(
            "Running OCR on plate crops...", "Cancel", 0, len(images), self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        for i, path in enumerate(images):
            if progress.wasCanceled():
                break
            self._get_ocr_prediction(path)
            progress.setValue(i + 1)
        progress.close()
        # Reclaim window activation/focus from the now-closed dialog so the
        # label page's arrow-key shortcuts and text field are usable right away.
        self.activateWindow()

    def _ocr_agreement_stats(self) -> Tuple[int, int]:
        """
        Returns (agreed, total) over confirmed labels that have a cached OCR
        prediction: how many of them the human-confirmed label matches
        exactly what OCR read.
        """
        agreed = 0
        total = 0
        for key, label in self.labels.items():
            if label is None or key not in self._ocr_cache:
                continue
            total += 1
            if (self._ocr_cache[key] or "") == label:
                agreed += 1
        return agreed, total

    # ---------------- UI construction ----------------

    def _build_ui(self) -> None:
        self.stack = QStackedWidget()
        self.triage_page = self._build_triage_page()
        self.label_page = self._build_label_page()
        self.stack.addWidget(self.triage_page)
        self.stack.addWidget(self.label_page)
        self.setCentralWidget(self.stack)

    def _build_triage_page(self) -> QWidget:
        page = QWidget()
        page.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout(page)

        self.triage_image_label = QLabel(alignment=Qt.AlignCenter)
        self.triage_image_label.setMinimumSize(200, 200)
        layout.addWidget(self.triage_image_label, stretch=1)

        self.triage_path_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.triage_path_label)

        self.triage_progress_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.triage_progress_label)

        hint = QLabel(
            "Could a human read a plate in this image? "
            "(← previous, ↓ unreadable, → readable)",
            alignment=Qt.AlignCenter,
        )
        layout.addWidget(hint)

        button_row = QHBoxLayout()
        prev_btn = QPushButton("<- Previous")
        prev_btn.clicked.connect(self._on_triage_previous)
        unreadable_btn = QPushButton("Unreadable (↓)")
        unreadable_btn.clicked.connect(self._on_triage_unreadable)
        readable_btn = QPushButton("Readable ->")
        readable_btn.clicked.connect(self._on_triage_readable)
        button_row.addWidget(prev_btn)
        button_row.addWidget(unreadable_btn)
        button_row.addWidget(readable_btn)
        layout.addLayout(button_row)

        left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), page)
        left_shortcut.activated.connect(self._on_triage_previous)
        left_shortcut.setContext(Qt.WidgetWithChildrenShortcut)

        right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), page)
        right_shortcut.activated.connect(self._on_triage_readable)
        right_shortcut.setContext(Qt.WidgetWithChildrenShortcut)

        unreadable_shortcut = QShortcut(QKeySequence(Qt.Key_Down), page)
        unreadable_shortcut.activated.connect(self._on_triage_unreadable)
        unreadable_shortcut.setContext(Qt.WidgetWithChildrenShortcut)

        return page

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setMinimumSize(200, 200)
        layout.addWidget(self.image_label, stretch=1)

        self.path_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.path_label)

        self.ocr_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.ocr_label)

        self.progress_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        self.text_entry = QLineEdit()
        self.text_entry.setPlaceholderText(
            "OCR's reading is pre-filled - press Enter to confirm it, edit it "
            "first if it's wrong, or clear it if no human can read the plate"
        )
        self.text_entry.returnPressed.connect(self._on_next)
        layout.addWidget(self.text_entry)

        button_row = QHBoxLayout()
        prev_btn = QPushButton("<- Previous")
        prev_btn.clicked.connect(self._on_previous)
        unreadable_btn = QPushButton("Unreadable")
        unreadable_btn.clicked.connect(self._on_unreadable)
        next_btn = QPushButton("Next ->")
        next_btn.clicked.connect(self._on_next)
        button_row.addWidget(prev_btn)
        button_row.addWidget(unreadable_btn)
        button_row.addWidget(next_btn)
        layout.addLayout(button_row)

        # No Left/Right shortcuts here (unlike the triage page): the text
        # field is focused essentially the whole time this page is visible,
        # and QLineEdit itself consumes arrow keys for cursor movement before
        # they'd ever reach a QShortcut, so they'd silently never fire.
        # Enter (returnPressed above) and the buttons cover navigation.

        return page

    # ---------------- triage stage ----------------

    def _show_triage_image(self) -> None:
        path = self.triage_images[self.triage_index]
        key = self._rel_key(path)

        self._active_image_label = self.triage_image_label
        self._current_pixmap = QPixmap(str(path))
        if self._current_pixmap.isNull():
            self.triage_image_label.setPixmap(QPixmap())
            self.triage_image_label.setText(f"Failed to load image:\n{path}")
        else:
            self._rescale_pixmap()

        self.triage_path_label.setText(key)
        self.triage_progress_label.setText(
            f"Triage: image {self.triage_index + 1} / {len(self.triage_images)}"
        )

    def _advance_triage(self) -> None:
        if self.triage_index < len(self.triage_images) - 1:
            self.triage_index += 1
            self._show_triage_image()
        else:
            self._start_labeling_stage()

    def _on_triage_readable(self) -> None:
        key = self._rel_key(self.triage_images[self.triage_index])
        self.labels[key] = None
        self._save()
        self._advance_triage()

    def _on_triage_unreadable(self) -> None:
        key = self._rel_key(self.triage_images[self.triage_index])
        self.labels[key] = ""
        self._save()
        self._advance_triage()

    def _on_triage_previous(self) -> None:
        if self.triage_index > 0:
            self.triage_index -= 1
        self._show_triage_image()

    def _start_labeling_stage(self) -> None:
        self.images = [
            p for p in self.images if self.labels.get(self._rel_key(p)) != ""
        ]
        self.stack.setCurrentWidget(self.label_page)

        if not self.images:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Every image was marked unreadable in triage.")
            self.path_label.setText("")
            self.ocr_label.setText("")
            self.progress_label.setText("")
            self.text_entry.setEnabled(False)
            return

        self.index = self._first_unlabeled_index()
        self._run_ocr_pass(self.images)
        self._show_current_image()

    # ---------------- labeling stage ----------------

    def _first_unlabeled_index(self) -> int:
        for i, path in enumerate(self.images):
            if self.labels.get(self._rel_key(path)) is None:
                return i
        return 0

    def _commit_forward(self) -> None:
        """
        Records the current image's label. The text field is pre-filled with
        OCR's reading, so submitting it unchanged confirms OCR was correct;
        editing it records a correction; and blank input is recorded as an
        explicit "" (unreadable by a human), rather than left unlabeled,
        since pressing Enter/Next past an image is the normal way to say
        "I looked at this one and there's nothing to read."
        """
        text = self.text_entry.text().strip()
        key = self._rel_key(self.images[self.index])
        self.labels[key] = normalize_plate_text(text) if text else ""

    def _commit_if_typed(self) -> None:
        """
        Records the current image's label only if something was actually
        typed. Used when stepping backward to review, so merely glancing at
        an already-labeled image without retyping its text doesn't blank it
        out to "unreadable".
        """
        text = self.text_entry.text().strip()
        if not text:
            return
        key = self._rel_key(self.images[self.index])
        self.labels[key] = normalize_plate_text(text)

    def _advance(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
        self._show_current_image()

    def _on_next(self) -> None:
        self._commit_forward()
        self._save()
        self._advance()

    def _on_previous(self) -> None:
        self._commit_if_typed()
        self._save()
        if self.index > 0:
            self.index -= 1
        self._show_current_image()

    def _on_unreadable(self) -> None:
        self.text_entry.clear()
        self._on_next()

    def _show_current_image(self) -> None:
        path = self.images[self.index]
        key = self._rel_key(path)

        self._active_image_label = self.image_label
        self._current_pixmap = QPixmap(str(path))
        if self._current_pixmap.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"Failed to load image:\n{path}")
        else:
            self._rescale_pixmap()

        self.path_label.setText(key)

        ocr_prediction = self._get_ocr_prediction(path)
        self.ocr_label.setText(f"OCR read: {ocr_prediction or '(no reading)'}")

        agreed, total = self._ocr_agreement_stats()
        agreement_text = f"OCR agreed {agreed}/{total}" if total else "OCR agreed -"
        labeled_count = sum(
            1 for p in self.images if self.labels.get(self._rel_key(p)) is not None
        )
        self.progress_label.setText(
            f"Image {self.index + 1} / {len(self.images)}    |    "
            f"Labeled {labeled_count} / {len(self.images)}    |    "
            f"{agreement_text}"
        )

        existing = self.labels.get(key)
        self.text_entry.setText(existing if existing is not None else (ocr_prediction or ""))
        self.text_entry.setFocus()
        self.text_entry.selectAll()

    # ---------------- shared ----------------

    def _rescale_pixmap(self) -> None:
        if (
            self._active_image_label is not None
            and self._current_pixmap is not None
            and not self._current_pixmap.isNull()
        ):
            scaled = self._current_pixmap.scaled(
                self._active_image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self._active_image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale_pixmap()

    def closeEvent(self, event) -> None:
        if self.stack.currentWidget() is self.label_page and self.images:
            self._commit_forward()
            self._save()
        super().closeEvent(event)


def main() -> None:
    """
    Main CLI entry point for the plate ground truth labeler app.

    Raises:
        SystemExit: If the input directory is missing/invalid or contains no images.
    """
    parser = argparse.ArgumentParser(
        description="Triage a directory of plate crop images (recursed into "
        "subdirectories) into readable/unreadable, then run OCR over the "
        "readable ones and confirm/correct each reading by hand, to produce a "
        "ground-truth JSON for benchmarking plate_dwellprofiler.py's OCR accuracy."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory of plate crop images to label.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to the ground-truth JSON file. Loaded to resume an existing "
        "session if it already exists, and overwritten after every decision. "
        "Defaults to ground_truth.json inside input_dir.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"{input_dir} is not a directory.")

    images = find_images(input_dir)
    if not images:
        parser.error(f"No images found under {input_dir}.")

    output_path = Path(args.output) if args.output else input_dir / "ground_truth.json"

    app = QApplication(sys.argv)
    window = PlateLabelerWindow(input_dir, images, output_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
