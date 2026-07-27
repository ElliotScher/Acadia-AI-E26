from typing import Literal
from dataclasses import dataclass
from PySide6 import QtCore, QtWidgets


@dataclass
class ExportOptions:
    mode: (
        Literal["images"]
        | Literal["interval"]
        | Literal["clusters"]
        | Literal["entities"]
    )
    path: str
    filtered: bool
    interval: int
    open: bool
    separateDirections: bool


class ExportDialog(QtWidgets.QDialog):
    startExport = QtCore.Signal(ExportOptions)

    def __init__(self, filtered: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filtered = filtered

        self.setWindowTitle("Export")

        layout = QtWidgets.QVBoxLayout()

        self.exportModeImages = QtWidgets.QRadioButton("Row per Image", self)
        self.exportModeImages.setChecked(True)
        self.exportModeImages.toggled.connect(self.enableSeparateDirections)
        layout.addWidget(self.exportModeImages)
        self.exportModeEntities = QtWidgets.QRadioButton("Row per Entity", self)
        layout.addWidget(self.exportModeEntities)
        self.exportModeClusters = QtWidgets.QRadioButton("Row per Cluster", self)
        layout.addWidget(self.exportModeClusters)
        self.exportModeInterval = QtWidgets.QRadioButton("Row per Time Interval", self)
        self.exportModeInterval.toggled.connect(self.enableIntervalTime)
        layout.addWidget(self.exportModeInterval)

        self.exportModeIntervalTime = QtWidgets.QSpinBox(
            self, suffix=" minutes", minimum=1, maximum=60, value=15
        )
        self.exportModeIntervalTime.setEnabled(False)
        layout.addWidget(self.exportModeIntervalTime)

        self.separateDirections = QtWidgets.QCheckBox("Separate By Direction")
        layout.addWidget(self.separateDirections)
        self.openExport = QtWidgets.QCheckBox("Open When Exported")
        layout.addWidget(self.openExport)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.export)
        self.buttons.rejected.connect(self.reject)

        self.setLayout(layout)

    @QtCore.Slot()
    def enableIntervalTime(self):
        self.exportModeIntervalTime.setEnabled(self.exportModeInterval.isChecked())
        self.enableSeparateDirections()

    @QtCore.Slot()
    def enableSeparateDirections(self):
        self.separateDirections.setEnabled(
            self.exportModeImages.isChecked() or self.exportModeInterval.isChecked()
        )

    @QtCore.Slot()
    def export(self):
        mode = (
            "images"
            if self.exportModeImages.isChecked()
            else (
                "interval"
                if self.exportModeInterval.isChecked()
                else ("clusters" if self.exportModeClusters.isChecked() else "entities")
            )
        )

        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export to...", mode + ".csv"
        )

        if len(path[0]) > 0:
            self.startExport.emit(
                ExportOptions(
                    mode,
                    path[0],
                    self.filtered,
                    self.exportModeIntervalTime.value(),
                    self.openExport.isChecked(),
                    self.separateDirections.isChecked()
                )
            )

        self.accept()
