from dataclasses import dataclass
from typing import Literal

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


class ExportDialog(QtWidgets.QDialog):
    startExport = QtCore.Signal(ExportOptions)

    def __init__(self, filtered: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filtered = filtered

        self.setWindowTitle("Export")

        layout = QtWidgets.QVBoxLayout()

        self.exportModeImages = QtWidgets.QRadioButton("Row per Image")
        self.exportModeImages.setChecked(True)
        layout.addWidget(self.exportModeImages)
        self.exportModeEntities = QtWidgets.QRadioButton("Row per Entity")
        layout.addWidget(self.exportModeEntities)
        self.exportModeClusters = QtWidgets.QRadioButton("Row per Cluster")
        layout.addWidget(self.exportModeClusters)
        self.exportModeInterval = QtWidgets.QRadioButton("Row per Time Interval")
        self.exportModeInterval.toggled.connect(self.enableIntervalTime)
        layout.addWidget(self.exportModeInterval)

        intervalTimeLayout = QtWidgets.QHBoxLayout()
        self.exportModeIntervalTime = QtWidgets.QSpinBox()
        self.exportModeIntervalTime.setRange(1, 60)
        self.exportModeIntervalTime.setValue(15)
        self.exportModeIntervalTime.setMaximumWidth(50)
        self.exportModeIntervalTime.setEnabled(False)
        intervalTimeLayout.addWidget(self.exportModeIntervalTime)
        intervalTimeLayout.addWidget(QtWidgets.QLabel("minutes"))
        layout.addLayout(intervalTimeLayout)

        layout.addSpacing(10)

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
                )
            )

        self.accept()
