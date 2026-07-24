from PySide6 import QtCore, QtWidgets
from sqlalchemy.orm import Session

from db.models import Image
from detection.bike_rider_merging import merge_bikes_riders


class BikeRiderMergeDialog(QtWidgets.QDialog):
    finish = QtCore.Signal()

    def __init__(self, session: Session, images: list[Image], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.session = session
        self.images: list[Image] = images

        self.setWindowTitle("Analyze " + str(len(self.images)) + " images")

        layout = QtWidgets.QVBoxLayout()

        thresholdLayout = QtWidgets.QHBoxLayout()
        thresholdLayout.addWidget(QtWidgets.QLabel("Minimum overlap threshold"))
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setSuffix("%")
        self.threshold.setRange(0, 1)
        self.threshold.setValue(0.2)
        self.threshold.setSingleStep(0.01)
        thresholdLayout.addWidget(self.threshold)
        layout.addLayout(thresholdLayout)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.analyze)
        self.buttons.rejected.connect(self.reject)

        self.setLayout(layout)

    @QtCore.Slot()
    def analyze(self):
        for image in self.images:
            merge_bikes_riders(self.session, image, self.threshold.value())

        self.finish.emit()
        self.accept()
