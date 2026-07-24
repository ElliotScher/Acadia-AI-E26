from PySide6 import QtCore, QtWidgets
from sqlalchemy.orm import Session
import datetime as dt

from db.models import Image
from detection.entity_iou_tracking import entity_iou_tracking


class IOUTrackingDialog(QtWidgets.QDialog):
    finish = QtCore.Signal()

    def __init__(self, session: Session, images: list[Image], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.session = session
        self.images: list[Image] = images

        self.setWindowTitle("Analyze " + str(len(self.images)) + " images")

        layout = QtWidgets.QVBoxLayout()

        trackingGapLayout = QtWidgets.QHBoxLayout()
        trackingGapLayout.addWidget(QtWidgets.QLabel("Maximum tracking gap"))
        self.trackingGap = QtWidgets.QDoubleSpinBox()
        self.trackingGap.setRange(0.1, 5)
        self.trackingGap.setValue(1)
        self.trackingGap.setSingleStep(0.1)
        trackingGapLayout.addWidget(self.trackingGap)
        trackingGapLayout.addWidget(QtWidgets.QLabel("seconds"))
        layout.addLayout(trackingGapLayout)

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
        entity_iou_tracking(
            self.session, self.images, dt.timedelta(seconds=self.trackingGap.value())
        )

        self.finish.emit()
        self.accept()
