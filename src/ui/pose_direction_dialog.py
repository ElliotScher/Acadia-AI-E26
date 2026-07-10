from PySide6 import QtCore, QtGui, QtWidgets
from detection.yolo import CLASS_ID_MAPPING, TARGET_CLASSES
from sqlalchemy.orm import Session
from db.models import Image
import math


class PoseDirectionDialog(QtWidgets.QDialog):
    def __init__(
        self, session: Session, yoloModel, images: list[Image], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.session = session
        self.yoloModel = yoloModel
        self.images = images

        self.setWindowTitle(
            "Analyze Direction"
        )

        layout = QtWidgets.QVBoxLayout()

        minConfidenceLayout = QtWidgets.QHBoxLayout()
        minConfidenceLayout.addWidget(QtWidgets.QLabel("Minimum confidence"))
        self.minConfidence = QtWidgets.QDoubleSpinBox()
        self.minConfidence.setSuffix("%")
        self.minConfidence.setRange(0, 1)
        self.minConfidence.setValue(0.25)
        self.minConfidence.setSingleStep(0.05)
        minConfidenceLayout.addWidget(self.minConfidence)
        layout.addLayout(minConfidenceLayout)

        minPointsLayout = QtWidgets.QHBoxLayout()
        minPointsLayout.addWidget(QtWidgets.QLabel("Minimum landmark points"))
        self.minPoints = QtWidgets.QSpinBox()
        self.minPoints.setRange(1, 4)
        self.minPoints.setValue(2)
        minPointsLayout.addWidget(self.minPoints)
        layout.addLayout(minPointsLayout)

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
            for instance in image.get_instances(self.session):
                instance.analyze_pose_direction(
                    self.session,
                    self.yoloModel,
                    self.minConfidence.value(),
                    self.minPoints.value(),
                )

        self.accept()
