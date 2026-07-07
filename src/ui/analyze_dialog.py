from PySide6 import QtCore, QtGui, QtWidgets
from detection.yolo import CLASS_ID_MAPPING, TARGET_CLASSES
from sqlalchemy.orm import Session
from db.models import Image
import math


class AnalyzeDialog(QtWidgets.QDialog):
    def __init__(
        self, session: Session, yoloModel, images: list[Image], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.session = session
        self.yoloModel = yoloModel
        self.images = images

        self.setWindowTitle("Analyze " + str(len(self.images)) + " images")

        layout = QtWidgets.QVBoxLayout()

        self.typesGroup = QtWidgets.QGroupBox("Target Types")
        self.typesGroupLayout = QtWidgets.QHBoxLayout()
        self.typesGroup.setLayout(self.typesGroupLayout)

        self.typesScroll = QtWidgets.QScrollArea()
        self.typesScroll.setWidgetResizable(True)
        self.typesScroll.setMinimumHeight(100)
        self.typesScroll.setMaximumHeight(200)
        self.typesContents = QtWidgets.QWidget()
        self.typesScroll.setWidget(self.typesContents)
        self.typesGroupLayout.addWidget(self.typesScroll)
        self.typesLayout = QtWidgets.QGridLayout()
        self.typesContents.setLayout(self.typesLayout)
        self.typesCheckboxes = []
        for id in CLASS_ID_MAPPING:
            checkbox = QtWidgets.QCheckBox(CLASS_ID_MAPPING[id].title())
            if id in TARGET_CLASSES:
                checkbox.setChecked(True)
            self.typesCheckboxes.append(checkbox)
            self.typesLayout.addWidget(checkbox, math.floor(id / 3), id % 3)

        layout.addWidget(self.typesGroup)

        minConfidenceLayout = QtWidgets.QHBoxLayout()
        minConfidenceLayout.addWidget(QtWidgets.QLabel("Minimum confidence"))
        self.minConfidence = QtWidgets.QDoubleSpinBox()
        self.minConfidence.setSuffix("%")
        self.minConfidence.setRange(0, 1)
        self.minConfidence.setValue(0.25)
        self.minConfidence.setSingleStep(0.05)
        minConfidenceLayout.addWidget(self.minConfidence)
        layout.addLayout(minConfidenceLayout)

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
        target_classes = []
        for i in range(len(self.typesCheckboxes)):
            if self.typesCheckboxes[i].isChecked():
                target_classes.append(i)

        for image in self.images:
            image.analyze(
                self.session, self.yoloModel, self.minConfidence.value(), target_classes
            )

        self.accept()
