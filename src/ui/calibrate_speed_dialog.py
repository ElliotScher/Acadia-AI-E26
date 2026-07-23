from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
import math
import datetime as dt

from db.models import Entity


class CalibrateSpeedDialog(QtWidgets.QDialog):
    finish = QtCore.Signal()

    def __init__(self, session: Session, knownEntity: Entity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.session = session
        self.knownEntity = knownEntity

        self.setWindowTitle("Calibrate Speeds")

        layout = QtWidgets.QVBoxLayout()

        speedLayout = QtWidgets.QHBoxLayout()
        speedLayout.addWidget(QtWidgets.QLabel("Known entity speed"))
        self.speed = QtWidgets.QSpinBox()
        self.speed.setRange(1, 100)
        self.speed.setValue(20)
        speedLayout.addWidget(self.speed)
        speedLayout.addWidget(QtWidgets.QLabel("mph"))
        layout.addLayout(speedLayout)

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
        entities = self.session.scalars(select(Entity)).all()

        scale = self.speed.value() / self.knownEntity.rawSpeed

        for entity in entities:
            if entity.rawSpeed is not None:
                entity.speed = entity.rawSpeed * scale
                self.session.add(entity)

        self.session.commit()
        self.finish.emit()
        self.accept()
