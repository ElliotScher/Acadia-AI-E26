from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
import math
from random import randint

from db.models import Image, Instance, Entity
from detection.cluster import process_clusters


class ClusterDialog(QtWidgets.QDialog):
    finish = QtCore.Signal()

    def __init__(self, session: Session, images: list[Image], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.session = session
        self.images: list[Image] = images

        self.setWindowTitle("Analyze " + str(len(self.images)) + " images")

        layout = QtWidgets.QVBoxLayout()

        distanceLayout = QtWidgets.QHBoxLayout()
        distanceLayout.addWidget(QtWidgets.QLabel("Maximum distance"))
        self.distance = QtWidgets.QSpinBox()
        self.distance.setRange(0, 500)
        self.distance.setValue(60)
        distanceLayout.addWidget(self.distance)
        layout.addLayout(distanceLayout)

        ratioLayout = QtWidgets.QHBoxLayout()
        ratioLayout.addWidget(QtWidgets.QLabel("Maximum size ratio"))
        self.ratio = QtWidgets.QDoubleSpinBox()
        self.ratio.setRange(0, 10)
        self.ratio.setValue(2.5)
        self.ratio.setSingleStep(0.1)
        ratioLayout.addWidget(self.ratio)
        layout.addLayout(ratioLayout)

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
            clusters = process_clusters(
                image.to_detection_result(self.session),
                self.distance.value(),
                self.ratio.value(),
            )

            for cluster in clusters:
                clusterId = randint(0, 99999999)

                for detection in cluster.detections.boxes:
                    entity = self.session.scalar(
                        select(Entity).where(Entity.id == detection[1])
                    )

                    if entity is not None:
                        entity.cluster = clusterId
                        self.session.add(entity)

        self.session.commit()
        self.finish.emit()
        self.accept()
