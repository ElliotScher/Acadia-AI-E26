from setuptools.config.setupcfg import Target
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
import logging
import math
from pathlib import Path
import functools

from db.models import Image, Instance, Entity
from detection.classes import (
    CLASS_ID_MAPPING,
    TARGET_CLASSES
)
from detection.image_yolo import (
    process_images,
    DetectionResult,
)
import utility.parallel as upl


class AnalyzeDialog(QtWidgets.QDialog):
    def __init__(self, session: Session, images: list[Image], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threadsRunning = 0
        self.results: list[DetectionResult] = list()

        self.session = session
        self.images: list[tuple[int, str]] = list(map(lambda i: (i.id, i.path), images))

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

        threadCountLayout = QtWidgets.QHBoxLayout()
        threadCountLayout.addWidget(QtWidgets.QLabel("Threads"))
        self.threadCount = QtWidgets.QSpinBox()
        self.threadCount.setRange(1, QtCore.QThread.idealThreadCount())
        threadCountLayout.addWidget(self.threadCount)
        layout.addLayout(threadCountLayout)

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
        assert self.threadsRunning == 0

        targetClasses = []
        for i in range(len(self.typesCheckboxes)):
            if self.typesCheckboxes[i].isChecked():
                targetClasses.append(i)

        threadCount = self.threadCount.value()
        imagesPerThread = math.ceil(len(self.images) / threadCount)

        self.threadsRunning = 0
        self.results: list[dict[int, list[DetectionResult]]] = list()

        for i in range(threadCount):
            images = self.images[(i * imagesPerThread) : ((i + 1) * imagesPerThread)]
            thread = upl.Async(
                "Analysis " + str(i + 1),
                functools.partial(
                    self.analyzeThread,
                    images,
                    self.minConfidence.value(),
                    targetClasses,
                ),
            )
            thread.result.connect(self.finishAnalysis)
            thread.start()
            self.threadsRunning += 1

        self.accept()

    def analyzeThread(
        self,
        images: list[tuple[int, str]],
        minConfidence: float,
        targetClasses: list[int],
    ) -> list[DetectionResult]:
        results = process_images(
            [Path(x[1]) for x in images],
            "yolo26s.pt",
            upl.ProgressTracker(len(images)),
            None,
            minConfidence,
            targetClasses,
        )
        return results

    @QtCore.Slot()
    def finishAnalysis(self, result: list[DetectionResult]):
        self.results.extend(result)
        self.threadsRunning -= 1
        if self.threadsRunning > 1:
            return
        for r in self.results:
            image = self.session.scalar(select(Image).where(Image.path == r.image_path.as_posix()))
            if not image:
                logger = logging.Logger("analyze_dialog")
                logger.log(logging.WARN, f"couldn't find image {r.image_path} in the database")
                return
            if image.analyzed:
                for instance in image.get_instances(self.session):
                    self.session.delete(instance)
            for detection in r.boxes:
                entity = Entity()
                box = detection[0]
                typeId = detection[1]
                conf = detection[2]
                instance = Instance(
                    image=image,
                    entity=entity,
                    x=box.x,
                    y=box.y,
                    width=box.w,
                    height=box.h,
                    type_id=typeId,
                    confidence=conf,
                )
                self.session.add_all((entity, instance))

            image.analyzed = True
            self.session.add(image)

        self.session.commit()
