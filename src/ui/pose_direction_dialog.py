from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
from pathlib import Path
import functools
import math

from db.models import Image, Instance
import utility.parallel as upl
from detection.yolo import CLASS_ID_MAPPING, TARGET_CLASSES, load_model
from detection.pose_direction import Direction, process_single_image


class PoseDirectionDialog(QtWidgets.QDialog):
    def __init__(self, session: Session, images: list[Image], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.threadsRunning = 0
        self.results: list[dict[tuple[int, int], list[Direction]]] = list()
        self.session = session
        self.images: list[Image] = images

        self.setWindowTitle("Analyze Direction")

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

        threadCount = self.threadCount.value()

        instancesList: list[tuple[int, int, str, tuple[int, int, int, int]]] = []
        for image in self.images:
            for instance in image.get_instances(self.session):
                instancesList.append(
                    (
                        instance.image_id,
                        instance.entity_id,
                        image.path,
                        (
                            instance.x,
                            instance.y,
                            instance.x + instance.width,
                            instance.y + instance.height,
                        ),
                    )
                )

        instancesPerThread = math.ceil(len(instancesList) / threadCount)

        self.threadsRunning = 0
        self.results: list[dict[tuple[int, int], list[Direction]]] = list()

        for i in range(threadCount):
            instances = instancesList[
                (i * instancesPerThread) : ((i + 1) * instancesPerThread)
            ]
            thread = upl.Async(
                "Analysis " + str(i + 1),
                functools.partial(
                    self.analyzeThread,
                    instances,
                    self.minConfidence.value(),
                    [0],
                    self.minPoints.value(),
                ),
            )
            thread.result.connect(self.finishAnalysis)
            thread.start()
            self.threadsRunning += 1

        self.accept()

    def analyzeThread(
        self,
        instances: list[tuple[int, int, str, tuple[int, int, int, int]]],
        minConfidence: float,
        targetClasses: list[int],
        minPoints: int,
    ) -> dict[tuple[int, int], list[Direction]]:
        results: dict[tuple[int, int], list[Direction]] = dict()
        for instance in instances:
            results[(instance[0], instance[1])] = process_single_image(
                load_model("yolo26s-pose.pt"),
                Path(instance[2]).resolve(),
                Path(),
                Path(),
                False,
                minConfidence,
                targetClasses,
                box=instance[3],
                min_points=minPoints,
            )
            upl.Async.progress(len(results) / len(instances))

        return results

    @QtCore.Slot()
    def finishAnalysis(self, result: dict[tuple[int, int], list[Direction]]):
        self.results.append(result)
        self.threadsRunning -= 1

        if self.threadsRunning == 0:
            for result in self.results:
                for instanceIds in result.keys():
                    instance: Instance = self.session.scalar(
                        select(Instance).where(
                            Instance.image_id == instanceIds[0],
                            Instance.entity_id == instanceIds[1],
                        )
                    )  # ty:ignore[invalid-assignment]

                    if len(result[instanceIds]) > 0:
                        instance.direction_fb = result[instanceIds][0].front_back
                        instance.direction_lr = result[instanceIds][0].left_right
                        self.session.add(instance)

            self.session.commit()
