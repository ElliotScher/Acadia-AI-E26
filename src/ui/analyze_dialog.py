from typing import Callable
from setuptools.config.setupcfg import Target
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
import functools
from typing import Callable
from setuptools.config.setupcfg import Target
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.orm import Session
from sqlalchemy import select
import logging
import math
import functools
import cv2
import os
from pathlib import Path
import datetime as dt

from db.models import Image, Instance, Entity, Video
from detection.video_yolo import (
    process_videos,
    open_video_capture,
    DetectionResult as VideoDetectionResult,
)
import utility.parallel as upl
from db.models import Entity, Image, Instance
from detection.classes import CLASS_ID_MAPPING, TARGET_CLASSES
from detection.image_yolo import (
    DetectionResult,
    process_images,
)


class AnalyzeDialog(QtWidgets.QDialog):
    finished = QtCore.Signal()

    def __init__(
        self,
        session: Session,
        files: list[tuple[int, str]],
        video=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.threadsRunning = 0
        self.results: list[DetectionResult | VideoDetectionResult] = list()

        self.session = session
        self.files: list[tuple[int, str]] = files

        self.setWindowTitle(
            "Analyze " + str(len(self.files)) + (" videos" if video else " images")
        )

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

        if video:
            downsampleLayout = QtWidgets.QHBoxLayout()
            downsampleLayout.addWidget(QtWidgets.QLabel("Downsample factor"))
            self.downsampleFactor = QtWidgets.QSpinBox()
            self.downsampleFactor.setRange(1, 60)
            self.downsampleFactor.setValue(2)
            downsampleLayout.addWidget(self.downsampleFactor)
            layout.addLayout(downsampleLayout)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(lambda: self.analyze(video))
        self.buttons.rejected.connect(self.reject)

        self.setLayout(layout)

    @QtCore.Slot()
    def analyze(self, video: bool):
        assert self.threadsRunning == 0

        targetClasses = []
        for i in range(len(self.typesCheckboxes)):
            if self.typesCheckboxes[i].isChecked():
                targetClasses.append(i)

        threadCount = self.threadCount.value()
        filesPerThread = math.ceil(len(self.files) / threadCount)

        self.threadsRunning = 0
        self.results = list()

        for i in range(1 if video else threadCount):
            files = self.files[(i * filesPerThread) : ((i + 1) * filesPerThread)]
            if video:
                thread = upl.Async(
                    "Video Analysis",
                    functools.partial(
                        self.analyzeVideoThread,
                        files,
                        self.minConfidence.value(),
                        targetClasses,
                        self.downsampleFactor.value(),
                        threadCount,
                    ),
                )
                thread.result.connect(self.finishVideoAnalysis)
            else:
                thread = upl.Async(
                    "Analysis " + str(i + 1),
                    functools.partial(
                        self.analyzeThread,
                        files,
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
        _,
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

    def analyzeVideoThread(
        self,
        videos: list[tuple[int, str]],
        minConfidence: float,
        targetClasses: list[int],
        downsampleFactor: int,
        threadCount: int,
        thread: upl.Async,
    ) -> list[VideoDetectionResult]:
        total_frames = 0
        for x in videos:
            cap = open_video_capture(x[1])
            if cap.isOpened():
                try:
                    val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    frames = int(val) if isinstance(val, (int, float)) else 0
                except (TypeError, ValueError):
                    frames = 0
                total_frames += frames
                cap.release()

        results = process_videos(
            [Path(x[1]) for x in videos],
            "yolo26s.pt",
            upl.ProgressTracker(total_frames, thread),
            None,
            minConfidence,
            targetClasses,
            threadCount,
            downsample_factor=downsampleFactor,
            write_frames=True,
        )
        return results

    @QtCore.Slot()
    def finishAnalysis(self, result: list[DetectionResult]):
        self.results.extend(result)
        self.threadsRunning -= 1
        if self.threadsRunning > 1:
            return
        for r in self.results:
            if not isinstance(r, DetectionResult):
                continue

            image = self.session.scalar(
                select(Image).where(Image.path == os.path.normpath(r.image_path.absolute()))
            )

            if not image:
                logger = logging.Logger("analyze_dialog")
                logger.log(
                    logging.WARN, f"couldn't find image {r.image_path} in the database"
                )
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
        self.finished.emit()

    @QtCore.Slot()
    def finishVideoAnalysis(self, result: list[VideoDetectionResult]):
        self.results.extend(result)
        self.threadsRunning -= 1
        if self.threadsRunning > 1:
            return

        for r in self.results:
            if not isinstance(r, VideoDetectionResult):
                continue

            frames: dict[int, Image] = dict()
            video = self.session.scalar(
                select(Video).where(Video.path == os.path.normpath(r.video_path.absolute()))
            )
            if not video:
                logger = logging.Logger("analyze_dialog")
                logger.log(
                    logging.WARN, f"couldn't find video {r.video_path} in the database"
                )
                return
            video.analyzed = True
            vidTime = dt.datetime.fromtimestamp(os.path.getmtime(r.video_path))
            for detection in r.boxes:
                frameIdx = detection[0]

                if not frameIdx in frames:
                    framePath = os.path.join(
                        str(r.video_path) + "-frames", str(frameIdx) + ".jpg"
                    )

                    frames[frameIdx] = Image(
                        path=framePath,
                        datetime=vidTime
                        + dt.timedelta(milliseconds=frameIdx * (1000 / video.fps)),
                        analyzed=True,
                    )
                    self.session.add(frames[frameIdx])

                if frames[frameIdx]:
                    entity = Entity()
                    box = detection[1]
                    typeId = detection[2]
                    conf = detection[3]
                    instance = Instance(
                        image=frames[frameIdx],
                        entity=entity,
                        x=box.x,
                        y=box.y,
                        width=box.w,
                        height=box.h,
                        type_id=typeId,
                        confidence=conf,
                    )
                    self.session.add_all((entity, instance))
            self.session.add(video)

        self.session.commit()
        self.finished.emit()

    @staticmethod
    @QtCore.Slot()
    def analyzeVideos(session, refresh: Callable):
        files = list(
            map(
                lambda i: (i.id, i.path),
                session.scalars(select(Video).where(Video.analyzed == False)).all(),
            )
        )

        if len(files) == 0:
            return

        dialog = AnalyzeDialog(
            session,
            files,
            video=True,
        )
        dialog.finished.connect(refresh)
        dialog.exec()
