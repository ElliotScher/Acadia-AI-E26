#!/usr/bin/env python

from typing import Callable
import os
import platform
import subprocess
import sys

from cluster_dialog import ClusterDialog
from entity_tab import EntitiesTab
from export_dialog import ExportDialog, ExportOptions
from image_tab import ImageTab
from iou_tracking_dialog import IOUTrackingDialog
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import utility.parallel as upl
from db import get_db
from db.models import Entity, Image, Video
from ui.analyze_dialog import AnalyzeDialog
from ui.calibrate_speed_dialog import CalibrateSpeedDialog
from ui.pose_direction_dialog import PoseDirectionDialog
from ui.bike_rider_merging_dialog import BikeRiderMergeDialog


class Root(QtWidgets.QMainWindow):
    db: Engine
    session: Session

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analyzer")

        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        layout = QtWidgets.QVBoxLayout(self.widget)

        self.imageTab = ImageTab()
        self.entitiesTab = EntitiesTab()
        self.imageTab.entityOpened.connect(self.openEntity)
        self.entitiesTab.imageOpened.connect(self.openImage)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.imageTab, "Images")
        self.tabs.addTab(self.entitiesTab, "Entities")
        layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.tabChanged)

        self.buildMenu()

        self.spinner = QtWidgets.QLabel()
        layout.addWidget(self.spinner)
        upl.ThreadTracker().threadAdded.connect(self.spin)
        upl.ThreadTracker().threadProgress.connect(self.spin)
        upl.ThreadTracker().threadRemoved.connect(self.spin)

    @QtCore.Slot(QtCore.QThread)
    def spin(self, thread: QtCore.QThread):
        self.spinner.setText(upl.ThreadTracker().spinText())

    def buildMenu(self):
        mFile = self.menuBar().addMenu("File")
        aOpen = QtGui.QAction("Open Images", self)
        aOpen.triggered.connect(self.fileOpen)
        mFile.addAction(aOpen)
        aOpenVideos = QtGui.QAction("Open Videos", self)
        aOpenVideos.triggered.connect(self.fileOpenVideos)
        mFile.addAction(aOpenVideos)
        aExportFiltered = QtGui.QAction("Export Filtered", self)
        aExportFiltered.triggered.connect(self.fileExportFiltered)
        mFile.addAction(aExportFiltered)
        aExportAll = QtGui.QAction("Export All", self)
        aExportAll.triggered.connect(self.fileExportAll)
        mFile.addAction(aExportAll)

        mAnalyze = self.menuBar().addMenu("Analyze")
        aAnalyzeFiltered = QtGui.QAction("Analyze Filtered", self)
        aAnalyzeFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyze, True)
        )
        mAnalyze.addAction(aAnalyzeFiltered)
        aAnalyzeAll = QtGui.QAction("Analyze All", self)
        aAnalyzeAll.triggered.connect(lambda: self.runAnalysis(self.doAnalyze, False))
        mAnalyze.addAction(aAnalyzeAll)

        aMergeBikesFiltered = QtGui.QAction("Merge Filtered Bikes and Riders", self)
        aMergeBikesFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doMergeBikes, True)
        )
        mAnalyze.addAction(aMergeBikesFiltered)
        aMergeBikesAll = QtGui.QAction("Merge All Bikes and Riders", self)
        aMergeBikesAll.triggered.connect(
            lambda: self.runAnalysis(self.doMergeBikes, False)
        )
        mAnalyze.addAction(aMergeBikesAll)

        aAnalyzeClustersFiltered = QtGui.QAction("Analyze Filtered Clusters", self)
        aAnalyzeClustersFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzeClusters, True)
        )
        mAnalyze.addAction(aAnalyzeClustersFiltered)
        aAnalyzeClustersAll = QtGui.QAction("Analyze All Clusters", self)
        aAnalyzeClustersAll.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzeClusters, False)
        )
        mAnalyze.addAction(aAnalyzeClustersAll)

        aIOUTracking = QtGui.QAction("Run IOU Tracking", self)
        aIOUTracking.triggered.connect(self.runIouTracking)
        mAnalyze.addAction(aIOUTracking)
        aCalibrateSpeed = QtGui.QAction("Calibrate Speed From Selected Entity", self)
        aCalibrateSpeed.triggered.connect(self.calibrateSpeed)
        mAnalyze.addAction(aCalibrateSpeed)

        aAnalyzePoseDirection = QtGui.QAction(
            "Analyze Filtered For Direction From Poses", self
        )
        aAnalyzePoseDirection.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzePoseDirection, True)
        )
        mAnalyze.addAction(aAnalyzePoseDirection)
        aAnalyzeAllPoseDirection = QtGui.QAction(
            "Analyze All For Direction From Poses", self
        )
        aAnalyzeAllPoseDirection.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzePoseDirection, False)
        )
        mAnalyze.addAction(aAnalyzeAllPoseDirection)

    def _fileOpen(self, path: str):
        self.db = get_db(os.path.join(path, "photos.db"))
        self.session = Session(self.db)
        Image.import_from_dir(self.session, path)

    @QtCore.Slot()
    def fileOpen(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder...")
        thread = upl.Async("File Open", lambda _: self._fileOpen(path))
        thread.finished.connect(lambda: self.imageTab.setsession(self.session))
        thread.finished.connect(lambda: self.entitiesTab.setsession(self.session))
        thread.start()

    def _fileOpenVideo(self, path: str):
        self.db = get_db(os.path.join(path, "videos.db"))
        self.session = Session(self.db)
        Video.import_from_dir(self.session, path)

    @QtCore.Slot()
    def fileOpenVideos(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder...")
        thread = upl.Async("File Open", lambda _: self._fileOpenVideo(path))
        thread.finished.connect(lambda: self.imageTab.setsession(self.session))
        thread.finished.connect(lambda: self.entitiesTab.setsession(self.session))
        thread.finished.connect(
            lambda: AnalyzeDialog.analyzeVideos(self.session, self.tabChanged)
        )
        thread.start()

    def tabChanged(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.refreshGallery()
        else:
            self.entitiesTab.refreshGallery()

    @QtCore.Slot()
    def fileExportFiltered(self):
        dialog = ExportDialog(True)
        dialog.startExport.connect(self.doExport)
        dialog.exec()

    @QtCore.Slot()
    def fileExportAll(self):
        dialog = ExportDialog(False)
        dialog.startExport.connect(self.doExport)
        dialog.exec()

    @QtCore.Slot()
    def runAnalysis(self, analysis: Callable[[list[Image]], None], filtered: bool):
        if not hasattr(self, "session"):
            self.warnDialog("Import images or video before running analysis")
            return

        if self.tabs.currentWidget() == self.imageTab:
            analysis(self.imageTab.getImages(filtered))
        else:
            entities = self.entitiesTab.getEntities(filtered)
            images: list[Image] = []
            entity: Entity
            for entity in entities:
                for instance in entity.get_instances(self.session):
                    if instance.image not in images:
                        images.append(instance.image)
            analysis(images)

    @QtCore.Slot()
    def runIouTracking(self):
        if hasattr(self, "session"):
            dialog = IOUTrackingDialog(self.session, self.imageTab.getImages(False))
            dialog.finish.connect(self.tabChanged)
            dialog.exec()

    def warnDialog(self, msg: str):
        d = QtWidgets.QMessageBox()
        d.setWindowTitle("Warning!")
        d.setText(msg)
        d.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        d.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        d.exec()

    @QtCore.Slot()
    def calibrateSpeed(self):
        if self.tabs.currentWidget() == self.entitiesTab:
            if len(self.entitiesTab.gallery.selectionModel().selectedIndexes()) == 1:
                entity = self.entitiesTab.galleryModel.getByIndex(
                    self.entitiesTab.gallery.selectionModel().selectedIndexes()[0]
                )

                if entity is None or entity.rawSpeed is None:
                    self.warnDialog(
                        "Select an entity with a known speed to calibrate speed from"
                    )
                    return

                dialog = CalibrateSpeedDialog(self.session, entity)
                dialog.finish.connect(self.tabChanged)
                dialog.exec()
            else:
                self.warnDialog(
                    "Select an entity with a known speed to calibrate speed from"
                )
        else:
            self.warnDialog(
                "Select an entity with a known speed in the entities tab to calibrate speed from"
            )

    @QtCore.Slot()
    def openImage(self, image: Image):
        r = self.imageTab.focusImage(image)
        if not r:
            self.warnDialog("Image is not within the current image filters.")
            return
        if self.tabs.currentWidget() == self.entitiesTab:
            self.tabs.setCurrentWidget(self.imageTab)

    @QtCore.Slot()
    def openEntity(self, entity: Entity):
        r = self.entitiesTab.focusEntity(entity)
        if not r:
            self.warnDialog("Image is not within the current entity filters.")
            return
        if self.tabs.currentWidget() == self.imageTab:
            self.tabs.setCurrentWidget(self.entitiesTab)

    @QtCore.Slot()
    def doAnalyze(self, images: list[Image]):
        dialog = AnalyzeDialog(
            self.session, list(map(lambda i: (i.id, i.path), images))
        )
        dialog.finish.connect(self.tabChanged)
        dialog.exec()

    @QtCore.Slot()
    def doExport(self, options: ExportOptions):
        if not hasattr(self, "session"):
            return

        if options.mode == "images":
            Image.export_to_csv(
                self.session,
                self.imageTab.getImages(options.filtered),
                options.path,
                separateDirections=options.separateDirections,
            )
        elif options.mode == "interval":
            Image.export_to_csv(
                self.session,
                self.imageTab.getImages(options.filtered),
                options.path,
                options.interval,
                options.separateDirections,
            )
        elif options.mode == "clusters":
            Entity.export_clusters_to_csv(
                self.session,
                self.entitiesTab.getEntities(options.filtered),
                options.path,
            )
        else:
            Entity.export_to_csv(
                self.session,
                self.entitiesTab.getEntities(options.filtered),
                options.path,
            )

        if options.open:
            if platform.system() == "Darwin":
                subprocess.call(("open", options.path))
            elif platform.system() == "Windows":
                subprocess.call(("start", options.path), shell=True)
            else:
                subprocess.call(("xdg-open", options.path))

    @QtCore.Slot()
    def doAnalyzeClusters(self, images: list[Image]):
        dialog = ClusterDialog(self.session, images)
        dialog.finish.connect(self.tabChanged)
        dialog.exec()

    @QtCore.Slot()
    def doAnalyzePoseDirection(self, images: list[Image]):
        dialog = PoseDirectionDialog(self.session, images)
        dialog.finish.connect(self.tabChanged)
        dialog.exec()

    @QtCore.Slot()
    def doMergeBikes(self, images: list[Image]):
        dialog = BikeRiderMergeDialog(self.session, images)
        dialog.finish.connect(self.tabChanged)
        dialog.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(960, 600)
    widget.show()

    sys.exit(app.exec())
