#!/usr/bin/env python

import os
import platform
import subprocess
import sys
import subprocess
import platform

from image_tab import ImageTab
from entity_tab import EntitiesTab
from export_dialog import ExportDialog, ExportOptions
from cluster_dialog import ClusterDialog
from iou_tracking_dialog import IOUTrackingDialog
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
import utility.parallel as upl
from db import get_db
from db.models import Entity, Image, Video
from ui.analyze_dialog import AnalyzeDialog

from detection.bike_rider_merging import merge_bikes_riders


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
        aAnalyzeFiltered.triggered.connect(self.analyzeFiltered)
        mAnalyze.addAction(aAnalyzeFiltered)
        aAnalyzeAll = QtGui.QAction("Analyze All", self)
        aAnalyzeAll.triggered.connect(self.analyzeAll)
        mAnalyze.addAction(aAnalyzeAll)
        aAnalyzeClustersFiltered = QtGui.QAction("Analyze Filtered Clusters", self)
        aAnalyzeClustersFiltered.triggered.connect(self.analyzeClustersFiltered)
        mAnalyze.addAction(aAnalyzeClustersFiltered)
        aAnalyzeClustersAll = QtGui.QAction("Analyze All Clusters", self)
        aAnalyzeClustersAll.triggered.connect(self.analyzeClustersAll)
        mAnalyze.addAction(aAnalyzeClustersAll)
        aMergeBikesFiltered = QtGui.QAction("Merge Filtered Bikes and Riders", self)
        aMergeBikesFiltered.triggered.connect(self.analyzeMergeBikesFiltered)
        mAnalyze.addAction(aMergeBikesFiltered)
        aMergeBikesAll = QtGui.QAction("Merge All Bikes and Riders", self)
        aMergeBikesAll.triggered.connect(self.analyzeMergeBikesAll)
        mAnalyze.addAction(aMergeBikesAll)
        aIOUTracking = QtGui.QAction("Run IOU Tracking", self)
        aIOUTracking.triggered.connect(self.runIouTracking)
        mAnalyze.addAction(aIOUTracking)

        aAnalyzePoseDirection = QtGui.QAction(
            "Analyze Filtered For Direction From Poses", self
        )
        aAnalyzePoseDirection.triggered.connect(self.analyzePoseDirection)
        mAnalyze.addAction(aAnalyzePoseDirection)
        aAnalyzeAllPoseDirection = QtGui.QAction(
            "Analyze All For Direction From Poses", self
        )
        aAnalyzeAllPoseDirection.triggered.connect(self.analyzeAllPoseDirection)
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
    def analyzeFiltered(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyze(True)

    @QtCore.Slot()
    def analyzeAll(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyze(False)

    @QtCore.Slot()
    def analyzeMergeBikesFiltered(self):
        self.imageTab.mergeBikes(True)

    @QtCore.Slot()
    def analyzeMergeBikesAll(self):
        self.imageTab.mergeBikes(False)

    @QtCore.Slot()
    def analyzeClustersFiltered(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.doAnalyzeClusters(self.imageTab.getImages(True))
        else:
            self.doAnalyzeClusters(self.entitiesTab.getEntities(True))

    @QtCore.Slot()
    def analyzeClustersAll(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.doAnalyzeClusters(self.imageTab.getImages(False))
        else:
            self.doAnalyzeClusters(self.entitiesTab.getEntities(False))

    def tabChanged(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.refreshGallery()
        else:
            self.entitiesTab.refreshGallery()

    def analyzePoseDirection(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyzePoseDirection(True)

    @QtCore.Slot()
    def analyzeAllPoseDirection(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyzePoseDirection(False)

    @QtCore.Slot()
    def runIouTracking(self):
        if hasattr(self, "session"):
            dialog = IOUTrackingDialog(self.session, self.imageTab.getImages(False))
            dialog.exec()

    def warnDialog(self, msg: str):
        d = QtWidgets.QMessageBox()
        d.setWindowTitle("Warning!")
        d.setText(msg)
        d.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        d.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        d.exec()

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
    def doExport(self, options: ExportOptions):
        if not hasattr(self, "session"):
            return

        if options.mode == "images":
            Image.export_to_csv(
                self.session, self.imageTab.getImages(options.filtered), options.path
            )
        elif options.mode == "interval":
            Image.export_to_csv(
                self.session,
                self.imageTab.getImages(options.filtered),
                options.path,
                options.interval,
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
    def doAnalyzeClusters(self, images: list[Image] | list[Entity]):
        if not hasattr(self, "session"):
            return

        if len(images) > 0 and isinstance(images[0], Entity):
            actualImages: list[Image] = []
            entity: Entity
            for entity in images:  # type: ignore
                for instance in entity.get_instances(self.session):
                    if instance.image not in actualImages:
                        actualImages.append(instance.image)
        else:
            actualImages: list[Image] = images  # type: ignore

        dialog = ClusterDialog(self.session, actualImages)
        dialog.accepted.connect(self.tabChanged)
        dialog.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(960, 600)
    widget.show()

    sys.exit(app.exec())
