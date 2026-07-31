#!/usr/bin/env python

from typing import Callable
import os
import platform
import subprocess
import sys

os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

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
        self.setWindowTitle("Park Vision")
        self.setWindowIcon(QtGui.QIcon("./assets/logo.png"))

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

        spinnerLayout = QtWidgets.QHBoxLayout()
        self.spinner = QtWidgets.QLabel("No background tasks.")
        spinnerLayout.addWidget(self.spinner)
        self.progressBar = QtWidgets.QProgressBar(minimum=0, maximum=100)
        self.progressBar.setVisible(False)
        spinnerLayout.addWidget(self.progressBar)
        layout.addLayout(spinnerLayout)
        upl.ThreadTracker().threadAdded.connect(self.spin)
        upl.ThreadTracker().threadProgress.connect(self.spin)
        upl.ThreadTracker().threadRemoved.connect(self.spin)

    @QtCore.Slot(QtCore.QThread)
    def spin(self, thread: QtCore.QThread):
        self.spinner.setText(upl.ThreadTracker().spinText())

        progress = upl.ThreadTracker().spinProgress()
        self.progressBar.setVisible(progress > -1)
        self.progressBar.setValue(progress)

    def buildMenu(self):
        mFile = self.menuBar().addMenu("File")
        aOpen = QtGui.QAction("Open Image Folder", self)
        aOpen.triggered.connect(self.fileOpen)
        mFile.addAction(aOpen)
        aOpenVideos = QtGui.QAction("Open Video Folder", self)
        aOpenVideos.triggered.connect(self.fileOpenVideos)
        mFile.addAction(aOpenVideos)

        smExport = mFile.addMenu("Export")
        aExportFiltered = QtGui.QAction("Filtered", self)
        aExportFiltered.triggered.connect(self.fileExportFiltered)
        smExport.addAction(aExportFiltered)
        aExportAll = QtGui.QAction("All", self)
        aExportAll.triggered.connect(self.fileExportAll)
        smExport.addAction(aExportAll)

        mAnalyze = self.menuBar().addMenu("Analyze")
        smAnalyze = mAnalyze.addMenu("Base Image Analysis")
        aAnalyzeFiltered = QtGui.QAction("Filtered", self)
        aAnalyzeFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyze, True)
        )
        smAnalyze.addAction(aAnalyzeFiltered)
        aAnalyzeAll = QtGui.QAction("All", self)
        aAnalyzeAll.triggered.connect(lambda: self.runAnalysis(self.doAnalyze, False))
        smAnalyze.addAction(aAnalyzeAll)

        smMergeBikes = mAnalyze.addMenu("Merge Bikes and Riders")
        aMergeBikesFiltered = QtGui.QAction("Filtered", self)
        aMergeBikesFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doMergeBikes, True)
        )
        smMergeBikes.addAction(aMergeBikesFiltered)
        aMergeBikesAll = QtGui.QAction("All", self)
        aMergeBikesAll.triggered.connect(
            lambda: self.runAnalysis(self.doMergeBikes, False)
        )
        smMergeBikes.addAction(aMergeBikesAll)

        smClusters = mAnalyze.addMenu("Find Clusters")
        aAnalyzeClustersFiltered = QtGui.QAction("Filtered", self)
        aAnalyzeClustersFiltered.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzeClusters, True)
        )
        smClusters.addAction(aAnalyzeClustersFiltered)
        aAnalyzeClustersAll = QtGui.QAction("All", self)
        aAnalyzeClustersAll.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzeClusters, False)
        )
        smClusters.addAction(aAnalyzeClustersAll)

        aIOUTracking = QtGui.QAction("Frame-By-Frame Tracking", self)
        aIOUTracking.triggered.connect(self.runIouTracking)
        mAnalyze.addAction(aIOUTracking)
        aCalibrateSpeed = QtGui.QAction("Calibrate Speed From Selected Entity", self)
        aCalibrateSpeed.triggered.connect(self.calibrateSpeed)
        mAnalyze.addAction(aCalibrateSpeed)

        smDirection = mAnalyze.addMenu("Direction From Pose")
        aAnalyzePoseDirection = QtGui.QAction("Filtered", self)
        aAnalyzePoseDirection.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzePoseDirection, True)
        )
        smDirection.addAction(aAnalyzePoseDirection)
        aAnalyzeAllPoseDirection = QtGui.QAction("All", self)
        aAnalyzeAllPoseDirection.triggered.connect(
            lambda: self.runAnalysis(self.doAnalyzePoseDirection, False)
        )
        smDirection.addAction(aAnalyzeAllPoseDirection)

        mView = self.menuBar().addMenu("View")
        aZoomIn = QtGui.QAction(
            "Zoom In Image",
            parent=self,
            shortcut=QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomIn),
        )
        aZoomIn.triggered.connect(lambda: self.doZoom(1))
        mView.addAction(aZoomIn)
        aZoomOut = QtGui.QAction(
            "Zoom Out Image",
            parent=self,
            shortcut=QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomOut),
        )
        aZoomOut.triggered.connect(lambda: self.doZoom(-1))
        mView.addAction(aZoomOut)
        aZoomReset = QtGui.QAction("Reset Image Zoom", parent=self)
        mView.addAction(aZoomReset)
        aZoomReset.triggered.connect(lambda: self.doZoom(0))

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

    def doZoom(self, factor: int):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.imageInfo.viewer.doZoom(factor)
        else:
            self.entitiesTab.entityInfo.viewer.doZoom(factor)

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
                        "Run frame-by-frame tracking and select an entity with a known speed to calibrate speed from"
                    )
                    return

                dialog = CalibrateSpeedDialog(self.session, entity)
                dialog.finish.connect(self.tabChanged)
                dialog.exec()
            else:
                self.warnDialog(
                    "Run frame-by-frame tracking and select an entity with a known speed to calibrate speed from"
                )
        else:
            self.warnDialog(
                "Run frame-by-frame tracking and select an entity with a known speed in the entities tab to calibrate speed from"
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
            self.warnDialog("Entity is not within the current entity filters.")
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
                subprocess.call(
                    ("start", options.path),
                    shell=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
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
    # pyi_splash only exists inside the frozen build (it's generated by the
    # Splash() target in main.spec), and only once the app is fully up do we
    # close it - otherwise the user sees a blank window before the splash
    # goes away.
    pyi_splash = None
    if getattr(sys, "frozen", False):
        try:
            import pyi_splash
        except ImportError:
            pyi_splash = None

    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(960, 600)
    widget.show()

    if pyi_splash is not None:
        pyi_splash.close()

    sys.exit(app.exec())
