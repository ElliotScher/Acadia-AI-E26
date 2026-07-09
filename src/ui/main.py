from pathlib import Path
from sqlalchemy import select
import os
import sys

import image_tab as it
from PySide6 import QtCore, QtGui, QtWidgets

from db import get_db
from db.models import Image, Instance
from detection.yolo import load_model
from detection.pose_direction import process_single_image
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


class Root(QtWidgets.QMainWindow):
    db: Engine
    session: Session

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analyzer")

        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        layout = QtWidgets.QVBoxLayout(self.widget)

        self.imageTab = it.ImageTab()
        self.entitiesTab = EntitiesTab()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.imageTab, "Images")
        self.tabs.addTab(self.entitiesTab, "Entities")
        layout.addWidget(self.tabs)

        self.buildMenu()

    def buildMenu(self):
        mFile = self.menuBar().addMenu("File")
        aOpen = QtGui.QAction("Open", self)
        aOpen.triggered.connect(self.fileOpen)
        mFile.addAction(aOpen)
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
        aAnalyzePoseDirection = QtGui.QAction("Analyze Direction From Poses", self)
        aAnalyzePoseDirection.triggered.connect(self.analyzePoseDirection)
        mAnalyze.addAction(aAnalyzePoseDirection)

        mSelect = self.menuBar().addMenu("Select")
        aSelectAll = QtGui.QAction("Select All", self)
        aSelectAll.triggered.connect(self.selectAll)
        aSelectAll.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)
        mSelect.addAction(aSelectAll)
        aSelectDeselect = QtGui.QAction("Deselect All", self)
        aSelectDeselect.triggered.connect(self.imageTab.gallery.clearSelection)
        aSelectDeselect.setShortcut(QtGui.QKeySequence.StandardKey.Deselect)
        mSelect.addAction(aSelectDeselect)
        aSelectInvert = QtGui.QAction("Invert Selection", self)
        aSelectInvert.triggered.connect(self.selectInverse)
        mSelect.addAction(aSelectInvert)

    @QtCore.Slot()
    def fileOpen(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder...")
        self.db = get_db(os.path.join(path, "photos.db"))
        self.session = Session(self.db)
        Image.import_from_dir(self.session, path)
        self.imageTab.setsession(self.session)

    @QtCore.Slot()
    def fileExportFiltered(self):
        isImages = self.tabs.currentWidget() == self.imageTab
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save export...", "images.csv" if isImages else "entities.csv"
        )
        if len(path[0]) > 0 and hasattr(self, "session"):
            if isImages:
                self.imageTab.export(True, path[0])

    @QtCore.Slot()
    def fileExportAll(self):
        isImages = self.tabs.currentWidget() == self.imageTab
        path = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save export...", "images.csv" if isImages else "entities.csv"
        )
        if len(path[0]) > 0 and hasattr(self, "session"):
            if isImages:
                self.imageTab.export(False, path[0])

    @QtCore.Slot()
    def selectAll(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.gallery.selectAll()

    @QtCore.Slot()
    def selectDeselect(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.gallery.clearSelection()

    @QtCore.Slot()
    def selectInverse(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.gallery.invertSelection()

    @QtCore.Slot()
    def analyzeFiltered(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyze(True)

    @QtCore.Slot()
    def analyzeAll(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.analyze(False)

    @QtCore.Slot()
    def analyzePoseDirection(self):
        if hasattr(self, "session"):
            model = load_model("yolo26s-pose.pt")
            instances = self.session.scalars(
                select(Instance).where(Instance.type_id == 0)
            ).all()

            for instance in instances:
                directions = process_single_image(
                    model,
                    Path(instance.image.path).resolve(),
                    Path(),
                    Path(),
                    False,
                    0.25,
                    [0],
                    (
                        instance.x,
                        instance.y,
                        instance.x + instance.width,
                        instance.y + instance.height,
                    ),
                )

                if len(directions) > 0:
                    instance.direction_fb = directions[0].front_back
                    instance.direction_lr = directions[0].left_right
                    self.session.add(instance)

            self.session.commit()


class EntitiesTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.text = QtWidgets.QLabel("Entities tab")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(960, 600)
    widget.show()

    sys.exit(app.exec())
