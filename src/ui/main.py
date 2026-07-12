#!/usr/bin/env python

import os
import sys

import image_tab as it
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from detection.bike_rider_merging import merge_bikes_riders
import utility.parallel as upl
from db import get_db
from db.models import Image


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
        aMergeBikesFiltered = QtGui.QAction("Merge Filtered Bikes and Riders", self)
        aMergeBikesFiltered.triggered.connect(self.analyzeMergeBikesFiltered)
        mAnalyze.addAction(aMergeBikesFiltered)
        aMergeBikesAll = QtGui.QAction("Merge All Bikes and Riders", self)
        aMergeBikesAll.triggered.connect(self.analyzeMergeBikesAll)
        mAnalyze.addAction(aMergeBikesAll)

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

    def _fileOpen(self, path: str):
        self.db = get_db(os.path.join(path, "photos.db"))
        self.session = Session(self.db)
        Image.import_from_dir(self.session, path)
        # self.imageTab.setsession(self.session)

    @QtCore.Slot()
    def fileOpen(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder...")
        thread = upl.Async("File Open", lambda: self._fileOpen(path))
        thread.finished.connect(lambda: self.imageTab.setsession(self.session))
        thread.start()

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
    def analyzeMergeBikesFiltered(self):
        self.imageTab.mergeBikes(True)

    @QtCore.Slot()
    def analyzeMergeBikesAll(self):
        self.imageTab.mergeBikes(False)


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
