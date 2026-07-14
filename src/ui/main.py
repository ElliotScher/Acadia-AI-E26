#!/usr/bin/env python

import os
import sys
import subprocess
import platform

from image_tab import ImageTab
from entity_tab import EntitiesTab
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import utility.parallel as upl
from db import get_db
from db.models import Image, Entity
from export_dialog import ExportDialog, ExportOptions


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

    @QtCore.Slot()
    def fileOpen(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select a folder...")
        thread = upl.Async("File Open", lambda: self._fileOpen(path))
        thread.finished.connect(lambda: self.imageTab.setsession(self.session))
        thread.finished.connect(lambda: self.entitiesTab.setsession(self.session))
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
    def tabChanged(self):
        if self.tabs.currentWidget() == self.imageTab:
            self.imageTab.refreshGallery()
        else:
            self.entitiesTab.refreshGallery()

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
            self.warnDialog("Image is not within the current image filters.")
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
        else:
            Entity.export_to_csv(
                self.session,
                self.entitiesTab.getEntities(options.filtered),
                options.path,
            )

        if open:
            if platform.system() == "Darwin":
                subprocess.call(("open", options.path))
            elif platform.system() == "Windows":
                subprocess.call(("start", options.path), shell=True)
            else:
                subprocess.call(("xdg-open", options.path))


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(960, 600)
    widget.show()

    sys.exit(app.exec())
