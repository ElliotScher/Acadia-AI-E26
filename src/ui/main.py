import os
import sys

import image_tab as it
from PySide6 import QtCore, QtGui, QtWidgets

from db import get_db
from db.models import Image
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

class Root(QtWidgets.QMainWindow):
    db: Engine

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
        aExport = QtGui.QAction("Export", self)
        mFile.addAction(aExport)

        mAnalyze = self.menuBar().addMenu("Analyze")
        aAnalyzeSelection = QtGui.QAction("Analyze Selection", self)
        mAnalyze.addAction(aAnalyzeSelection)

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

        with Session(self.db) as session:
            Image.import_from_dir(session, path)
        # self.imageTab.gallery.addImages(images)

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


# class SetupTab(QtWidgets.QWidget):
#     folder = None
#
#     def __init__(self):
#         super().__init__()
#
#         self.base_group = QtWidgets.QGroupBox("Base Model")
#         self.base_group_layout = QtWidgets.QVBoxLayout()
#         self.base_group.setLayout(self.base_group_layout)
#
#         self.base_select_folder = QtWidgets.QPushButton("Select Folder")
#         self.base_group_layout.addWidget(self.base_select_folder)
#         self.base_select_folder.clicked.connect(self.show_select_folder)
#         self.base_select_folder.setDefault(True)
#         self.base_folder_selection = QtWidgets.QLabel("No folder selected")
#         self.base_group_layout.addWidget(self.base_folder_selection)
#
#         self.types_group = QtWidgets.QGroupBox("Target Types")
#         self.types_group_layout = QtWidgets.QHBoxLayout()
#         self.types_group.setLayout(self.types_group_layout)
#         self.types_group.setDisabled(True)
#         self.types_scroll = QtWidgets.QScrollArea()
#         self.types_scroll.setWidgetResizable(True)
#         self.types_scroll.setMinimumHeight(100)
#         self.types_scroll_contents = QtWidgets.QWidget()
#         self.types_scroll.setWidget(self.types_scroll_contents)
#         self.types_group_layout.addWidget(self.types_scroll)
#         self.types_scroll_layout = QtWidgets.QGridLayout()
#         self.types_scroll_contents.setLayout(self.types_scroll_layout)
#
#         types = (
#             "People",
#             "Bikes",
#             "Cars",
#             "Trucks",
#             "Buses",
#             "Boats",
#             "Motorcycles",
#             "Dogs",
#             "Horses",
#             "Birds",
#             "Bears",
#             "Backpacks",
#             "Handbags",
#             "Suitcases",
#             "Umbrellas",
#             "Skateboards",
#             "Snowboards",
#             "Skis",
#         )
#         self.types_options = []
#         for i in range(len(types)):
#             type_checkbox = QtWidgets.QCheckBox(types[i])
#             self.types_scroll_layout.addWidget(type_checkbox, math.floor(i / 3), i % 3)
#             self.types_options.append(type_checkbox)
#
#         self.base_group_layout.addWidget(self.types_group)
#
#         self.base_denoising = QtWidgets.QCheckBox("Denoise images")
#         self.base_group_layout.addWidget(self.base_denoising)
#         self.base_denoising.setDisabled(True)
#         self.base_run_model = QtWidgets.QPushButton("Run Base Model")
#         self.base_group_layout.addWidget(self.base_run_model)
#         self.base_run_model.clicked.connect(self.run_base_model)
#         self.base_run_model.setDisabled(True)
#
#         self.matching_group = QtWidgets.QGroupBox("Matching Models")
#         self.matching_group.setDisabled(True)
#         self.matching_group_layout = QtWidgets.QVBoxLayout()
#         self.matching_group.setLayout(self.matching_group_layout)
#
#         self.matching_reidentification = QtWidgets.QCheckBox("Re-identification")
#         self.matching_group_layout.addWidget(self.matching_reidentification)
#         self.matching_tracking = QtWidgets.QCheckBox("Tracking")
#         self.matching_group_layout.addWidget(self.matching_tracking)
#         self.run_matching_models = QtWidgets.QPushButton("Run Matching Models")
#         self.matching_group_layout.addWidget(self.run_matching_models)
#
#         self.additional_group = QtWidgets.QGroupBox("Additional Models")
#         self.additional_group.setDisabled(True)
#         self.additional_group_layout = QtWidgets.QVBoxLayout()
#         self.additional_group.setLayout(self.additional_group_layout)
#
#         self.additional_speed = QtWidgets.QCheckBox("Speed")
#         self.additional_group_layout.addWidget(self.additional_speed)
#         self.additional_direction = QtWidgets.QCheckBox("Direction")
#         self.additional_group_layout.addWidget(self.additional_direction)
#         self.additional_bike_type = QtWidgets.QCheckBox("Bike Type")
#         self.additional_group_layout.addWidget(self.additional_bike_type)
#         self.additional_clusters = QtWidgets.QCheckBox("Clusters")
#         self.additional_group_layout.addWidget(self.additional_clusters)
#         self.run_additional_models = QtWidgets.QPushButton("Run Additional Models")
#         self.additional_group_layout.addWidget(self.run_additional_models)
#
#         self.layout = QtWidgets.QVBoxLayout(self)
#         self.layout.addWidget(self.base_group)
#         self.layout.addWidget(self.matching_group)
#         self.layout.addWidget(self.additional_group)
#
#     @QtCore.Slot()
#     def show_select_folder(self):
#         old_folder = self.folder
#         self.folder = QtWidgets.QFileDialog.getExistingDirectory(
#             options=QtWidgets.QFileDialog.Option.ShowDirsOnly
#         )
#         if self.folder:
#             self.base_folder_selection.setText("Folder: " + self.folder.split("/")[-2])
#             self.base_select_folder.setDefault(False)
#             self.types_group.setDisabled(False)
#             self.base_denoising.setDisabled(False)
#             self.base_run_model.setDisabled(False)
#             self.base_run_model.setDefault(True)
#             self.parentWidget().parentWidget().parentWidget().tabs.setTabEnabled(
#                 1, True
#             )
#             if old_folder != self.folder:
#                 self.parentWidget().parentWidget().parentWidget().tabs.setTabEnabled(
#                     2, False
#                 )
#                 self.matching_group.setDisabled(True)
#                 self.additional_group.setDisabled(True)
#         else:
#             self.base_folder_selection.setText("No folder selected")
#             self.base_select_folder.setDefault(True)
#             self.types_group.setDisabled(True)
#             self.base_denoising.setDisabled(True)
#             self.base_run_model.setDisabled(True)
#             self.base_run_model.setDefault(False)
#             self.parentWidget().parentWidget().parentWidget().tabs.setTabEnabled(
#                 1, False
#             )
#             self.parentWidget().parentWidget().parentWidget().tabs.setTabEnabled(
#                 2, False
#             )
#             self.matching_group.setDisabled(True)
#             self.additional_group.setDisabled(True)
#
#     @QtCore.Slot()
#     def run_base_model(self):
#         self.matching_group.setDisabled(False)
#         self.additional_group.setDisabled(False)
#         self.base_run_model.setDefault(False)
#         self.parentWidget().parentWidget().parentWidget().tabs.setTabEnabled(2, True)


class EntitiesTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.text = QtWidgets.QLabel("Entities tab")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.text)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = Root()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())

