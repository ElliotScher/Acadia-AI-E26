import os
import datetime
import math

from PySide6 import QtCore, QtGui, QtWidgets


class Image(QtWidgets.QListWidgetItem):
    def __init__(self, parent: QtWidgets.QListWidget, fname: str):
        super().__init__(parent)
        self.name = fname
        self.setIcon(QtGui.QIcon(self.name))

    def getdt(self):
        return datetime.datetime.fromtimestamp(os.stat(self.name).st_mtime)

class ImageTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.gallery = ImageGallery()
        layout.addWidget(self.gallery)
        self.imageInfo = ImageInfo()
        layout.addWidget(self.imageInfo)
        self.gallery.itemSelectionChanged.connect(self.newselection)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Image] = self.gallery.selectedItems() # type: ignore
        self.imageInfo.showinfo(selection)


class ImageGallery(QtWidgets.QListWidget):
    def __init__(self):
        super().__init__()
        self.setUniformItemSizes(True)
        self.iconaspect = 1.5
        self.setIconSize(QtCore.QSize(math.floor(self.iconaspect * 100), 100))
        self.setViewMode(self.ViewMode.IconMode)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setResizeMode(self.ResizeMode.Adjust)
        self.setSelectionMode(self.SelectionMode.MultiSelection)
        self.setDragEnabled(False)
        self.setLayoutMode(self.LayoutMode.Batched)
        self.setBatchSize(100)

    def addImages(self, images: list[str]):
        for image in images:
            reader = QtGui.QImageReader(image)
            if reader.canRead():
                Image(self, image)

    @QtCore.Slot()
    def invertSelection(self):
        first = self.indexFromItem(self.item(0))
        last = self.indexFromItem(self.item(self.count() - 1))
        self.selectionModel().select(
            QtCore.QItemSelection(first, last),
            self.selectionModel().SelectionFlag.Toggle,
        )


class ImageInfo(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__()
        self.setTitle("Image Info")
        self.setMinimumSize(250, 500)
        layout = QtWidgets.QVBoxLayout(self)

        self.placeholder = QtWidgets.QWidget()
        layout.addWidget(self.placeholder)
        layout.setAlignment(self.alignment().AlignTop)
        
        self.info = self.buildinfo()
        layout.addWidget(self.info)
        self.info.hide()

    def showinfo(self, images: list[Image]):
        if len(images) == 1:
            self.showone(images[0])
        elif len(images) > 1:
            self.showmultiple(images)
        else:
            self.info.hide()
            self.placeholder.show()

    def showone(self, image: Image):
        self.imgcount.setText("There can only be one!\n")
        self.imgdate.setText(image.getdt().strftime("%Y-%m-%d %H:%M:%S"))
        self.info.show()
        self.placeholder.hide()

    def showmultiple(self, images: list[Image]):
        self.imgcount.setText(f"{len(images)} selected.")
        self.info.show()
        self.placeholder.hide()

    def buildinfo(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.imgcount = QtWidgets.QLabel("THE IMAGE INFO BOX :)")
        layout.addWidget(self.imgcount)
        self.imgdate = QtWidgets.QLabel("A long time ago...")
        layout.addWidget(self.imgdate)
        return widget
