import os
import datetime
import math

from sqlalchemy.orm import Session
from sqlalchemy import select
from db.models import Image
from PySide6 import QtCore, QtGui, QtWidgets

class GalleryModel(QtCore.QAbstractListModel):
    session: Session
    images: dict[str, Image]

    def __init__(self, session: Session):
        self.session = session
        self.images = dict()
        super().__init__()
    
    def getByIndex(self, index:  QtCore.QModelIndex):
        return self.session.scalar(select(Image).offset(index.row()).limit(1))
    
    def data(self, index: QtCore.QModelIndex, role):
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            data = self.getByIndex(index)
            if data.path in self.images:
                return self.images[data.path]
            else:
                if len(self.images) > 300:
                    self.images = dict()
                img = QtGui.QImage(data.path).scaled(180, 160)
                self.images[data.path] = img
                return img

    def rowCount(self, index):
        return self.session.query(Image).count()

class ImageTab(QtWidgets.QWidget):
    session: Session

    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout(self)
        self.gallery = ImageGallery()
        self.layout.addWidget(self.gallery)
        self.imageInfo = ImageInfo()
        self.layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Image] = list(map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())) # type: ignore
        self.imageInfo.showinfo(selection)
    
    @QtCore.Slot()
    def setsession(self, session: Session):
        self.session = session

        self.galleryModel = GalleryModel(session)
        self.gallery.setModel(self.galleryModel)
        self.gallery.selectionModel().selectionChanged.connect(self.newselection)


class ImageGallery(QtWidgets.QListView):
    def __init__(self):
        super().__init__()
        self.setUniformItemSizes(True)
        self.setIconSize(QtCore.QSize(100, 100))
        self.setViewMode(self.ViewMode.IconMode)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setResizeMode(self.ResizeMode.Adjust)
        self.setSelectionMode(self.SelectionMode.MultiSelection)
        self.setDragEnabled(False)
        self.setLayoutMode(self.LayoutMode.Batched)
        self.setBatchSize(100)

    @QtCore.Slot()
    def invertSelection(self):
        first = self.model().createIndex(0, 0)
        last = self.model().createIndex(self.model().rowCount(0) - 1, 0)
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
        self.imgdate.setText(image.datetime.strftime("%Y-%m-%d %H:%M:%S"))
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
