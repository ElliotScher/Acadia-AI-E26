import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Image


class GalleryModel(QtCore.QAbstractListModel):
    session: Session
    images: dict[str, QtGui.QIcon]
    size: int = 0

    def __init__(self, session: Session):
        self.session = session
        self.images = dict()
        super().__init__()

    def getByIndex(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> Image:
        return self.session.scalar(select(Image).offset(index.row()).limit(1))  # type: ignore

    def data(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0
    ) -> typing.Any:
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            data = self.getByIndex(index)
            if data.path in self.images:
                return self.images[data.path]
            else:
                if len(self.images) > 300:
                    self.images = dict()
                img = QtGui.QIcon(data.path)
                self.images[data.path] = img
                return img

    def rowCount(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        # return self.session.query(Image).count()
        return self.size

    def fetchMore(
        self,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ):
        newmax = max(self.size + 300, self.session.query(Image).count())
        self.beginInsertRows(QtCore.QModelIndex(), self.size, newmax - 1)
        self.size = newmax
        self.endInsertRows()

    def canFetchMore(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex, /
    ) -> bool:
        return self.size < self.session.query(Image).count()


class ImageTab(QtWidgets.QWidget):
    session: Session

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.gallery = ImageGallery()
        layout.addWidget(self.gallery)
        self.imageInfo = ImageInfo()
        layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Image] = list(
            map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())
        )
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
        last = self.model().createIndex(self.model().rowCount() - 1, 0)
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
