import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select, Select
from sqlalchemy.orm import Session
from datetime import datetime

from db.models import Image, Instance


class GalleryModel(QtCore.QAbstractListModel):
    session: Session
    images: dict[str, QtGui.QIcon]
    size: int = 0
    filters: 'Filters'

    def __init__(self, session: Session):
        self.session = session
        self.images = dict()
        super().__init__()

    def getByIndex(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> Image:
        return self.session.scalar(self.filters.makeFilter(select(Image)).order_by(Image.datetime).offset(index.row()).limit(1))  # type: ignore

    def data(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0
    ) -> typing.Any:
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            data = self.getByIndex(index)
            if data:
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
        parent: (
            QtCore.QModelIndex | QtCore.QPersistentModelIndex
        ) = QtCore.QModelIndex(),
    ) -> int:
        return self.size

    def fetchMore(
        self,
        parent: (
            QtCore.QModelIndex | QtCore.QPersistentModelIndex
        ) = QtCore.QModelIndex(),
    ):
        newmax = min(self.size + 300, self.filters.makeFilter(self.session.query(Image)).count())
        self.beginInsertRows(QtCore.QModelIndex(), self.size, newmax - 1)
        self.size = newmax
        self.endInsertRows()

    def canFetchMore(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex, /
    ) -> bool:
        return self.size < self.filters.makeFilter(self.session.query(Image)).count()


class ImageTab(QtWidgets.QWidget):
    session: Session

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        gallerySide = QtWidgets.QWidget()
        gallerySideLayout = QtWidgets.QVBoxLayout(gallerySide)

        self.filters = Filters()
        gallerySideLayout.addWidget(self.filters)
        self.gallery = ImageGallery()
        gallerySideLayout.addWidget(self.gallery)

        layout.addWidget(gallerySide)
        self.imageInfo = ImageInfo()
        layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Image] = list(
            map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())
        )
        self.imageInfo.showinfo(selection, self.session)

    @QtCore.Slot()
    def setsession(self, session: Session):
        self.session = session

        self.refreshGallery()

        minDt = datetime_to_qdatetime(Image.get_earliest_image(session).datetime)
        maxDt = datetime_to_qdatetime(Image.get_latest_image(session).datetime)
        self.filters.startDate.setDateTimeRange(minDt, maxDt)
        self.filters.endDate.setDateTimeRange(minDt, maxDt)
        self.filters.startDate.setDateTime(minDt)
        self.filters.endDate.setDateTime(maxDt)

    @QtCore.Slot()
    def refreshGallery(self):
        if not hasattr(self, "session"):
            return
        self.galleryModel = GalleryModel(self.session)
        self.galleryModel.filters = self.filters
        self.gallery.setModel(self.galleryModel)
        self.gallery.selectionModel().selectionChanged.connect(self.newselection)


class Filters(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        filterLayout = QtWidgets.QHBoxLayout(self)

        self.startDate = QtWidgets.QDateTimeEdit()
        self.startDate.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.startDate.setCalendarPopup(True)
        self.startDate.dateTimeChanged.connect(self.refreshGallery)
        filterLayout.addWidget(self.startDate)

        filterLayout.addWidget(QtWidgets.QLabel("to"))

        self.endDate = QtWidgets.QDateTimeEdit()
        self.endDate.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        self.endDate.setCalendarPopup(True)
        self.endDate.dateTimeChanged.connect(self.refreshGallery)
        filterLayout.addWidget(self.endDate)

        self.analyzedFilter = QtWidgets.QComboBox()
        self.analyzedFilter.addItems(["All", "Only Analyzed", "Only Unanalyzed"])
        self.analyzedFilter.currentIndexChanged.connect(self.refreshGallery)
        filterLayout.addWidget(self.analyzedFilter)
    
    def makeFilter(self, select: Select) -> Select:
        filters = select.where(Image.datetime >= self.startDate.dateTime().toPython()).where(Image.datetime <= self.endDate.dateTime().toPython())
        if self.analyzedFilter.currentIndex() == 1:
            filters = filters.where(Image.analyzed == True)
        elif self.analyzedFilter.currentIndex() == 2:
            filters = filters.where(Image.analyzed == False)
        return filters

    @QtCore.Slot()
    def refreshGallery (self):
        self.parentWidget().parentWidget().refreshGallery()

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
        self.setMinimumSize(400, 500)
        self.setMaximumWidth(400)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(self.alignment().AlignTop)

        self.viewer = ImageViewer()
        self.viewer.resize(300, 150)
        self.viewer.hide()
        layout.addWidget(self.viewer)

        self.placeholder = QtWidgets.QWidget()
        layout.addWidget(self.placeholder)

        self.info = self.buildinfo()
        layout.addWidget(self.info)
        self.info.hide()

    def showinfo(self, images: list[Image], session: Session):
        if len(images) == 1:
            self.showone(images[0], session)
            self.viewer.show()
        elif len(images) > 1:
            self.showmultiple(images)
            self.viewer.hide()
        else:
            self.info.hide()
            self.viewer.hide()
            self.placeholder.show()

    def showone(self, image: Image, session: Session):
        self.viewer.set(image, session)
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


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.scene)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set(self, image: Image, session: Session):
        self.scene.clear()

        pixmap = QtGui.QPixmap(image.path)
        self.resize(pixmap.width(), pixmap.height())
        self.pixmapItem = self.scene.addPixmap(QtGui.QPixmap())
        self.pixmapItem.setPixmap(pixmap)
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        for instance in image.get_instances(session):
            self.scene.addRect(
                instance.x,
                instance.y,
                instance.width,
                instance.height,
                self.getpen("green"),
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmapItem:
            self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def getpen(color: str) -> QtGui.QPen:
        pen = QtGui.QPen(color)
        pen.setWidth(10)
        return pen


def datetime_to_qdatetime(dt: datetime):
    return QtCore.QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, 0)
