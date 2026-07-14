import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import Session

from detection.yolo import CLASS_ID_MAPPING
from db.models import Image, Instance, Entity
from filters import Filters
from filters.image import (
    EntityFilter,
    NoEntityFilter,
    ImageTimeFilter,
    ImageDateFilter,
    AnalyzedFilter,
    NotAnalyzedFilter,
)
from analyze_dialog import AnalyzeDialog

colors = (
    "#00ff00",
    "#ff00ff",
    "#00ffff",
    "#ffff00",
    "#ff0000",
    "#0000ff",
    "#ff8800",
    "#8800ff",
    "#0088ff",
    "#008800",
    "#888888",
    "#ff0088",
    "#ff8888",
    "#88ff88",
    "#8888ff",
    "#ffff88",
    "#ff88ff",
    "#88ffff",
)


class GalleryModel(QtCore.QAbstractListModel):
    session: Session
    size: int = 0
    filters: "Filters"

    def __init__(self, session: Session):
        self.session = session
        self.thumbnails: dict[int, QtGui.QIcon] = dict()
        self.results: list[int] = []
        super().__init__()

    def getByIndex(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> Image:
        return self.getById(self.results[index.row()])

    def getById(self, id: int) -> Image:
        return self.session.scalar(select(Image).where(Image.id == id))  # type: ignore

    def data(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0
    ) -> typing.Any:
        if role == QtCore.Qt.ItemDataRole.DecorationRole:
            data = self.getByIndex(index)
            if data:
                if data.id in self.thumbnails:
                    return self.thumbnails[data.id]
                else:
                    if len(self.thumbnails) > 300:
                        self.thumbnails: dict[int, QtGui.QIcon] = dict()
                    img = QtGui.QIcon(data.path)
                    self.thumbnails[data.id] = img
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
        tofetch = min(300, len(self.results) - self.size)
        if tofetch < 1:
            return
        self.beginInsertRows(QtCore.QModelIndex(), self.size, self.size + tofetch - 1)
        self.size += tofetch
        self.endInsertRows()

    def canFetchMore(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex, /
    ) -> bool:
        return self.size < len(self.results)


class ImageTab(QtWidgets.QWidget):
    session: Session
    entityOpened = QtCore.Signal(Entity)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        gallerySide = QtWidgets.QWidget()
        gallerySideLayout = QtWidgets.QVBoxLayout(gallerySide)

        self.filters = Filters(
            (
                EntityFilter,
                NoEntityFilter,
                ImageTimeFilter,
                ImageDateFilter,
                AnalyzedFilter,
                NotAnalyzedFilter,
            )
        )
        gallerySideLayout.addWidget(self.filters)
        self.filters.changed.connect(self.refreshGallery)
        self.count = QtWidgets.QLabel("0 images")
        gallerySideLayout.addWidget(self.count)
        self.gallery = ImageGallery()
        gallerySideLayout.addWidget(self.gallery)

        layout.addWidget(gallerySide)
        self.imageInfo = ImageInfo()
        self.imageInfo.entityOpened.connect(self.entityOpened.emit)
        layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Image] = list(
            map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())
        )
        self.imageInfo.showImage(
            selection[0] if len(selection) > 0 else None, self.session
        )

    @QtCore.Slot()
    def setsession(self, session: Session):
        self.session = session

        self.refreshGallery()

    @QtCore.Slot()
    def refreshGallery(self):
        if not hasattr(self, "session"):
            return
        self.galleryModel = GalleryModel(self.session)
        self.galleryModel.filters = self.filters
        self.gallery.setModel(self.galleryModel)
        self.gallery.selectionModel().selectionChanged.connect(self.newselection)

        subquery = self.filters.makeFilter(
            Image.id, Image.path, Image.datetime
        ).subquery()
        query = (
            select(subquery)
            .select_from(subquery)
            .order_by(subquery.c.datetime)
            .distinct()
        )
        self.galleryModel.results = list(
            map(lambda d: d[0], self.session.execute(query).unique().all())
        )
        self.count.setText(str(len(self.galleryModel.results)) + " images")

    def getImages(self, filtered: bool) -> list[Image]:
        if not hasattr(self, "session"):
            return []

        if filtered:
            return list(map(self.galleryModel.getById, self.galleryModel.results))
        else:
            return list(
                self.session.scalars(select(Image).order_by(Image.datetime).distinct())
            )

    @QtCore.Slot()
    def analyze(self, filtered: bool):
        if not hasattr(self, "session"):
            return

        images = self.getImages(filtered)

        dialog = AnalyzeDialog(self.session, images)
        dialog.accepted.connect(self.refreshGallery)
        dialog.exec()

    @QtCore.Slot(Image, result=bool)
    def focusImage(self, image: Image) -> bool:
        if image.id not in self.galleryModel.results:
            return False
        i = self.galleryModel.results.index(image.id)
        # this is scary, and should ideally be removed
        while i >= self.galleryModel.rowCount():
            if not self.galleryModel.canFetchMore(QtCore.QModelIndex()):
                return False
            self.galleryModel.fetchMore(QtCore.QModelIndex())
        index = self.galleryModel.index(i, 0)
        self.gallery.scrollTo(index)
        self.gallery.selectionModel().select(
            QtCore.QItemSelection(index, index),
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
        )
        return True


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
    entityOpened = QtCore.Signal(Entity)

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

    def showImage(self, image: Image | None, session: Session):
        if image:
            self.showInfo(image, session)
        else:
            self.info.hide()
            self.viewer.hide()
            self.placeholder.show()

    def showInfo(self, image: Image, session: Session):
        instances = image.get_instances(session)
        self.entities.clear()

        def makeEntityButton(
            entity: Entity,
        ):  # this has to be a function to capture the entity in the lambda
            button = QtWidgets.QPushButton()
            button.setIcon(QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.ViewFullscreen))
            button.setIconSize(QtCore.QSize(15, 15))
            button.setFixedSize(QtCore.QSize(25, 25))
            button.clicked.connect(lambda: self.entityOpened.emit(entity))
            return button

        for i in range(len(instances)):
            instance = instances[i]
            widget = QtWidgets.QWidget(self.entities)
            widget.setFixedHeight(40)
            layout = QtWidgets.QHBoxLayout(widget)
            tlabel = QtWidgets.QLabel(
                f"<font color={colors[i % len(colors)]}>{CLASS_ID_MAPPING[instance.type_id]}</font>",
                widget,
            )
            layout.addWidget(tlabel)
            c = instance.confidence
            color = (
                f"#{int(30 * c + 255 * (1 - c)):02x}{int(255 * c + 30 * (1 - c)):02x}1e"
            )
            clabel = QtWidgets.QLabel(f"<font color={color}>{c:0.2%}</font> ", widget)
            layout.addWidget(clabel)
            button = makeEntityButton(instance.entity)
            layout.addWidget(button)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 40))
            self.entities.addItem(item)
            self.entities.setItemWidget(item, widget)
        self.viewer.set(image, instances)
        self.imgdate.setText(image.datetime.strftime("%Y-%m-%d %H:%M:%S"))
        self.info.show()
        self.viewer.show()
        self.placeholder.hide()

    def buildinfo(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(widget)
        self.imgdate = QtWidgets.QLabel("A long time ago...", widget)
        layout.addWidget(self.imgdate)
        self.entities = QtWidgets.QListWidget(widget)
        layout.addWidget(self.entities)
        self.entities.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        return widget


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thisScene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.thisScene)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set(self, image: Image, instances: list[Instance]):
        self.thisScene.clear()

        pixmap = QtGui.QPixmap(image.path)
        self.resize(pixmap.width(), pixmap.height())
        self.pixmapItem = self.thisScene.addPixmap(QtGui.QPixmap())
        self.pixmapItem.setPixmap(pixmap)
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        for i in range(len(instances)):
            instance = instances[i]
            self.thisScene.addRect(
                instance.x,
                instance.y,
                instance.width,
                instance.height,
                self.getpen(colors[i % len(colors)]),
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmapItem:
            try:
                self.fitInView(
                    self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio
                )
            except:
                pass

    @staticmethod
    def getpen(color: str) -> QtGui.QPen:
        pen = QtGui.QPen(color)
        pen.setWidth(10)
        return pen
