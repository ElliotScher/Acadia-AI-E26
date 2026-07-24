import typing
import cv2

from filters import Filters
from filters.image import (
    AnalyzedFilter,
    ClusterCountFilter,
    EntityFilter,
    ImageDateFilter,
    ImageTimeFilter,
    NoEntityFilter,
    NotAnalyzedFilter,
)
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Entity, Image, Instance
from detection.classes import CLASS_ID_MAPPING

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
    ) -> Image | None:
        return self.getById(self.results[index.row()])

    def getById(self, id: int) -> Image:
        return self.session.scalar(select(Image).where(Image.id == id))  # type: ignore

    def data(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex, role: int = 0
    ) -> typing.Any:
        if role != QtCore.Qt.ItemDataRole.DecorationRole:
            return
        data = self.getByIndex(index)
        if not data:
            return
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
                ClusterCountFilter,
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
        selection = self.gallery.selectedIndexes()
        image: Image | None = None
        if len(selection) > 0:
            image = self.galleryModel.getByIndex(selection[0])
        self.imageInfo.showImage(image, self.session)

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
        self.setSelectionMode(self.SelectionMode.SingleSelection)
        self.setDragEnabled(False)
        self.setLayoutMode(self.LayoutMode.Batched)
        self.setBatchSize(100)


class ImageInfo(QtWidgets.QGroupBox):
    entityOpened = QtCore.Signal(Entity)
    image: Image | None

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
        self.image = image
        self.session = session
        if image:
            self.showInfo(image)
        else:
            self.info.hide()
            self.viewer.hide()
            self.placeholder.show()

    def showInfo(self, image: Image):
        instances = image.get_instances(self.session)
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
            directions: list[str] = []
            if instance.direction_lr == -1:
                directions.append("left")
            elif instance.direction_lr == 1:
                directions.append("right")
            if instance.direction_fb == -1:
                directions.append("back")
            elif instance.direction_fb == 1:
                directions.append("forward")
            widget = QtWidgets.QWidget(self.entities)
            widget.setFixedHeight(40)
            layout = QtWidgets.QHBoxLayout(widget)
            tlabel = QtWidgets.QLabel(
                f"<font color={colors[i % len(colors)]}>{CLASS_ID_MAPPING[instance.type_id] + ((' (' + ', '.join(directions) + ')') if len(directions) > 0 else '')}</font>",
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
        hb = QtWidgets.QHBoxLayout(widget)
        layout.addLayout(hb)
        self.imgdate = QtWidgets.QLabel("A long time ago...", widget)
        hb.addWidget(self.imgdate)
        self.exportButton = QtWidgets.QToolButton(widget)
        self.exportButton.setIcon(
            QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.DocumentSave)
        )
        self.exportButton.pressed.connect(self.exportImage)
        hb.addWidget(self.exportButton)
        self.entities = QtWidgets.QListWidget(widget)
        layout.addWidget(self.entities)
        self.entities.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        return widget

    @QtCore.Slot()
    def exportImage(self):
        if not self.image:
            return
        imbuf = cv2.imread(str(self.image.path))
        if imbuf is None:
            return
        path = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save as...",
            str(self.image.path),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )[0]
        if not path:
            return
        i = 0
        for instance in self.image.get_instances(self.session):
            c = colors[i]
            cv2.rectangle(
                imbuf,
                (instance.x, instance.y),
                (instance.x + instance.width, instance.y + instance.height),
                (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)),
                (imbuf.shape[0] + imbuf.shape[1]) // 500,
            )
            i += 1
        cv2.imwrite(path, imbuf)


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.thisScene = QtWidgets.QGraphicsScene(self)
        self.setScene(self.thisScene)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set(self, image: Image, instances: list[Instance]):
        self.thisScene.clear()
        pixmap = QtGui.QPixmap(image.path)
        pixmapItem = self.thisScene.addPixmap(pixmap)
        pen = QtGui.QPen()
        pen.setWidth(int((pixmap.width() + pixmap.height()) / 500))
        for i in range(len(instances)):
            pen.setColor(colors[i % len(colors)])
            instance = instances[i]
            self.thisScene.addRect(
                instance.x,
                instance.y,
                instance.width,
                instance.height,
                pen,
            )
        self.fitInView(pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1 + event.angleDelta().y() / 1000
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
