import typing

from analyze_dialog import AnalyzeDialog
from filters import Filters
from filters.image import (
    AnalyzedFilter,
    EntityFilter,
    ImageDateFilter,
    ImageTimeFilter,
    NoEntityFilter,
    NotAnalyzedFilter,
)
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Image, Instance
from detection.yolo import CLASS_ID_MAPPING, load_model

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

    def getById(self, id: int) -> Image | None:
        return self.session.scalar(select(Image).where(Image.id == id))

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
        newmax = min(
            self.size + 300,
            len(self.results),
        )
        self.beginInsertRows(QtCore.QModelIndex(), self.size, newmax - 1)
        self.size = newmax
        self.endInsertRows()

    def canFetchMore(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex, /
    ) -> bool:
        return self.size < len(self.results)


class ImageTab(QtWidgets.QWidget):
    session: Session

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
        layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection = self.gallery.selectedIndexes()
        image: Image | None = None
        if len(selection) > 0:
            image = self.galleryModel.getByIndex(selection[0])
        self.imageInfo.showinfo(image, self.session)

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

    @QtCore.Slot()
    def analyze(self, filtered: bool):
        if not hasattr(self, "session"):
            return

        if not hasattr(self, "yoloModel"):
            self.yoloModel = load_model("yolo26s.pt")

        images: list[Image]
        if filtered:
            images = list(map(self.galleryModel.getById, self.galleryModel.results))
        else:
            images = list(
                self.session.scalars(select(Image).order_by(Image.datetime).distinct())
            )

        dialog = AnalyzeDialog(self.session, images)
        dialog.accepted.connect(self.refreshGallery)
        dialog.exec()

    @QtCore.Slot()
    def export(self, filtered: bool, path: str):
        if not hasattr(self, "session"):
            return

        if filtered:
            images = list(map(self.galleryModel.getById, self.galleryModel.results))
        else:
            images = list(
                self.session.scalars(select(Image).order_by(Image.datetime).distinct())
            )

        Image.export_to_csv(self.session, images, path)


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

    def showinfo(self, image: Image | None, session: Session):
        if image:
            self.showImg(image, session)
            self.viewer.show()
        else:
            self.info.hide()
            self.viewer.hide()
            self.placeholder.show()

    def showImg(self, image: Image, session: Session):
        instances = image.get_instances(session)
        instancesText = ""
        for i in range(len(instances)):
            instance = instances[i]
            instancesText += (
                '<font color="'
                + colors[i % len(colors)]
                + '">'
                + CLASS_ID_MAPPING[instance.type_id].title()
                + " "
                + str(round(instance.confidence * 10000) / 100)
                + "% confidence</font><br>"
            )

        self.viewer.set(image, instances)
        self.imgdate.setText(image.datetime.strftime("%Y-%m-%d %H:%M:%S"))
        self.imginstances.setText(instancesText)
        self.info.show()
        self.placeholder.hide()

    def buildinfo(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.imgdate = QtWidgets.QLabel("A long time ago...")
        layout.addWidget(self.imgdate)
        self.imginstances = QtWidgets.QLabel()
        layout.addWidget(self.imginstances)
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
