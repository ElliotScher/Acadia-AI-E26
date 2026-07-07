import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select
from sqlalchemy.orm import Session

from detection.yolo import CLASS_ID_MAPPING
from db.models import Entity, Instance
from filters import Filters

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


class EntitiesTab(QtWidgets.QWidget):
    session: Session

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        gallerySide = QtWidgets.QWidget()
        gallerySideLayout = QtWidgets.QVBoxLayout(gallerySide)

        self.filters = Filters(())
        gallerySideLayout.addWidget(self.filters)
        self.filters.changed.connect(self.refreshGallery)
        self.count = QtWidgets.QLabel("0 images")
        gallerySideLayout.addWidget(self.count)
        self.gallery = EntityGallery()
        gallerySideLayout.addWidget(self.gallery)

        layout.addWidget(gallerySide)
        self.imageInfo = EntityInfo()
        layout.addWidget(self.imageInfo)

    @QtCore.Slot()
    def newselection(self):
        selection: list[Entity] = [
            x
            for x in [
                self.galleryModel.getByIndex(x) for x in self.gallery.selectedIndexes()
            ]
            if x is not None
        ]
        self.imageInfo.showinfo(selection, self.session)

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

        self.galleryModel.results = [x.id for x in self.session.scalars(select(Entity)).all()]
        self.count.setText(str(len(self.galleryModel.results)) + " images")

class EntityGallery(QtWidgets.QListView):
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


class GalleryModel(QtCore.QAbstractListModel):
    session: Session
    results: list[int]
    thumbnails: dict[int, QtGui.QIcon]
    size: int = 0
    filters: "Filters"

    def __init__(self, session: Session):
        self.session = session
        self.thumbnails = dict()
        self.results = []
        super().__init__()

    def getByIndex(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> Entity | None:
        return self.session.scalar(
            select(Entity).where(Entity.id == self.results[index.row()])
        )

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
                self.thumbnails = dict()
            img = QtGui.QIcon(data.get_latest_image(self.session).path)
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


class EntityInfo(QtWidgets.QGroupBox):
    def __init__(self):
        super().__init__()
        self.setTitle("Image Info")
        self.setMinimumSize(400, 500)
        self.setMaximumWidth(400)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(self.alignment().AlignTop)

        self.viewer = EntityViewer()
        self.viewer.resize(300, 150)
        self.viewer.hide()
        layout.addWidget(self.viewer)

        self.placeholder = QtWidgets.QWidget()
        layout.addWidget(self.placeholder)

        self.info = self.buildinfo()
        layout.addWidget(self.info)
        self.info.hide()

    def showinfo(self, images: list[Entity], session: Session):
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

    def showone(self, entity: Entity, session: Session):
        instances = entity.get_instances(session)
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

        self.viewer.set(entity, session)
        self.imgcount.setText("1 selected.\n")
        self.imginstances.setText(instancesText)
        self.info.show()
        self.placeholder.hide()

    def showmultiple(self, images: list[Entity]):
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
        self.imginstances = QtWidgets.QLabel()
        layout.addWidget(self.imginstances)
        return widget


class EntityViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set(self, entity: Entity, session: Session):
        self.scene().clear()

        image = entity.get_latest_image(session)
        instance = session.scalar(
            select(Instance).where(
                Instance.image_id == image.id and Instance.entity_id == entity.id
            )
        )
        if not instance:
            return
        pixmap = QtGui.QPixmap(image.path)
        print(instance.x, instance.y, instance.width, instance.height, pixmap.size())
        self.pixmapItem = self.scene().addPixmap(
            pixmap.copy(
                instance.x,
                instance.y,
                instance.width,
                instance.height,
            )
        )
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmapItem:
            self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    @staticmethod
    def getpen(color: str) -> QtGui.QPen:
        pen = QtGui.QPen(color)
        pen.setWidth(10)
        return pen
