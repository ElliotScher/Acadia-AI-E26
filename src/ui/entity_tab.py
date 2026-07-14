from sympy.integrals.meijerint import z
import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from detection.yolo import CLASS_ID_MAPPING
from db.models import Entity, Instance, Image
from filters import Filters
from filters.entity import EntityDateFilter, EntityTimeFilter, EntityTypeFilter

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
    imageOpened = QtCore.Signal(Image)

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)

        gallerySide = QtWidgets.QWidget()
        gallerySideLayout = QtWidgets.QVBoxLayout(gallerySide)

        self.filters = Filters((EntityDateFilter, EntityTimeFilter, EntityTypeFilter))
        gallerySideLayout.addWidget(self.filters)
        self.filters.changed.connect(self.refreshGallery)
        self.count = QtWidgets.QLabel("0 entities")
        gallerySideLayout.addWidget(self.count)
        self.gallery = EntityGallery()
        gallerySideLayout.addWidget(self.gallery)

        layout.addWidget(gallerySide)
        self.entityInfo = EntityInfo()
        self.entityInfo.imageOpened.connect(self.imageOpened.emit)
        layout.addWidget(self.entityInfo)

    @QtCore.Slot()
    def newselection(self):
        selection = self.gallery.selectedIndexes()
        entity = None
        if len(selection) > 0:
            entity = self.galleryModel.getByIndex(selection[0])
        if entity:
            self.entityInfo.showEntity(entity, self.session)

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

        subquery = self.filters.makeFilter(Entity.id).subquery()
        query = (
            select(subquery).select_from(subquery).order_by(subquery.c.id).distinct()
        )
        self.galleryModel.results = list(
            map(lambda d: d[0], self.session.execute(query).unique().all())
        )
        self.count.setText(str(len(self.galleryModel.results)) + " entities")

    def getEntities(self, filtered: bool) -> list[Entity]:
        if not hasattr(self, "session"):
            return []

        if filtered:
            return list(map(self.galleryModel.getById, self.galleryModel.results))
        else:
            return list(self.session.scalars(select(Entity).distinct()))


class EntityGallery(QtWidgets.QListView):
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
    filters: Filters

    def __init__(self, session: Session):
        self.session = session
        self.thumbnails = dict()
        self.results = []
        super().__init__()

    def getByIndex(
        self, index: QtCore.QModelIndex | QtCore.QPersistentModelIndex
    ) -> Entity | None:
        return self.getById(self.results[index.row()])

    def getById(self, id: int) -> Entity:
        return self.session.scalar(
            select(Entity).where(Entity.id == id)
        )  # type: ignore

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
    imageOpened = QtCore.Signal(Image)

    def __init__(self):
        super().__init__()
        self.instances: list[Instance] = []

        self.setTitle("Entity Info")
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

    def showEntity(self, entity: Entity, session: Session):
        if entity:
            self.showInfo(entity, session)
            self.viewer.show()
        else:
            self.info.hide()
            self.viewer.hide()
            self.placeholder.show()

    def showInfo(self, entity: Entity, session: Session):
        self.instances = entity.get_instances(session)
        latest = entity.get_latest_image(session)
        self.images.clear()
        for i in range(len(self.instances)):
            instance = self.instances[i]
            widget = QtWidgets.QWidget(self.images)
            widget.setFixedHeight(40)
            layout = QtWidgets.QHBoxLayout(widget)
            icon = QtGui.QIcon(instance.image.path)
            iconWidget = QtWidgets.QLabel(parent=widget, pixmap=icon.pixmap(50, 50))
            layout.addWidget(iconWidget)
            c = instance.confidence
            color = (
                f"#{int(30 * c + 255 * (1 - c)):02x}{int(255 * c + 30 * (1 - c)):02x}1e"
            )
            date = instance.image.datetime.strftime("%c")
            label = QtWidgets.QLabel(
                f"<font color={color}>{c:0.2%}</font> taken on {date}",
            )
            layout.addWidget(label)
            item = QtWidgets.QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 40))
            button = QtWidgets.QPushButton(parent=widget)
            buttonicon = QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.ViewFullscreen)
            button.setIcon(buttonicon)
            button.setIconSize(QtCore.QSize(15, 15))
            button.setFixedSize(QtCore.QSize(25, 25))
            button.clicked.connect(lambda: self.openInImageTab(instance.image))
            layout.addWidget(button)
            self.images.addItem(item)
            if instance.image_id == latest.id:
                self.images.setSelection(
                    QtCore.QRect(0, i, 1, 1),
                    QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect,
                )
            self.images.setItemWidget(item, widget)
        if len(self.instances) > 0 and self.instances[0].type_id in CLASS_ID_MAPPING:
            self.typeLabel.setText(
                CLASS_ID_MAPPING[self.instances[0].type_id].title() + " seen in:"
            )
        self.viewer.set(entity, session)
        self.info.show()
        self.placeholder.hide()

    def buildinfo(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.typeLabel = QtWidgets.QLabel(parent=self)
        layout.addWidget(self.typeLabel)
        self.images = QtWidgets.QListWidget(parent=self)
        layout.addWidget(self.images)
        self.images.selectionModel().selectionChanged.connect(self.setViewImage)
        return widget

    @QtCore.Slot()
    def setViewImage(self):
        selection = self.images.selectedIndexes()
        if len(selection) < 1 or len(self.instances) <= selection[0].row():
            return
        instance = self.instances[selection[0].row()]
        self.viewer.setInstance(instance.image, instance)

    def openInImageTab(self, image: Image):
        self.imageOpened.emit(image)


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
                and_(Instance.image_id == image.id, Instance.entity_id == entity.id)
            )
        )
        if instance:
            self.setInstance(image, instance)

    @QtCore.Slot(Entity, Image, Instance)
    def setInstance(self, image: Image, instance: Instance):
        pixmap = QtGui.QPixmap(image.path)
        self.scene().addPixmap(pixmap)
        pen = QtGui.QPen("#ffffff")
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
        pen.setWidth(int((pixmap.width() + pixmap.height()) / 500))
        self.scene().addRect(
            instance.x,
            instance.y,
            instance.width,
            instance.height,
            pen,
        )
        self.fitInView(
            instance.x,
            instance.y,
            instance.width,
            instance.height,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        )

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            factor = 1 + event.angleDelta().y() / 1000
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
