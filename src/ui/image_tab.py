import typing

from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import select, Select, func, union
from sqlalchemy.orm import Session
from datetime import datetime, time

from detection.yolo import load_model, CLASS_ID_MAPPING
from db.models import Image, Instance
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
    ) -> Image:
# <<<<<<< HEAD
        return self.session.scalar(
            select(Image).where(Image.id == self.results[index.row()])
        )
# =======
#         return self.session.scalar(self.filters.makeFilter(select(Image)).order_by(Image.datetime).offset(index.row()).limit(1))
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)

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
                        self.thumbnails = dict()
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
# <<<<<<< HEAD
            len(self.results),
# =======
#             self.filters.makeFilter(self.session.query(Image)).count(),  # type: ignore
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)
        )
        self.beginInsertRows(QtCore.QModelIndex(), self.size, newmax - 1)
        self.size = newmax
        self.endInsertRows()

    def canFetchMore(
        self, parent: QtCore.QModelIndex | QtCore.QPersistentModelIndex, /
    ) -> bool:
# <<<<<<< HEAD
        return self.size < len(self.results)
# =======
#         return self.size < self.filters.makeFilter(self.session.query(Image)).count()  # type: ignore
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)


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
        selection: list[Image] = list(
            map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())
        )
        self.imageInfo.showinfo(selection, self.session)

    @QtCore.Slot()
    def setsession(self, session: Session):
        self.session = session

        self.refreshGallery()

# <<<<<<< HEAD
# =======
#         minDt = datetime_to_qdatetime(Image.get_earliest_image(session).datetime) # type: ignore
#         maxDt = datetime_to_qdatetime(Image.get_latest_image(session).datetime) # type: ignore
#         self.filters.startDate.setDateTimeRange(minDt, maxDt)
#         self.filters.endDate.setDateTimeRange(minDt, maxDt)
#         self.filters.startDate.setDateTime(minDt)
#         self.filters.endDate.setDateTime(maxDt)
#
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)
    @QtCore.Slot()
    def refreshGallery(self):
        if not hasattr(self, "session"):
            return
        self.galleryModel = GalleryModel(self.session)
        self.galleryModel.filters = self.filters
        self.gallery.setModel(self.galleryModel)
        self.gallery.selectionModel().selectionChanged.connect(self.newselection)

# <<<<<<< HEAD
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
# =======
#         count = self.filters.makeFilter(self.session.query(Image)).count()  # type: ignore
#         self.count.setText(str(count) + " images")
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)

    @QtCore.Slot()
    def analyze(self, filtered: bool):
        if not hasattr(self, "session"):
            return

        if not hasattr(self, "yoloModel"):
            self.yoloModel = load_model("yolo26s.pt")

        images: list[Image] = []
        if filtered:
# <<<<<<< HEAD
            images = self.session.scalars(self.filters.makeFilter(Image)).all()
# =======
#             images = self.session.scalars(self.filters.makeFilter(select(Image))).all()  # type: ignore
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)
        else:
            images = list(
                map(self.galleryModel.getByIndex, self.gallery.selectedIndexes())
            )

# <<<<<<< HEAD
        dialog = AnalyzeDialog(self.session, self.yoloModel, images)
        dialog.accepted.connect(self.refreshGallery)
        dialog.exec()
# =======
#         for image in images:
#             image.analyze(self.session, self.yoloModel)
#
#         self.refreshGallery()
#
#
# class Filters(QtWidgets.QWidget):
#     typeFilterMap: dict[str, int]
#
#     def __init__(self):
#         super().__init__()
#         filterLayout = QtWidgets.QHBoxLayout(self)
#
#         self.startDate = QtWidgets.QDateTimeEdit()
#         self.startDate.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
#         self.startDate.setCalendarPopup(True)
#         self.startDate.dateTimeChanged.connect(self.refreshGallery)
#         filterLayout.addWidget(self.startDate)
#
#         filterLayout.addWidget(QtWidgets.QLabel("to"))
#
#         self.endDate = QtWidgets.QDateTimeEdit()
#         self.endDate.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
#         self.endDate.setCalendarPopup(True)
#         self.endDate.dateTimeChanged.connect(self.refreshGallery)
#         filterLayout.addWidget(self.endDate)
#
#         self.analyzedFilter = QtWidgets.QComboBox()
#         self.analyzedFilter.addItems(["All", "Only Analyzed", "Only Unanalyzed"])
#         self.analyzedFilter.currentIndexChanged.connect(self.refreshGallery)
#         filterLayout.addWidget(self.analyzedFilter)
#
#         self.minFilter = QtWidgets.QSpinBox()
#         self.minFilter.setRange(0, 100)
#         self.minFilter.valueChanged.connect(self.refreshGallery)
#         filterLayout.addWidget(self.minFilter)
#
#         filterLayout.addWidget(QtWidgets.QLabel("-"))
#
#         self.maxFilter = QtWidgets.QSpinBox()
#         self.maxFilter.setRange(0, 100)
#         self.maxFilter.setValue(100)
#         self.maxFilter.valueChanged.connect(self.refreshGallery)
#         filterLayout.addWidget(self.maxFilter)
#
#         self.typeFilter = QtWidgets.QComboBox()
#         self.typeFilter.currentIndexChanged.connect(self.refreshGallery)
#         self.typeFilterMap = dict()
#         typeFilterOptions = ["All"]
#         for type_id in CLASS_ID_MAPPING:
#             if CLASS_ID_MAPPING[type_id] not in typeFilterOptions:
#                 typeFilterOptions.append(CLASS_ID_MAPPING[type_id])
#                 self.typeFilterMap[CLASS_ID_MAPPING[type_id]] = type_id
#         self.typeFilter.addItems(typeFilterOptions)
#         filterLayout.addWidget(self.typeFilter)
#
#     def makeFilter(self, select: Select) -> Select:
#         filters = select.where(
#             Image.datetime >= self.startDate.dateTime().toPython()
#         ).where(Image.datetime <= self.endDate.dateTime().toPython())
#
#         if self.analyzedFilter.currentIndex() == 1:
#             filters = filters.where(Image.analyzed)
#         elif self.analyzedFilter.currentIndex() == 2:
#             filters = filters.where(Image.analyzed)
#
#         if (
#             self.typeFilter.currentIndex() != 0
#             or self.minFilter.value() > 0
#             or self.maxFilter.value() < self.maxFilter.maximum()
#         ):
#             filters = filters.join(Image.instances).group_by(Image.id)
#             if self.typeFilter.currentIndex() != 0:
#                 filters = filters.having(
#                     Instance.type_id
#                     == self.typeFilterMap[self.typeFilter.currentText()]
#                 )
#             if self.minFilter.value() > 0:
#                 filters = filters.having(
#                     func.count(Instance.entity_id) >= self.minFilter.value()
#                 )
#             if self.maxFilter.value() < self.maxFilter.maximum():
#                 filters = filters.having(
#                     func.count(Instance.entity_id) <= self.maxFilter.value()
#                 )
#         return filters
#
#     @QtCore.Slot()
#     def refreshGallery(self):
#         if self.parentWidget() and self.parentWidget().parentWidget(): # type: ignore
#             self.parentWidget().parentWidget().refreshGallery() # type: ignore
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)


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
        self.imgcount.setText("1 selected.\n")
        self.imgdate.setText(image.datetime.strftime("%Y-%m-%d %H:%M:%S"))
        self.imginstances.setText(instancesText)
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
        self.imginstances = QtWidgets.QLabel()
        layout.addWidget(self.imginstances)
        return widget


class ImageViewer(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def set(self, image: Image, instances: list[Instance]):
        self._scene.clear()

        pixmap = QtGui.QPixmap(image.path)
        self.resize(pixmap.width(), pixmap.height())
        self.pixmapItem = self._scene.addPixmap(QtGui.QPixmap())
        self.pixmapItem.setPixmap(pixmap)
        self.fitInView(self.pixmapItem, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

# <<<<<<< HEAD
        for i in range(len(instances)):
            instance = instances[i]
            self.scene.addRect(
# =======
#         for instance in instances:
#             self._scene.addRect(
# >>>>>>> 08c1dec (refactor(ui): added parallel utility module)
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
