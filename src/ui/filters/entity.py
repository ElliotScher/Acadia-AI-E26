from PySide6 import QtCore, QtWidgets
from sqlalchemy import Select, and_, func, select

from db.models import Entity, Instance, Image
from detection.classes import CLASS_ID_MAPPING
from filters import DateFilter, Filter, TimeFilter


class EntityDateFilter(DateFilter):
    @QtCore.Slot()
    def makeFilter(self, query: Select) -> Select:
        return (
            query.join(Entity.instances)
            .join(Instance.image)
            .where(
                and_(
                    Image.datetime >= self.startDate.dateTime().toPython(),
                    Image.datetime <= self.endDate.dateTime().toPython(),
                )
            )
            .distinct(Entity.id)
        )


class EntityTimeFilter(TimeFilter):
    @QtCore.Slot()
    def makeFilter(self, query: Select) -> Select:
        return (
            query.join(Entity.instances)
            .join(Instance.image)
            .where(
                and_(
                    Image.time >= self.startTime.time().toPython(),
                    Image.time <= self.endTime.time().toPython(),
                )
            )
            .distinct(Entity.id)
        )


class EntityTypeFilter(Filter):
    name = "Type"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.typeFilter = QtWidgets.QComboBox()
        self.typeFilter.addItems(
            [CLASS_ID_MAPPING[x].title() for x in CLASS_ID_MAPPING]
        )
        self.typeFilter.addItem("Ebike")
        self.typeFilter.currentIndexChanged.connect(self.changed)
        self.thisLayout.insertWidget(0, self.typeFilter)

    @QtCore.Slot()
    def makeFilter(self, query: Select) -> Select:
        if self.typeFilter.currentIndex() == len(CLASS_ID_MAPPING):
            return query.where(Entity.ebike == True)  # noqa
        else:
            return query.join(Entity.instances).where(
                Instance.type_id == self.typeFilter.currentIndex()
            )


class ClusterSizeFilter(Filter):
    name = "Cluster Size"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.minFilter = QtWidgets.QSpinBox()
        self.minFilter.setRange(1, 100)
        self.minFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(0, self.minFilter)

        self.dash = QtWidgets.QLabel("-")
        self.thisLayout.insertWidget(1, self.dash)

        self.maxFilter = QtWidgets.QSpinBox()
        self.maxFilter.setRange(1, 100)
        self.maxFilter.setValue(100)
        self.maxFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(2, self.maxFilter)

        self.thisLayout.insertWidget(3, QtWidgets.QLabel("Cluster Size"))

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        subq = (
            select(Entity)
            .group_by(Entity.cluster)
            .having(
                and_(
                    func.count(Entity.id) >= self.minFilter.value(),
                    func.count(Entity.id) <= self.maxFilter.value(),
                )
            )
            .subquery()
        )
        return query.join(subq, Entity.cluster == subq.c.cluster)


class CountConfidenceFilter(Filter):
    name = "Count / Confidence"
    expanded: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.minFilter = QtWidgets.QSpinBox()
        self.minFilter.setRange(1, 100)
        self.minFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(0, self.minFilter)

        self.thisLayout.insertWidget(1, QtWidgets.QLabel("-"))

        self.maxFilter = QtWidgets.QSpinBox()
        self.maxFilter.setRange(1, 100)
        self.maxFilter.setValue(100)
        self.maxFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(2, self.maxFilter)

        self.thisLayout.insertWidget(3, QtWidgets.QLabel("instances"))

        self.minConfidence = QtWidgets.QDoubleSpinBox()
        self.minConfidence.setSuffix("%")
        self.minConfidence.setRange(0, 1)
        self.minConfidence.setValue(0.25)
        self.minConfidence.setSingleStep(0.05)
        self.minConfidence.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(4, self.minConfidence)

        self.thisLayout.insertWidget(5, QtWidgets.QLabel("confidence"))

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        query = query.join(Entity.instances).where(
            Instance.confidence >= self.minConfidence.value()
        )

        if (
            self.minFilter.value() > 1
            or self.maxFilter.value() < self.maxFilter.maximum()
        ):
            query = query.group_by(Entity.id).having(
                and_(
                    func.count(Instance.image_id) >= self.minFilter.value(),
                    func.count(Instance.image_id) <= self.maxFilter.value(),
                )
            )

        return query


class DirectionFilter(Filter):
    name = "Direction"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.directionFilter = QtWidgets.QComboBox()
        self.directionFilter.addItems(
            (
                "left",
                "right",
                "unknown left/right",
                "forward",
                "back",
                "unknown forward/back",
            )
        )
        self.directionFilter.currentIndexChanged.connect(self.changed)
        self.thisLayout.insertWidget(0, self.directionFilter)

    @QtCore.Slot()
    def makeFilter(self, query: Select) -> Select:
        if self.directionFilter.currentIndex() < 3:
            return query.join(Entity.instances).where(
                Instance.direction_lr
                == (
                    -1
                    if self.directionFilter.currentIndex() == 0
                    else (1 if self.directionFilter.currentIndex() == 1 else 0)
                )
            )
        else:
            return query.join(Entity.instances).where(
                Instance.direction_fb
                == (
                    -1
                    if self.directionFilter.currentIndex() == 4
                    else (1 if self.directionFilter.currentIndex() == 3 else 0)
                )
            )


class SpeedFilter(Filter):
    name = "Speed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.minFilter = QtWidgets.QSpinBox()
        self.minFilter.setRange(0, 100)
        self.minFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(0, self.minFilter)

        self.dash = QtWidgets.QLabel("-")
        self.thisLayout.insertWidget(1, self.dash)

        self.maxFilter = QtWidgets.QSpinBox()
        self.maxFilter.setRange(0, 100)
        self.maxFilter.setValue(100)
        self.maxFilter.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(2, self.maxFilter)

        self.thisLayout.insertWidget(3, QtWidgets.QLabel("mph"))

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return query.where(
            and_(
                Entity.speed >= self.minFilter.value(),
                Entity.speed <= self.maxFilter.value(),
            )
        )
