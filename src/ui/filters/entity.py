from PySide6 import QtCore, QtWidgets
from sqlalchemy import Select, and_, func, select

from db.models import Entity, Instance, Image
from detection.classes import CLASS_ID_MAPPING
from ui.filters import DateFilter, Filter, TimeFilter


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
