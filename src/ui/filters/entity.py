from PySide6 import QtCore, QtWidgets
from sqlalchemy import Select, and_

from db.models import Entity, Instance, Image
from detection.yolo import CLASS_ID_MAPPING
from filters import DateFilter, Filter, TimeFilter


class EntityDateFilter(DateFilter):
    @QtCore.Slot()
    def makeFilter(self, query: Select) -> Select:
        return (
            query.join(Entity.instances).join(Instance.image)
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
            query.join(Entity.instances).join(Instance.image)
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
            return query.where(Entity.ebike == True) # noqa
        else:
            return query.join(Entity.instances).where(
                Instance.type_id == self.typeFilter.currentIndex()
            )
