from __future__ import annotations
from PySide6 import QtCore, QtWidgets
from sqlalchemy import Select, and_, func
from db.models import Image, Instance
from detection.classes import CLASS_ID_MAPPING
from filters import Filter, DateFilter, TimeFilter


class AnalyzedFilter(Filter):
    name = "Analyzed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.thisLayout.insertWidget(0, QtWidgets.QLabel("Analyzed"))

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return query.where(Image.analyzed == True)  # noqa


class NotAnalyzedFilter(Filter):
    name = "Not Analyzed"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.thisLayout.insertWidget(0, QtWidgets.QLabel("Not analyzed"))

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return query.where(Image.analyzed == False)  # noqa


class ImageTimeFilter(TimeFilter):
    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return query.where(
            and_(
                Image.time >= self.startTime.time().toPython(),
                Image.time <= self.endTime.time().toPython(),
            )
        )


class ImageDateFilter(DateFilter):
    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return query.where(
            and_(
                Image.datetime >= self.startDate.dateTime().toPython(),
                Image.datetime <= self.endDate.dateTime().addSecs(86399).toPython(),
            )
        )


class EntityFilter(Filter):
    name = "Entity"
    expanded: bool

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

        self.typeFilter = QtWidgets.QComboBox()
        self.typeFilter.addItems(["Anything"])
        for typeId in CLASS_ID_MAPPING:
            self.typeFilter.addItems([CLASS_ID_MAPPING[typeId].title()])
        self.typeFilter.setMinimumWidth(100)
        self.typeFilter.currentTextChanged.connect(self.changed)
        self.thisLayout.insertWidget(3, self.typeFilter)

        self.minConfidence = QtWidgets.QDoubleSpinBox()
        self.minConfidence.setSuffix("%")
        self.minConfidence.setRange(0, 1)
        self.minConfidence.setValue(0.25)
        self.minConfidence.setSingleStep(0.05)
        self.minConfidence.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(4, self.minConfidence)

        self.minConfidenceLabel = QtWidgets.QLabel("confidence")
        self.thisLayout.insertWidget(5, self.minConfidenceLabel)

        self.expandButton = QtWidgets.QPushButton(">")
        self.expandButton.setMaximumWidth(25)
        self.thisLayout.insertWidget(6, self.expandButton)
        self.expandButton.pressed.connect(self.toggleExpand)

        self.expanded = True
        self.toggleExpand()

    @QtCore.Slot()
    def toggleExpand(self):
        self.expanded = not self.expanded
        self.minFilter.setVisible(self.expanded)
        self.dash.setVisible(self.expanded)
        self.maxFilter.setVisible(self.expanded)
        self.minConfidence.setVisible(self.expanded)
        self.minConfidenceLabel.setVisible(self.expanded)
        self.expandButton.setText("<" if self.expanded else ">")

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        if self.typeFilter.currentIndex() == 0:
            query = query.join(Image.instances).where(
                Instance.confidence >= self.minConfidence.value()
            )
        else:
            query = query.join(Image.instances).where(
                and_(
                    Instance.type_id == self.typeFilter.currentIndex() - 1,
                    Instance.confidence >= self.minConfidence.value(),
                )
            )

        if (
            self.minFilter.value() > 1
            or self.maxFilter.value() < self.maxFilter.maximum()
        ):
            query = query.group_by(Image.id).having(
                and_(
                    func.count(Instance.entity_id) >= self.minFilter.value(),
                    func.count(Instance.entity_id) <= self.maxFilter.value(),
                )
            )

        return query


class NoEntityFilter(Filter):
    name = "No Entity"
    expanded: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.thisLayout.insertWidget(0, QtWidgets.QLabel("No"))

        self.typeFilter = QtWidgets.QComboBox()
        for typeId in CLASS_ID_MAPPING:
            self.typeFilter.addItems([CLASS_ID_MAPPING[typeId].title()])
        self.typeFilter.setMinimumWidth(100)
        self.typeFilter.currentTextChanged.connect(self.changed)
        self.thisLayout.insertWidget(1, self.typeFilter)

        self.maxConfidence = QtWidgets.QDoubleSpinBox()
        self.maxConfidence.setSuffix("%")
        self.maxConfidence.setRange(0, 1)
        self.maxConfidence.setValue(0.25)
        self.maxConfidence.setSingleStep(0.05)
        self.maxConfidence.valueChanged.connect(self.changed)
        self.thisLayout.insertWidget(2, self.maxConfidence)

        self.maxConfidenceLabel = QtWidgets.QLabel("confidence")
        self.thisLayout.insertWidget(3, self.maxConfidenceLabel)

        self.expandButton = QtWidgets.QPushButton(">")
        self.expandButton.setMaximumWidth(25)
        self.thisLayout.insertWidget(4, self.expandButton)
        self.expandButton.pressed.connect(self.toggleExpand)

        self.expanded = True
        self.toggleExpand()

    @QtCore.Slot()
    def toggleExpand(self):
        self.expanded = not self.expanded
        self.maxConfidence.setVisible(self.expanded)
        self.maxConfidenceLabel.setVisible(self.expanded)
        self.expandButton.setText("<" if self.expanded else ">")

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        return (
            query.outerjoin(
                Instance,
                and_(
                    Image.id == Instance.image_id,
                    Instance.type_id == self.typeFilter.currentIndex(),
                ),
            )
            .group_by(Image.id)
            .having(
                func.coalesce(func.max(Instance.confidence), 0)
                <= self.maxConfidence.value()
            )
        )
