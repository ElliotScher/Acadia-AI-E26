from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from sqlalchemy import Select, union, intersect, select
from datetime import datetime


class Filters(QtWidgets.QWidget):
    filterTypes: list[Filter]
    changed = QtCore.Signal()

    def __init__(self, filterTypes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.filterTypes = filterTypes
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setVerticalSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetFixedSize
        )
        self.layout.setContentsMargins(2, 2, 2, 2)
        firstFilterRow = FilterRow(self.filterTypes, first=True)
        self.layout.addWidget(firstFilterRow)
        self.layout.setAlignment(firstFilterRow, QtCore.Qt.AlignmentFlag.AlignTop)
        firstFilterRow.newFilterRow.connect(self.newFilterRow)
        firstFilterRow.changed.connect(self.changed)

    @QtCore.Slot()
    def newFilterRow(self):
        newFilterRow = FilterRow(self.filterTypes)
        self.layout.addWidget(newFilterRow)
        self.layout.setAlignment(newFilterRow, QtCore.Qt.AlignmentFlag.AlignTop)
        newFilterRow.changed.connect(self.changed)
        self.changed.emit()

    @QtCore.Slot()
    def makeFilter(self, *args):
        selects = []
        for child in self.children():
            if hasattr(child, "makeFilter") and not child.deleted:
                selects.append(child.makeFilter(select(*args)))
        if len(selects) == 0:
            return select(*args)
        elif len(selects) == 1:
            return selects[0]
        return union(*list(map(lambda s: select(s.subquery()), selects)))


class FilterRow(QtWidgets.QGroupBox):
    filterTypes: list[Filter]
    deleted: bool
    newFilterRow = QtCore.Signal()
    changed = QtCore.Signal()

    def __init__(self, filterTypes, first=False, *args, **kwargs):
        super().__init__("" if first else "or", *args, **kwargs)

        self.deleted = False

        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.filterTypes = filterTypes
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setVerticalSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetFixedSize)
        if first:
            self.newFilterRowBtn = QtWidgets.QPushButton("+")
            self.newFilterRowBtn.setMaximumWidth(25)
            self.newFilterRowBtn.pressed.connect(self.newFilterRow)
            layout.addWidget(self.newFilterRowBtn)
        else:
            self.removeFilterRowBtn = QtWidgets.QPushButton("-")
            self.removeFilterRowBtn.setMaximumWidth(25)
            self.removeFilterRowBtn.pressed.connect(self.removeFilterRow)
            layout.addWidget(self.removeFilterRowBtn)

        self.filterScroll = QtWidgets.QScrollArea()
        filterScrollLayout = QtWidgets.QHBoxLayout()
        self.filterScroll.setMaximumHeight(50)
        self.filterScroll.setLayout(filterScrollLayout)
        self.filterScroll.setWidgetResizable(True)
        self.filterList = QtWidgets.QWidget()
        self.filterScroll.setWidget(self.filterList)
        self.filterListLayout = QtWidgets.QHBoxLayout(self.filterList)
        layout.addWidget(self.filterScroll)
        self.filterListLayout.setSpacing(2)
        self.filterListLayout.setSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetFixedSize
        )
        self.filterListLayout.setContentsMargins(2, 2, 2, 2)

        self.newFilterBtn = QtWidgets.QPushButton("+")
        self.newFilterBtn.pressed.connect(self.newFilter)
        self.newFilterBtn.setMaximumWidth(25)
        layout.addWidget(self.newFilterBtn)
        layout.setAlignment(self.newFilterBtn, QtCore.Qt.AlignmentFlag.AlignRight)

    @QtCore.Slot()
    def removeFilterRow(self):
        self.parentWidget().layout.removeWidget(self)
        self.deleteLater()
        self.deleted = True
        self.changed.emit()

    @QtCore.Slot()
    def newFilter(self):
        self.menu = QtWidgets.QMenu()
        for i in range(len(self.filterTypes)):
            action = self.menu.addAction(self.filterTypes[i].name)
            action.setData(i)
        self.menu.triggered.connect(self.newFilterSelected)
        pos = self.newFilterBtn.mapToGlobal(
            QtCore.QPointF(0, self.newFilterBtn.size().height())
        )
        self.menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())), None)

    @QtCore.Slot()
    def newFilterSelected(self, action: QtGui.QAction):
        newFilter = self.filterTypes[action.data()]()
        self.filterListLayout.addWidget(newFilter)
        self.filterListLayout.setAlignment(newFilter, QtCore.Qt.AlignmentFlag.AlignLeft)
        self.filterScroll.ensureWidgetVisible(newFilter)
        newFilter.changed.connect(self.changed)
        self.changed.emit()

    @QtCore.Slot()
    def makeFilter(self, query: Select):
        selects = []
        for child in self.filterList.children():
            if hasattr(child, "makeFilter") and not child.deleted:
                selects.append(child.makeFilter(query))
        if len(selects) == 0:
            return query
        elif len(selects) == 1:
            return selects[0]
        return intersect(*selects)


class Filter(QtWidgets.QGroupBox):
    name: str
    deleted: bool
    changed = QtCore.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.deleted = False

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setSpacing(5)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setHorizontalSizeConstraint(
            QtWidgets.QLayout.SizeConstraint.SetFixedSize
        )

        self.deleteFilterButton = QtWidgets.QPushButton("-")
        self.deleteFilterButton.setMaximumWidth(25)
        self.deleteFilterButton.pressed.connect(self.deleteFilter)
        self.layout.addWidget(self.deleteFilterButton)

    @QtCore.Slot()
    def deleteFilter(self):
        self.parentWidget().layout().removeWidget(self)
        self.deleteLater()
        self.deleted = True
        self.changed.emit()


class TimeFilter(Filter):
    name = "Time Range"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.startTime = QtWidgets.QTimeEdit()
        self.startTime.setDisplayFormat("hh:mm:ss")
        self.startTime.timeChanged.connect(self.changed)
        self.layout.insertWidget(0, self.startTime)

        self.layout.insertWidget(1, QtWidgets.QLabel("-"))

        self.endTime = QtWidgets.QTimeEdit()
        self.endTime.setDisplayFormat("hh:mm:ss")
        self.endTime.setTime(QtCore.QTime(23, 59, 59))
        self.endTime.timeChanged.connect(self.changed)
        self.layout.insertWidget(2, self.endTime)


class DateFilter(Filter):
    name = "Date Range"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.startDate = QtWidgets.QDateEdit()
        self.startDate.setDisplayFormat("yyyy-MM-dd")
        self.startDate.setCalendarPopup(True)
        self.startDate.setMaximumWidth(120)
        self.startDate.dateChanged.connect(self.changed)
        self.layout.insertWidget(0, self.startDate)

        self.layout.insertWidget(1, QtWidgets.QLabel("-"))

        self.endDate = QtWidgets.QDateEdit()
        self.endDate.setDisplayFormat("yyyy-MM-dd")
        self.endDate.setCalendarPopup(True)
        self.endDate.setMaximumWidth(120)
        self.endDate.setDate(
            QtCore.QDate(datetime.now().year, datetime.now().month, datetime.now().day)
        )
        self.endDate.dateChanged.connect(self.changed)
        self.layout.insertWidget(2, self.endDate)
