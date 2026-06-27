import math

from PySide6 import QtCore, QtGui, QtWidgets


class ImageTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        self.gallery = ImageGallery()
        layout.addWidget(self.gallery)

class ImageGallery(QtWidgets.QListWidget):
    def __init__(self):
        super().__init__()
        self.defaultIcon = QtGui.QIcon.fromTheme(QtGui.QIcon.ThemeIcon.ImageLoading)
        self.setUniformItemSizes(True)
        self.iconaspect = 1.5
        self.setIconSize(QtCore.QSize(math.floor(self.iconaspect * 100), 100))
        self.setViewMode(self.ViewMode.IconMode)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setResizeMode(self.ResizeMode.Adjust)

    def addImages(self, images: list[str]):
        for image in images:
            reader = QtGui.QImageReader(image)
            if reader.canRead():
                item = QtWidgets.QListWidgetItem(self)
                item.setIcon(QtGui.QIcon(image))
