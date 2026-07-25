import functools
from typing_extensions import Any
import typing

from PySide6 import QtCore


class Async(QtCore.QThread):
    """
    WARNING: Qt has multiple event loops which QObject instances belong to.
    Instantiating a QObject in a thread will often result in unintended behavior
    unless that object is moved to the main thread with `.moveToThread()`. Get
    the main thread with `QtWidgets.QApplication.instance().thread()`.
    """

    result = QtCore.Signal(Any)

    def __init__(self, name: str, fn: typing.Callable[["Async"], Any]):
        super().__init__()
        self.setObjectName(name)
        self.fn = fn
        self.started.connect(lambda: ThreadTracker().addThread(self))
        self.finished.connect(lambda: ThreadTracker().removeThread(self))

    @QtCore.Slot()
    def run(self):
        self.result.emit(self.fn(self))

    def progess(self, value: float):
        ThreadTracker().progressThread(self, value)

    @staticmethod
    def progress(value: float):
        thread = QtCore.QThread.currentThread()
        if not isinstance(thread, Async):
            return
        ThreadTracker().progressThread(thread, value)


class _T(QtCore.QObject):
    threadAdded = QtCore.Signal(QtCore.QThread)
    threadProgress = QtCore.Signal(QtCore.QThread, float)
    threadRemoved = QtCore.Signal(QtCore.QThread)

    threads: dict[QtCore.QThread, float] = {}
    mutex = QtCore.QMutex()

    @QtCore.Slot(QtCore.QThread)
    def addThread(self, thread: QtCore.QThread):
        self.mutex.lock()
        if thread in self.threads:
            return
        self.threads[thread] = 0
        self.threadAdded.emit(thread)
        self.mutex.unlock()

    @QtCore.Slot()
    def progressThread(self, thread: QtCore.QThread, value: float):
        self.mutex.lock()
        if thread not in self.threads:
            return
        self.threads[thread] = value
        self.threadProgress.emit(thread, value)
        self.mutex.unlock()

    @QtCore.Slot()
    def removeThread(self, thread: QtCore.QThread):
        self.mutex.lock()
        if thread not in self.threads:
            return
        del self.threads[thread]
        self.threadRemoved.emit(thread)
        self.mutex.unlock()

    def spinText(self):
        if len(self.threads) == 0:
            return "No background tasks."
        if len(self.threads) == 1:
            thread, progress = list(self.threads.items())[0]
            if progress == 0:
                return f"Waiting on {thread.objectName()}..."
            return f"Waiting on {thread.objectName()} ({progress:.2%})..."
        progress = functools.reduce(
            lambda v, t: t[1] + v, list(self.threads.items()), 0
        ) / len(self.threads)
        if progress == 0:
            return f"Waiting on {len(self.threads)} tasks..."
        return f"Waiting on {len(self.threads)} tasks ({progress:.2%})..."


class ThreadTracker(_T):
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = _T()
        return cls._instance

class ProgressTracker():
    def __init__(self, total: int, thread: Async | None = None):
        self.done = 0
        self.thread = thread
        self.total = total

    def update(self, n: int):
        self.done += n
        if self.thread is not None:
            self.thread.progess(self.done / self.total)
        else:
            Async.progress(self.done / self.total)