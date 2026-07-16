import os

import pytest
from PySide6 import QtCore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.utility.parallel import Async, ThreadTracker


@pytest.fixture(autouse=True)
def clean_thread_tracker():
    ThreadTracker().threads.clear()
    yield
    ThreadTracker().threads.clear()


def test_thread_tracker_is_a_singleton():
    assert ThreadTracker() is ThreadTracker()


def test_spin_text_no_tasks():
    assert ThreadTracker().spinText() == "No background tasks."


def test_spin_text_single_task_no_progress():
    thread = Async("job1", lambda: None)
    ThreadTracker().addThread(thread)
    assert ThreadTracker().spinText() == "Waiting on job1..."


def test_spin_text_single_task_with_progress():
    thread = Async("job1", lambda: None)
    tracker = ThreadTracker()
    tracker.addThread(thread)
    tracker.progressThread(thread, 0.5)
    assert tracker.spinText() == "Waiting on job1 (50.00%)..."


def test_spin_text_multiple_tasks_no_progress():
    tracker = ThreadTracker()
    tracker.addThread(Async("job1", lambda: None))
    tracker.addThread(Async("job2", lambda: None))
    assert tracker.spinText() == "Waiting on 2 tasks..."


def test_spin_text_multiple_tasks_with_average_progress():
    tracker = ThreadTracker()
    t1 = Async("job1", lambda: None)
    t2 = Async("job2", lambda: None)
    tracker.addThread(t1)
    tracker.addThread(t2)
    tracker.progressThread(t1, 0.4)
    tracker.progressThread(t2, 0.6)
    assert tracker.spinText() == "Waiting on 2 tasks (50.00%)..."


def test_add_thread_is_idempotent():
    tracker = ThreadTracker()
    thread = Async("job1", lambda: None)
    tracker.addThread(thread)
    tracker.addThread(thread)
    assert len(tracker.threads) == 1


def test_remove_thread_not_tracked_is_a_noop():
    tracker = ThreadTracker()
    thread = Async("job1", lambda: None)
    # never added; should not raise
    tracker.removeThread(thread)
    assert len(tracker.threads) == 0


def test_progress_thread_not_tracked_is_a_noop():
    tracker = ThreadTracker()
    thread = Async("job1", lambda: None)
    tracker.progressThread(thread, 0.9)
    assert len(tracker.threads) == 0


def test_async_run_emits_result():
    results = []
    thread = Async("job1", lambda: 42)
    thread.result.connect(lambda value: results.append(value))
    thread.run()
    assert results == [42]
