# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``ZView`` controller logic that doesn't require real curses."""

import curses
import queue
import sys
from collections import deque
from unittest.mock import MagicMock

import pytest

from zview.backend.base import HeapInfo, ThreadInfo, WorkqInfo
from zview.backend.image import ImageMismatch
from zview.frontend import zview_tui
from zview.frontend.tui.views.base import ZViewState
from zview.frontend.zview_tui import ZView


@pytest.fixture
def app() -> ZView:
    """``ZView`` instance without going through curses init."""
    a = ZView.__new__(ZView)
    a.threads_data = []
    a.heaps_data = []
    a.semaphores_data = []
    a.mutexes_data = []
    a.mutex_history = {}
    a.sem_history = {}
    a.workqs_data = []
    a.workq_history = {}
    a.status_message = ""
    a.messages = deque(maxlen=64)
    a.update_count = 0
    a.data_queue = queue.Queue()
    a.scraper = MagicMock()
    a.scraper.idle_thread_addresses = (0xDEAD,)
    a.state = ZViewState.THREAD_LIST_VIEW
    a.stdscr = MagicMock()
    a._overlay = None
    return a


def test_purge_queue_clears_pending_items(app):
    for i in range(5):
        app.data_queue.put({"i": i})
    assert app.data_queue.qsize() == 5
    app.purge_queue()
    assert app.data_queue.empty()


def test_process_data_fatal_transitions_to_error_state(app):
    app.process_data({"fatal_error": "synthetic loss"})
    assert app.state is ZViewState.FATAL_ERROR
    assert "TARGET LOST" in app.status_message
    assert "synthetic loss" in app.status_message


def test_process_data_replay_complete_sets_status_only(app):
    app.process_data({"replay_complete": "drained"})
    assert "Recording ended" in app.status_message
    # State is unchanged on a clean replay end.
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_error_sets_error_status_without_state_change(app):
    app.process_data({"error": "transient"})
    assert app.status_message.startswith("Error")
    assert "transient" in app.status_message
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_running_status_increments_with_threads_payload(app):
    """A normal frame: status reads ``Running...``, threads payload is captured."""
    threads = [
        ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None),
    ]
    app.process_data({"threads": threads})
    assert app.status_message.startswith("Running")
    assert app.threads_data == threads
    assert app.update_count == 1


def test_a_frame_with_work_queues_records_their_backlog(app):
    """A frame without the group keeps the last list and adds no sample."""
    workq = WorkqInfo(name="bench_workq", address=0x6000, flags=0b11, pending=("trail_work",))

    app.process_data({"threads": [], "workqs": [workq]})
    app.process_data({"threads": []})

    assert app.workqs_data == [workq]
    assert list(app.workq_history[0x6000]) == [1]


def test_process_data_strips_every_idle_thread(app):
    """An SMP build idles on each CPU, and none of its idle threads is listed."""
    app.scraper.idle_thread_addresses = (0xDEAD, 0xBEEF)
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    idle_00 = ThreadInfo(
        address=0xDEAD, stack_start=0, stack_size=128, name="idle 00", runtime=None
    )
    idle_01 = ThreadInfo(
        address=0xBEEF, stack_start=0, stack_size=128, name="idle 01", runtime=None
    )

    app.process_data({"threads": [main, idle_00, idle_01]})

    assert app.threads_data == [main]


def test_a_thread_detail_scans_only_the_detailed_threads_stack(app):
    """The idle thread's cycles are read with every other thread's, so the view names only it."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main[0]", runtime=None)
    app.views = {ZViewState.THREAD_DETAIL_VIEW: MagicMock()}
    app.detailing_thread = "main[0]"
    app.scraper.all_threads = {"main[0]": main}
    app.scraper._m_scraper.is_live = True

    app.transition_to(ZViewState.THREAD_DETAIL_VIEW)

    assert app.scraper.view.reads("threads", "main[0]")
    assert not app.scraper.view.reads("threads", "main0")
    assert not app.scraper.view.reads_group("semaphores")


def test_process_data_heaps_payload_captured(app):
    heap = HeapInfo(
        name="my_heap",
        address=0x4000,
        free_bytes=100,
        allocated_bytes=50,
        max_allocated_bytes=80,
        usage_percent=33.0,
        chunks=None,
    )
    app.process_data({"heaps": [heap]})
    assert app.heaps_data == [heap]


def test_process_data_empty_payload_does_not_clear_existing(app):
    """A no-op frame must not erase last-known data."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    app.threads_data = [main]
    app.process_data({})
    assert app.threads_data == [main]


def test_a_resize_dismisses_nothing(app):
    """A resize arrives as a key, but it is a size change, not a keypress."""
    app._overlay = "messages"
    app.views = {app.state: MagicMock()}
    app.stdscr.getch.return_value = curses.KEY_RESIZE

    app.process_events()

    assert app._overlay == "messages"
    assert app.stdscr.clear.called
    assert not app.views[app.state].handle_input.called


def test_a_frame_rejected_mid_resize_does_not_end_the_session(app):
    """The screen shrank under the frame: drop it, the next one fits."""
    app.views = {app.state: MagicMock()}
    app.views[app.state].render.side_effect = curses.error("addwstr() returned ERR")
    app.stdscr.getmaxyx.return_value = (57, 209)
    app.min_dimensions = (14, 85)

    app.draw_frame()

    assert app.views[app.state].render.called


def test_process_data_stores_kernel_objects(app):
    """Semaphore and mutex frames land on the controller."""
    from zview.backend.base import MutexInfo, SemaphoreInfo

    sem = SemaphoreInfo(name="data_ready", address=0x2000, count=1, limit=4, waiters=())
    mutex = MutexInfo(name="spi_bus", address=0x3000, lock_count=0, owner_address=0)

    app.process_data({"threads": [], "semaphores": [sem], "mutexes": [mutex]})

    assert app.semaphores_data == [sem]
    assert app.mutexes_data == [mutex]


def test_process_data_without_kernel_objects_keeps_the_last_frame(app):
    """A frame without the keys leaves the last values in place."""
    from zview.backend.base import SemaphoreInfo

    sem = SemaphoreInfo(name="s", address=0x2000, count=1, limit=1, waiters=())
    app.process_data({"threads": [], "semaphores": [sem]})

    app.process_data({"threads": []})

    assert app.semaphores_data == [sem]


def test_contention_history_samples_once_per_frame(app):
    """Each frame appends one mark: free, held alone, or held with waiters."""
    from zview.backend.base import MutexInfo, MutexState

    free = MutexInfo(name="m", address=0x3000, lock_count=0, owner_address=0, waiters=())
    contended = MutexInfo(
        name="m",
        address=0x3000,
        lock_count=1,
        owner_address=0x4000,
        owner_name="t",
        waiters=("other",),
    )
    held_uncontended = MutexInfo(
        name="m", address=0x3000, lock_count=1, owner_address=0x4000, waiters=()
    )

    for frame_mutex in (free, contended, held_uncontended):
        app.process_data({"threads": [], "mutexes": [frame_mutex]})

    assert list(app.mutex_history[0x3000]) == [
        MutexState.FREE,
        MutexState.CONTENDED,
        MutexState.LOCKED,
    ]


def test_contention_history_is_bounded(app):
    """History is capped at 256 samples per mutex."""
    from zview.backend.base import MutexInfo

    mutex = MutexInfo(name="m", address=0x3000, lock_count=0, owner_address=0, waiters=())
    for _ in range(600):
        app.process_data({"threads": [], "mutexes": [mutex]})

    assert len(app.mutex_history[0x3000]) == 256


@pytest.fixture
def reloading_app(app):
    """``app`` ready for ``R``: a live scraper, no drawing, every history present."""
    app.stop_event = MagicMock()
    app.msgq_history = {}
    app.mem_slab_history = {}
    app.sem_history = {0x10: deque([1, 2])}
    app.scraper._m_scraper.is_live = True
    app.scraper.absent_features.return_value = ()
    app.draw_frame = MagicMock()
    app.transition_to = MagicMock()
    return app


def test_reload_reports_its_stderr_and_restarts_polling(reloading_app, monkeypatch):
    def reload_elf():
        print("Loaded cached ELF @ cache.bin", file=sys.stderr)

    reloading_app.scraper.reload_elf.side_effect = reload_elf
    monkeypatch.setattr(zview_tui, "verify_target_image", lambda scraper, elf_path: None)

    reloading_app.attempt_reconnect()

    assert any(entry.text == "Loaded cached ELF @ cache.bin" for entry in reloading_app.messages)
    reloading_app.scraper.start_polling_thread.assert_called_once()
    assert reloading_app.sem_history == {}
    assert reloading_app._overlay is None


def test_reload_onto_a_target_running_another_image_is_fatal(reloading_app, monkeypatch):
    def mismatch(scraper, elf_path):
        raise ImageMismatch("the target is not running this ELF")

    monkeypatch.setattr(zview_tui, "verify_target_image", mismatch)

    reloading_app.attempt_reconnect()

    assert reloading_app.state is ZViewState.FATAL_ERROR
    assert "the target is not running this ELF" in reloading_app.status_message
    reloading_app.scraper.start_polling_thread.assert_not_called()
