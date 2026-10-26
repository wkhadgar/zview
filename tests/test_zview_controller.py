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

from zview.backend.base import HeapInfo, SemaphoreInfo, ThreadInfo, WorkqInfo
from zview.backend.image import ImageMismatch
from zview.frames import Fatal, Frame, FrameHistory, Notice, ReplayEnd
from zview.frontend import zview_tui
from zview.frontend.tui.views.base import SpecialCode, ZViewState
from zview.frontend.tui.views.thread_detail import ThreadDetailView
from zview.frontend.tui.views.thread_list import listed_threads
from zview.frontend.zview_tui import ZView
from zview.selection import Selection


@pytest.fixture
def app() -> ZView:
    """``ZView`` instance without going through curses init."""
    a = ZView.__new__(ZView)
    a.frames = FrameHistory()
    a.selection = Selection()
    a.status_message = ""
    a.messages = deque(maxlen=64)
    a.update_count = 0
    a.data_queue = queue.Queue()
    a.scraper = MagicMock()
    a.scraper.groups = {"threads": MagicMock(idle_addresses=(0xDEAD,))}
    a.state = ZViewState.THREAD_LIST_VIEW
    a.stdscr = MagicMock()
    a._overlay = None
    a._prompt = None
    a._group_cursor = 0
    return a


def test_purge_queue_clears_pending_items(app):
    for i in range(5):
        app.data_queue.put({"i": i})
    assert app.data_queue.qsize() == 5
    app.purge_queue()
    assert app.data_queue.empty()


def test_process_data_fatal_transitions_to_error_state(app):
    app.process_data(Fatal("synthetic loss"))
    assert app.state is ZViewState.FATAL_ERROR
    assert "TARGET LOST" in app.status_message
    assert "synthetic loss" in app.status_message


def test_process_data_replay_complete_sets_status_only(app):
    app.process_data(ReplayEnd("drained"))
    assert "Recording ended" in app.status_message
    # State is unchanged on a clean replay end.
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_error_sets_error_status_without_state_change(app):
    app.process_data(Notice("transient"))
    assert app.status_message.startswith("Error")
    assert "transient" in app.status_message
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_running_status_increments_with_threads_payload(app):
    """A normal frame: status reads ``Running...``, threads payload is captured."""
    threads = [
        ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None),
    ]
    app.process_data(Frame({"threads": threads}))
    assert app.status_message.startswith("Running")
    assert listed_threads(app) == threads
    assert app.update_count == 1


def test_a_frame_with_work_queues_records_their_backlog(app):
    """A frame without the group keeps the last list and adds no sample."""
    workq = WorkqInfo(name="bench_workq", address=0x6000, flags=0b11, pending=("trail_work",))

    app.process_data(Frame({"threads": [], "workqs": [workq]}))
    app.process_data(Frame({"threads": []}))

    assert app.frames.latest("workqs") == [workq]
    assert [q.depth for q in app.frames.of("workqs", 0x6000)] == [1]


def test_process_data_strips_every_idle_thread(app):
    """An SMP build idles on each CPU, and none of its idle threads is listed."""
    app.scraper.groups = {"threads": MagicMock(idle_addresses=(0xDEAD, 0xBEEF))}
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    idle_00 = ThreadInfo(
        address=0xDEAD, stack_start=0, stack_size=128, name="idle 00", runtime=None
    )
    idle_01 = ThreadInfo(
        address=0xBEEF, stack_start=0, stack_size=128, name="idle 01", runtime=None
    )

    app.process_data(Frame({"threads": [main, idle_00, idle_01]}))

    assert listed_threads(app) == [main]


def test_a_thread_detail_scans_only_the_detailed_threads_stack(app):
    """The idle thread's cycles are read with every other thread's, so the view names only it."""
    app.views = {
        ZViewState.THREAD_LIST_VIEW: MagicMock(),
        ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(app, MagicMock()),
    }
    app.detailing_thread = "main[0]"
    app.scraper.rows = {"threads": {"main[0]": [0x1000]}}
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
    app.process_data(Frame({"heaps": [heap]}))
    assert app.frames.latest("heaps") == [heap]


def test_process_data_empty_payload_does_not_clear_existing(app):
    """A no-op frame must not erase last-known data."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    app.frames.add(Frame({"threads": [main]}))
    app.process_data(Frame({}))
    assert listed_threads(app) == [main]


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

    app.process_data(Frame({"threads": [], "semaphores": [sem], "mutexes": [mutex]}))

    assert app.frames.latest("semaphores") == [sem]
    assert app.frames.latest("mutexes") == [mutex]


def test_process_data_without_kernel_objects_keeps_the_last_frame(app):
    """A frame without the keys leaves the last values in place."""
    from zview.backend.base import SemaphoreInfo

    sem = SemaphoreInfo(name="s", address=0x2000, count=1, limit=1, waiters=())
    app.process_data(Frame({"threads": [], "semaphores": [sem]}))

    app.process_data(Frame({"threads": []}))

    assert app.frames.latest("semaphores") == [sem]


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
        app.process_data(Frame({"threads": [], "mutexes": [frame_mutex]}))

    assert [m.state for m in app.frames.of("mutexes", 0x3000)] == [
        MutexState.FREE,
        MutexState.CONTENDED,
        MutexState.LOCKED,
    ]


def test_contention_history_is_bounded(app):
    """History is capped at 256 samples per mutex."""
    from zview.backend.base import MutexInfo

    mutex = MutexInfo(name="m", address=0x3000, lock_count=0, owner_address=0, waiters=())
    for _ in range(600):
        app.process_data(Frame({"threads": [], "mutexes": [mutex]}))

    assert len(app.frames.of("mutexes", 0x3000)) == 256


@pytest.fixture
def reloading_app(app):
    """``app`` ready for ``R``: a live scraper, no drawing, every history present."""
    app.stop_event = MagicMock()
    app.frames.add(Frame({"semaphores": [SemaphoreInfo("s", 0x10, 1, 4, ())]}))
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
    assert reloading_app.frames.latest("semaphores") == []
    assert reloading_app.frames.of("semaphores", 0x10) == []
    assert reloading_app._overlay is None


def test_a_reload_reads_the_muted_rows_again(reloading_app, monkeypatch):
    """The history starts over, and a muted row would never be read into it."""
    monkeypatch.setattr(zview_tui, "verify_target_image", lambda scraper, elf_path: None)
    reloading_app.toggle_mute("semaphores", "s")

    reloading_app.attempt_reconnect()

    assert reloading_app.selection.muted == frozenset()
    assert reloading_app.scraper.selection.muted == frozenset()


def test_a_replayed_row_hidden_or_muted_shows_again_after_the_replay_ends(app):
    """Until then a muted row stays at its last read."""
    app.scraper._m_scraper.is_live = False

    def frame(size: int) -> Frame:
        return Frame({"threads": [ThreadInfo(a, 0, size, n, None) for a, n in enumerate("abc", 1)]})

    app.process_data(frame(512))
    app.select(Selection(exclude=((None, "b"),), muted=frozenset({("threads", "c")})))
    app.process_data(frame(1024))
    assert [(t.name, t.stack_size) for t in listed_threads(app)] == [("a", 1024), ("c", 512)]

    app.select(Selection())
    assert [t.stack_size for t in listed_threads(app)] == [1024, 1024, 1024]


def test_a_muted_object_opens_on_its_last_read(app):
    from zview.frontend.tui.views.base import ZViewTUIAttributes
    from zview.frontend.tui.views.semaphore_detail import SemaphoreDetailView

    sem = SemaphoreInfo("gate", 0x10, 1, 4, ())
    app.process_data(Frame({"semaphores": [sem]}))
    app.toggle_mute("semaphores", "gate")
    app.process_data(Frame({"semaphores": []}))
    app.detailing = ("semaphores", 0x10)

    assert SemaphoreDetailView(app, ZViewTUIAttributes.create_mono())._target() == sem


def test_reload_onto_a_target_running_another_image_is_fatal(reloading_app, monkeypatch):
    def mismatch(scraper, elf_path):
        raise ImageMismatch("the target is not running this ELF")

    monkeypatch.setattr(zview_tui, "verify_target_image", mismatch)

    reloading_app.attempt_reconnect()

    assert reloading_app.state is ZViewState.FATAL_ERROR
    assert "the target is not running this ELF" in reloading_app.status_message
    reloading_app.scraper.start_polling_thread.assert_not_called()


def test_escape_drops_the_pattern_line_and_enter_selects_it(app):
    """The second line opens on the patterns selected, so a dropped edit would show in it."""
    keys = [
        ord("/"),
        *map(ord, "idle"),
        SpecialCode.ESCAPE,
        ord("/"),
        *map(ord, "maix"),
        SpecialCode.DELETE,
        ord("n"),
        SpecialCode.NEWLINE,
    ]
    app.stdscr.getch.side_effect = keys

    for _ in keys:
        app.process_events()

    assert app.selection.include == ((None, "main"),)
    assert app._prompt is None


def test_space_in_the_groups_popup_flips_the_group_under_its_cursor(app):
    """The popup lists the groups the session can read, and the others stay as they were."""
    app.scraper.available = {"threads": True, "heaps": False, "semaphores": True, "mutexes": True}
    app.scraper.groups = dict.fromkeys(app.scraper.available)
    keys = [ord("g"), curses.KEY_DOWN, SpecialCode.SPACE]
    app.stdscr.getch.side_effect = [*keys, SpecialCode.SPACE, SpecialCode.ESCAPE]

    for _ in keys:
        app.process_events()
    assert [key for key in app.scraper.groups if app.selection.reads_group(key)] == [
        "threads",
        "heaps",
        "mutexes",
    ]

    app.process_events()
    app.process_events()
    assert app.selection.reads_group("semaphores")
    assert app._overlay is None


def test_the_groups_popup_reopens_inside_a_list_that_shrank(app):
    """A reload onto a build with fewer groups leaves the cursor past the end."""
    app.scraper.available = dict.fromkeys(["threads", "heaps", "semaphores", "mutexes"], True)
    app.scraper.groups = dict.fromkeys(app.scraper.available)
    app._group_cursor = 3
    app.scraper.available = {"threads": True, "heaps": True}
    app.scraper.groups = dict.fromkeys(app.scraper.available)
    app.stdscr.getch.side_effect = [ord("g"), SpecialCode.SPACE]

    app.process_events()
    app.process_events()

    assert not app.selection.reads_group("heaps")


def test_a_pattern_line_naming_an_unknown_group_keeps_the_patterns(app):
    app.scraper.check_selection.side_effect = ValueError("Unknown group thread")

    app._select_patterns("thread:main")

    assert app.selection == Selection()
    assert app.messages[-1].text == "Error: Unknown group thread"
