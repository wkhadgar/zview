# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``ZView`` controller logic that doesn't require real curses."""

import curses
import sys
from unittest.mock import MagicMock

import pytest

from zview.backend.base import HeapInfo, SemaphoreInfo, ThreadInfo, WorkqInfo
from zview.backend.image import ImageMismatch
from zview.frames import Fatal, Frame, Notice, ReplayEnd
from zview.frontend.tui.views.base import SpecialCode, ZViewState
from zview.frontend.tui.views.thread_detail import ThreadDetailView
from zview.frontend.tui.views.thread_list import listed_threads
from zview.frontend.zview_tui import ZView
from zview.selection import Selection


def _feed(app: ZView, frame: Frame) -> None:
    """Poll ``frame`` through the core, as the TUI's poll does, and hand it to the controller."""
    app.scraper.read_frame.return_value = frame.groups
    app.core.poll()
    app.process_data(app.data_queue.get_nowait())


def _open_objects(app: ZView) -> None:
    """Open the kernel objects view, so the core reads the objects."""
    app.state = ZViewState.KERNEL_OBJECT_LIST_VIEW
    app.select(app.patterns)


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
    _feed(app, Frame({"threads": threads}))
    assert app.status_message.startswith("Running")
    assert listed_threads(app) == threads
    assert app.update_count == 1


def test_a_frame_with_work_queues_records_their_backlog(app):
    """A frame without the group keeps the last list and adds no sample."""
    workq = WorkqInfo(name="bench_workq", address=0x6000, flags=0b11, pending=("trail_work",))

    _feed(app, Frame({"threads": [], "workqs": [workq]}))
    _feed(app, Frame({"threads": []}))

    assert app.frames.latest("workqs") == [workq]
    assert [q.depth for q in app.frames.of("workqs", 0x6000)] == [1]


def test_process_data_strips_every_idle_thread(app):
    """An SMP build idles on each CPU, and none of its idle threads is listed."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    idle_00 = ThreadInfo(0xDEAD, 0, 128, "idle 00", None, idle=True)
    idle_01 = ThreadInfo(0xBEEF, 0, 128, "idle 01", None, idle=True)

    _feed(app, Frame({"threads": [main, idle_00, idle_01]}))

    assert listed_threads(app) == [main]


def test_a_thread_detail_scans_only_the_detailed_threads_stack(app):
    """The idle thread's cycles are read with every other thread's, so the view names only it."""
    app.views = {
        ZViewState.THREAD_LIST_VIEW: MagicMock(),
        ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(app, MagicMock()),
    }
    app.detailing_thread = "main[0]"
    app.scraper.rows = {"threads": {"main[0]": [0x1000]}}
    app.scraper.is_live = True

    app.transition_to(ZViewState.THREAD_DETAIL_VIEW)

    assert app.scraper.selection.reads("threads", "main[0]")
    assert not app.scraper.selection.reads("threads", "main0")
    assert not app.scraper.selection.reads_group("semaphores")


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
    _feed(app, Frame({"heaps": [heap]}))
    assert app.frames.latest("heaps") == [heap]


def test_process_data_empty_payload_does_not_clear_existing(app):
    """A no-op frame must not erase last-known data."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    app.frames.add(Frame({"threads": [main]}))
    _feed(app, Frame({}))
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

    _feed(app, Frame({"threads": [], "semaphores": [sem], "mutexes": [mutex]}))

    assert app.frames.latest("semaphores") == [sem]
    assert app.frames.latest("mutexes") == [mutex]


def test_process_data_without_kernel_objects_keeps_the_last_frame(app):
    """A frame without the keys leaves the last values in place."""
    from zview.backend.base import SemaphoreInfo

    sem = SemaphoreInfo(name="s", address=0x2000, count=1, limit=1, waiters=())
    _feed(app, Frame({"threads": [], "semaphores": [sem]}))

    _feed(app, Frame({"threads": []}))

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
        _feed(app, Frame({"threads": [], "mutexes": [frame_mutex]}))

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
        _feed(app, Frame({"threads": [], "mutexes": [mutex]}))

    assert len(app.frames.of("mutexes", 0x3000)) == 256


@pytest.fixture
def reloading_app(app):
    """``app`` ready for ``R``: a live scraper, no drawing, every history present."""
    app.core.start = MagicMock()
    app.core.stop = MagicMock()
    app.core.period = 0.1
    app.frames.add(Frame({"semaphores": [SemaphoreInfo("s", 0x10, 1, 4, ())]}))
    app.scraper.is_live = True
    app.scraper.absent_features.return_value = ()
    app.draw_frame = MagicMock()
    app.transition_to = MagicMock()
    return app


def test_reload_reports_its_stderr_and_restarts_polling(reloading_app):
    def reattach():
        print("Loaded cached ELF @ cache.bin", file=sys.stderr)

    reloading_app.scraper.reattach.side_effect = reattach

    reloading_app.attempt_reconnect()

    assert any(entry.text == "Loaded cached ELF @ cache.bin" for entry in reloading_app.messages)
    reloading_app.core.start.assert_called_once_with(0.1)
    assert reloading_app.frames.latest("semaphores") == []
    assert reloading_app.frames.of("semaphores", 0x10) == []
    assert reloading_app._overlay is None


def test_a_reload_reads_the_muted_rows_again(reloading_app):
    """The history starts over, and a muted row would never be read into it."""
    _open_objects(reloading_app)
    reloading_app.toggle_mute("semaphores", "s")

    reloading_app.attempt_reconnect()

    assert reloading_app.muted == {}
    assert reloading_app.core.selection.reads("semaphores", "s")


def test_a_replayed_row_hidden_or_muted_shows_again_after_the_replay_ends(app):
    """Until then a muted row stays at its last read."""
    app.scraper.is_live = False

    def frame(size: int) -> Frame:
        return Frame({"threads": [ThreadInfo(a, 0, size, n, None) for a, n in enumerate("abc", 1)]})

    _feed(app, frame(512))
    app.toggle_mute("threads", "c")
    app.select(Selection(exclude=((None, "b"),)))
    _feed(app, frame(1024))
    assert [(t.name, t.stack_size) for t in listed_threads(app)] == [("a", 1024), ("c", 512)]

    app.toggle_mute("threads", "c")
    app.select(Selection())
    assert [t.stack_size for t in listed_threads(app)] == [1024, 1024, 1024]


def test_the_usage_window_keys_double_and_halve_it_within_its_bounds(app):
    app.scraper.parts = {"usage_snapshot"}
    threads = app.scraper.groups["threads"]
    threads.usage_window = 1

    app.scale_usage_window(wider=False)
    assert threads.usage_window == 1

    for _ in range(8):
        app.scale_usage_window(wider=True)
    assert threads.usage_window == 64

    app.scale_usage_window(wider=False)
    assert threads.usage_window == 32


def test_a_recording_from_before_the_snapshot_sums_no_window(app):
    """Its shares are of each poll alone, so the bar must not claim a window."""
    app.scraper.groups["threads"].usage_window = 8

    app.scraper.parts = set()
    assert app.core.usage_window == 1

    app.scraper.parts = {"usage_snapshot"}
    assert app.core.usage_window == 8


def test_a_muted_row_stays_as_last_read_and_only_where_the_patterns_read(app):
    """A replay's frames still hold it, and the patterns decide whether it is listed."""

    def frame(count: int) -> Frame:
        gate, slots = (
            SemaphoreInfo("gate", 0x10, count, 4, ()),
            SemaphoreInfo("slots", 0x20, count, 4, ()),
        )
        return Frame({"semaphores": [gate, slots]})

    _open_objects(app)
    _feed(app, frame(1))
    app.toggle_mute("semaphores", "gate")
    _feed(app, frame(2))
    assert [(s.name, s.count) for s in app.rows("semaphores")] == [("slots", 2), ("gate", 1)]

    app.select(Selection(include=(("semaphores", "slots"),)))
    assert [s.name for s in app.rows("semaphores")] == ["slots"]


def test_a_muted_object_opens_on_its_last_read(app):
    from zview.frontend.tui.views.base import ZViewTUIAttributes
    from zview.frontend.tui.views.semaphore_detail import SemaphoreDetailView

    sem = SemaphoreInfo("gate", 0x10, 1, 4, ())
    _open_objects(app)
    _feed(app, Frame({"semaphores": [sem]}))
    app.toggle_mute("semaphores", "gate")
    _feed(app, Frame({"semaphores": []}))
    app.detailing = ("semaphores", 0x10)

    assert SemaphoreDetailView(app, ZViewTUIAttributes.create_mono())._target() == sem


def test_a_muted_threads_detail_reads_it_again_once_it_opens(app):
    """It would otherwise sit on the last read, frozen. A thread gone keeps its mute."""
    from zview.frontend.tui.views.base import ZViewTUIAttributes
    from zview.frontend.tui.views.thread_list import ThreadListView

    theme = ZViewTUIAttributes.create_mono()
    app.views = {
        ZViewState.THREAD_LIST_VIEW: ThreadListView(app, theme),
        ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(app, theme),
    }
    app.scraper.rows = {"threads": {"worker": [1]}}
    app.toggle_mute("threads", "worker")
    app.toggle_mute("threads", "gone")

    app.detailing_thread = "gone"
    app.transition_to(ZViewState.THREAD_DETAIL_VIEW)
    assert app.state is ZViewState.THREAD_LIST_VIEW
    assert ("threads", "gone") in app.muted

    app.detailing_thread = "worker"
    app.transition_to(ZViewState.THREAD_DETAIL_VIEW)
    assert app.state is ZViewState.THREAD_DETAIL_VIEW
    assert set(app.muted) == {("threads", "gone")}


def test_reload_onto_a_target_running_another_image_is_fatal(reloading_app):
    reloading_app.scraper.reattach.side_effect = ImageMismatch("the target is not running this ELF")

    reloading_app.attempt_reconnect()

    assert reloading_app.state is ZViewState.FATAL_ERROR
    assert "the target is not running this ELF" in reloading_app.status_message
    reloading_app.core.start.assert_not_called()


def test_escape_drops_the_pattern_line_and_enter_selects_it(app):
    """The second line opens on the patterns selected, so a dropped edit would show in it."""
    keys = [
        ord("/"),
        *map(ord, "idle"),
        SpecialCode.ESCAPE,
        ord("/"),
        SpecialCode.DELETE,
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


def test_a_pattern_line_naming_an_unknown_group_keeps_the_patterns(app):
    app.scraper.check_selection.side_effect = ValueError("Unknown group thread")

    app._select_patterns("thread:main")

    assert app.selection == Selection()
    assert app.messages[-1].text == "Error: Unknown group thread"
