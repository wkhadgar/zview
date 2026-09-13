# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Logic coverage for the kernel-object views (no curses rendering)."""

import curses
from unittest.mock import MagicMock, patch

import pytest

from backend.base import HeapInfo, MemSlabInfo, MsgqInfo, MutexInfo, MutexState, SemaphoreInfo
from frontend.tui.views.base import SpecialCode, ZViewState, ZViewTUIAttributes
from frontend.tui.views.kernel_object_list import (
    ALLOCATORS,
    HEAP,
    MEM_SLAB,
    MSGQ,
    MUTEX,
    SEMAPHORE,
    KernelObjectListView,
)
from frontend.tui.views.mem_slab_detail import MemSlabDetailView
from frontend.tui.views.msgq_detail import MsgqDetailView
from frontend.tui.views.mutex_detail import MutexDetailView
from frontend.tui.views.semaphore_detail import SemaphoreDetailView


@pytest.fixture
def theme() -> ZViewTUIAttributes:
    return ZViewTUIAttributes.create_mono()


@pytest.fixture
def distinct_theme() -> ZViewTUIAttributes:
    """Theme whose attributes differ, so color assertions mean something."""
    return ZViewTUIAttributes(*range(1, 11))


@pytest.fixture
def controller() -> MagicMock:
    c = MagicMock()
    c.scraper.has_semaphores = True
    c.scraper.has_mutexes = True
    c.scraper.has_msgqs = True
    c.scraper.has_mem_slabs = True
    c.waiters_unknown = False
    c.status_message = "Running"
    c.semaphores_data = [
        SemaphoreInfo(name="data_ready", address=0x2000, count=0, limit=4, waiters=("sensor",)),
        SemaphoreInfo(name="conn_pool", address=0x2100, count=3, limit=4, waiters=()),
    ]
    c.mutexes_data = [
        MutexInfo(name="uart_mutex", address=0x3000, lock_count=0, owner_address=0),
        MutexInfo(
            name="spi_bus",
            address=0x3100,
            lock_count=1,
            owner_address=0x4000,
            owner_name="spi_task",
            waiters=("sensor_task", "logger_task"),
        ),
    ]
    c.msgqs_data = [
        MsgqInfo(
            name="sensor_q",
            address=0x4000,
            used_msgs=2,
            max_msgs=8,
            msg_size=16,
            waiters=(),
        ),
    ]
    c.mem_slabs_data = [
        MemSlabInfo(
            name="frame_slab",
            address=0x5000,
            num_blocks=16,
            block_size=256,
            num_used=6,
            max_used=11,
            waiters=(),
        ),
    ]
    c.mutex_history = {}
    c.msgq_history = {}
    c.mem_slab_history = {}
    return c


def _filter_to(view: KernelObjectListView, step: str) -> None:
    """Press ``f`` until the view sits on one filter step."""
    for _ in range(len(view._FILTERS)):
        if view.filter_name == step:
            return
        view.handle_input(SpecialCode.FILTER)

    raise AssertionError(f"filter {step} is not in the cycle")


def test_rows_include_every_type(controller, theme):
    controller.heaps_data = [_heap()]
    view = KernelObjectListView(controller, theme)

    kinds = {kind for kind, _ in view._rows()}

    assert kinds == {SEMAPHORE, MUTEX, MSGQ, MEM_SLAB, HEAP}
    assert len(view._rows()) == 7


def test_filter_cycles_through_types(controller, theme):
    view = KernelObjectListView(controller, theme)
    assert view.filter_name == "ALL"

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == SEMAPHORE
    assert {kind for kind, _ in view._rows()} == {SEMAPHORE}

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == MUTEX
    assert {kind for kind, _ in view._rows()} == {MUTEX}

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == MSGQ
    assert {kind for kind, _ in view._rows()} == {MSGQ}

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == ALLOCATORS
    assert {kind for kind, _ in view._rows()} == {MEM_SLAB}

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == HEAP
    assert view._rows() == []

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == MEM_SLAB
    assert {kind for kind, _ in view._rows()} == {MEM_SLAB}

    view.handle_input(SpecialCode.FILTER)
    assert view.filter_name == "ALL"


def test_filter_change_resets_the_cursor(controller, theme):
    """Changing the filter resets cursor and scroll position."""
    view = KernelObjectListView(controller, theme)
    view.cursor = 3

    view.handle_input(SpecialCode.FILTER)

    assert view.cursor == 0
    assert view.top_line == 0


def test_semaphore_row_renders_count_and_waiters(controller, theme):
    view = KernelObjectListView(controller, theme)
    sem = controller.semaphores_data[0]

    fill, label, waiters, _ = view._info.row_values(SEMAPHORE, sem)

    assert fill == 0.0
    assert label == "0/4"
    assert waiters == "1 (sensor)"


def test_semaphore_row_bar_tracks_the_count(controller, theme):
    view = KernelObjectListView(controller, theme)
    full = SemaphoreInfo(name="s", address=0x1, count=4, limit=4, waiters=())
    half = SemaphoreInfo(name="s", address=0x1, count=2, limit=4, waiters=())
    empty = SemaphoreInfo(name="s", address=0x1, count=0, limit=4, waiters=())

    assert view._info.row_values(SEMAPHORE, full)[0] == 100.0
    assert view._info.row_values(SEMAPHORE, half)[0] == 50.0
    assert view._info.row_values(SEMAPHORE, empty)[0] == 0.0


def test_semaphore_row_survives_a_zero_limit(controller, theme):
    """A zero limit must not divide by zero."""
    view = KernelObjectListView(controller, theme)
    sem = SemaphoreInfo(name="s", address=0x1, count=0, limit=0, waiters=())

    assert view._info.row_values(SEMAPHORE, sem)[0] == 0.0
    assert view._info.row_values(SEMAPHORE, sem)[1] == "0/0"


def test_unknown_waiters_render_as_a_question_mark(controller, theme):
    """``waiters=None`` renders as ``?``, an empty queue as ``-``."""
    view = KernelObjectListView(controller, theme)
    unknown = SemaphoreInfo(name="s", address=0x1, count=1, limit=1, waiters=None)
    empty = SemaphoreInfo(name="s", address=0x1, count=1, limit=1, waiters=())

    assert view._info.row_values(SEMAPHORE, unknown)[2] == "?"
    assert view._info.row_values(SEMAPHORE, empty)[2] == "-"


def test_free_mutex_row(controller, theme):
    view = KernelObjectListView(controller, theme)

    fill, label, waiters, attr = view._info.row_values(MUTEX, controller.mutexes_data[0])

    assert fill == 0.0
    assert label == "FREE"
    # The fixture leaves waiters unset, i.e. the queue was not walked.
    assert waiters == "?"
    assert attr == 0


def test_free_mutex_with_an_empty_queue(controller, theme):
    view = KernelObjectListView(controller, theme)
    mutex = MutexInfo(name="m", address=0x1, lock_count=0, owner_address=0, waiters=())

    assert view._info.row_values(MUTEX, mutex)[2] == "-"


def test_contended_mutex_row_names_the_owner(controller, theme):
    view = KernelObjectListView(controller, theme)

    fill, label, waiters, _ = view._info.row_values(MUTEX, controller.mutexes_data[1])

    assert fill == 100.0
    assert label.startswith("LOCKED")
    assert "spi_task" in label
    assert waiters.startswith("2 (sensor_task, logger_task")


def test_locked_mutex_with_unknown_owner_shows_the_address(controller, theme):
    view = KernelObjectListView(controller, theme)
    mutex = MutexInfo(name="m", address=0x1, lock_count=1, owner_address=0x4000, waiters=())

    assert "0x4000" in view._info.row_values(MUTEX, mutex)[1]


def test_summary_counts_contention(controller, theme):
    view = KernelObjectListView(controller, theme)

    summary = view._summary(view._rows())

    assert "6 objects" in summary
    assert "1 contended" in summary
    # Every type counts here, and a contended mutex is one of them.
    assert "2 with waiters" in summary


def test_a_full_queue_counts_as_contended(controller, theme):
    controller.msgqs_data = [
        MsgqInfo(name="sensor_q", address=0x4000, used_msgs=8, max_msgs=8, msg_size=16, waiters=())
    ]
    view = KernelObjectListView(controller, theme)

    assert "2 contended" in view._summary(view._rows())


def test_an_exhausted_slab_counts_as_contended(controller, theme):
    controller.mem_slabs_data = [
        MemSlabInfo(
            name="frame_slab",
            address=0x5000,
            num_blocks=16,
            block_size=256,
            num_used=16,
            max_used=16,
            waiters=(),
        )
    ]
    view = KernelObjectListView(controller, theme)

    assert "2 contended" in view._summary(view._rows())


def test_summary_flags_unobservable_wait_queues(controller, theme):
    controller.waiters_unknown = True
    view = KernelObjectListView(controller, theme)

    assert "not observable" in view._summary(view._rows())


def test_empty_message_distinguishes_absent_from_filtered(controller, theme):
    view = KernelObjectListView(controller, theme)
    controller.semaphores_data = []
    controller.mutexes_data = []
    controller.msgqs_data = []
    controller.mem_slabs_data = []

    assert "No ALL objects." in view._empty_message()

    controller.scraper.has_semaphores = False
    controller.scraper.has_mutexes = False
    controller.scraper.has_msgqs = False
    controller.scraper.has_mem_slabs = False
    controller.scraper.has_heaps = False

    assert "No statically declared" in view._empty_message()


def test_enter_on_a_mutex_opens_the_detail_view(controller, theme):
    view = KernelObjectListView(controller, theme)
    view.handle_input(SpecialCode.FILTER)  # SEM
    view.handle_input(SpecialCode.FILTER)  # MTX
    view.cursor = 0

    target = view._rows()[0][1]
    state = view.handle_input(SpecialCode.NEWLINE)

    assert state == ZViewState.MUTEX_DETAIL_VIEW
    assert controller.detailing_mutex_address == target.address


def test_enter_on_a_semaphore_opens_its_detail_view(controller, theme):
    view = KernelObjectListView(controller, theme)
    view.handle_input(SpecialCode.FILTER)  # SEM
    view.cursor = 0

    target = view._rows()[0][1]
    state = view.handle_input(SpecialCode.NEWLINE)

    assert state == ZViewState.SEMAPHORE_DETAIL_VIEW
    assert controller.detailing_semaphore_address == target.address


def test_enter_on_an_empty_list_is_a_no_op(controller, theme):
    controller.semaphores_data = []
    controller.mutexes_data = []
    controller.msgqs_data = []
    controller.mem_slabs_data = []
    view = KernelObjectListView(controller, theme)

    assert view.handle_input(SpecialCode.NEWLINE) is None


def test_k_returns_to_the_thread_list(controller, theme):
    view = KernelObjectListView(controller, theme)

    assert view.handle_input(SpecialCode.KERNEL_OBJECTS) == ZViewState.THREAD_LIST_VIEW


def test_quit_stops_the_controller(controller, theme):
    view = KernelObjectListView(controller, theme)

    view.handle_input(SpecialCode.QUIT)

    assert controller.running is False


def test_footer_lists_the_view_bindings(controller, theme):
    view = KernelObjectListView(controller, theme)
    parts = [p.strip() for p in view._footer_hint().rstrip().split("|")]

    assert parts[0] == "Help: ?"
    assert "Detail: <Enter>" in parts
    assert parts[-1] == "…"


def test_mutex_detail_finds_its_target(controller, theme):
    controller.detailing_mutex_address = 0x3100
    view = MutexDetailView(controller, theme)

    assert view._target().name == "spi_bus"


def test_mutex_detail_target_absent(controller, theme):
    """A target absent from the current frame resolves to None."""
    controller.detailing_mutex_address = 0xDEAD
    view = MutexDetailView(controller, theme)

    assert view._target() is None

    controller.detailing_mutex_address = None
    assert view._target() is None


@pytest.mark.parametrize(
    "key",
    (curses.KEY_LEFT, SpecialCode.KERNEL_OBJECTS, SpecialCode.NEWLINE, 27),
)
def test_mutex_detail_returns_to_the_list(controller, theme, key):
    view = MutexDetailView(controller, theme)

    assert view.handle_input(key) == ZViewState.KERNEL_OBJECT_LIST_VIEW


def test_columns_size_to_their_content(controller, theme):
    """Name and metric columns grow to fit the longest value they hold."""
    long_name = "log_process_thread_semaphore_x"
    controller.mutexes_data = [
        MutexInfo(name=long_name, address=0x3000, lock_count=0, owner_address=0)
    ]
    view = KernelObjectListView(controller, theme)
    schema = list(KernelObjectListView.SCHEMA.values())

    widths = view.compute_widths(200, [obj.name for _, obj in view._rows()])

    assert widths[0] == schema[0]
    assert widths[3] == schema[3]
    assert widths[1] == len(long_name) + 1
    # The bar column absorbs the spare width.
    assert sum(widths) + len(widths) - 1 == 199


def test_short_names_keep_the_schema_width(controller, theme):
    view = KernelObjectListView(controller, theme)

    widths = view.compute_widths(200, [obj.name for _, obj in view._rows()])

    assert widths[1] == KernelObjectListView.SCHEMA["Name"]


def test_name_column_is_capped(controller, theme):
    controller.mutexes_data = [
        MutexInfo(name="m" * 80, address=0x3000, lock_count=0, owner_address=0)
    ]
    view = KernelObjectListView(controller, theme)

    widths = view.compute_widths(300, [obj.name for _, obj in view._rows()])

    assert widths[1] == KernelObjectListView._MAX_NAME


def test_columns_shrink_to_fit_the_minimum_terminal(controller, theme):
    """At the TUI's 85-column minimum the row fits, and no column goes below schema."""
    controller.mutexes_data = [
        MutexInfo(
            name="a_long_mutex_name_here",
            address=0x3000,
            lock_count=4,
            owner_address=0x4000,
            owner_name="a_long_owner_thread_name",
            waiters=("waiter_one", "waiter_two"),
        )
    ]
    view = KernelObjectListView(controller, theme)
    schema = list(KernelObjectListView.SCHEMA.values())

    names = [obj.name for _, obj in view._rows()]
    wide = view.compute_widths(200, names)
    narrow = view.compute_widths(85, names)

    assert all(w >= f for w, f in zip(narrow, schema, strict=True))
    # Only the bar column changes with the terminal width.
    assert narrow[2] < wide[2]
    assert narrow[:2] == wide[:2]
    assert narrow[3] == wide[3]


@pytest.mark.parametrize(
    ("presses", "expected_column"),
    [(0, 0), (1, 1), (2, 3), (3, 2), (4, 0)],
)
def test_sort_indicator_tracks_the_sorted_column(controller, theme, presses, expected_column):
    """The arrow marks the column the active sort key orders by."""
    view = KernelObjectListView(controller, theme)
    for _ in range(presses):
        view.handle_input(SpecialCode.SORT)

    assert view._sort_columns[view._current_sort_idx] == expected_column


def test_the_last_sort_key_orders_every_type_by_its_bar(controller, theme):
    """One key for how full a thing is, whatever the thing counts."""
    controller.heaps_data = [_heap(free=0, used=4096)]
    view = KernelObjectListView(controller, theme)
    for _ in range(3):
        view.handle_input(SpecialCode.SORT)

    fills = [view._info.row_values(kind, obj)[0] for kind, obj in view._rows()]

    assert fills == sorted(fills)
    assert fills[-1] == pytest.approx(100.0)


def test_a_wide_terminal_puts_the_queue_beside_the_graph(controller, theme):
    view = SemaphoreDetailView(controller, theme)

    panels = view._panels(height=24, width=200, top=6)

    assert panels.graph_w == 200 - view._QUEUE_WIDTH
    assert panels.queue_x == panels.graph_w
    assert panels.queue_w == view._QUEUE_WIDTH
    # Both take every row left, which is what the graph wants and what fits names.
    assert panels.graph_h == panels.queue_h == 24 - 2 - 6
    assert panels.queue_y == 6


def test_a_short_terminal_still_leaves_both_panels_a_border(controller, theme):
    """Three rows is the least a box can be drawn in."""
    view = SemaphoreDetailView(controller, theme)

    panels = view._panels(height=10, width=60, top=6)

    assert panels.graph_h >= 3
    assert panels.queue_h >= 3


class _StubWin:
    """curses window stand-in recording ``(row, col, text)`` writes."""

    def __init__(self, height: int = 24, width: int = 209):
        self._h, self._w = height, width
        self.writes: list[tuple[int, int, str]] = []
        self.styled: list[tuple[int, int, str, int]] = []

    def getmaxyx(self):
        return self._h, self._w

    def addstr(self, y: int, x: int, text: str, attr: int = 0):
        assert 0 <= y < self._h, f"row {y} outside height {self._h}"
        assert 0 <= x < self._w, f"col {x} outside width {self._w}"
        assert x + len(text) <= self._w, f"row {y} overruns width: {x}+{len(text)} > {self._w}"
        if y == self._h - 1 and x + len(text) == self._w:
            raise curses.error("addwstr() returned ERR")
        self.writes.append((y, x, text))
        self.styled.append((y, x, text, attr))

    def erase(self): ...
    def refresh(self): ...
    def attron(self, attr): ...
    def attroff(self, attr): ...
    def move(self, y, x): ...
    def clrtoeol(self): ...
    def hline(self, y: int, x: int, ch: int, n: int):
        assert 0 <= y < self._h, f"row {y} outside height {self._h}"
        assert x + n <= self._w, f"row {y} overruns width: {x}+{n} > {self._w}"

    def getbkgd(self):
        return 0


def _row_columns(win: _StubWin, row: int) -> list[int]:
    return [x for y, x, _ in win.writes if y == row]


def _free_lock() -> MutexInfo:
    return MutexInfo(name="bench_lock", address=0x3000, lock_count=0, owner_address=0)


def _held_lock() -> MutexInfo:
    return MutexInfo(
        name="bench_lock",
        address=0x3000,
        lock_count=1,
        owner_address=0x4000,
        owner_name="lock_owner_id",
        waiters=("lock_waiter_id",),
    )


def test_columns_hold_position_when_a_row_changes_state(controller, theme):
    """A mutex flipping FREE/LOCKED must not shift the table sideways."""
    if not hasattr(curses, "ACS_S3"):  # only defined after curses init
        curses.ACS_S3 = ord("-")

    view = KernelObjectListView(controller, theme)
    controller.semaphores_data = []

    controller.mutexes_data = [_free_lock()]
    free_win = _StubWin()
    view.render(free_win, 24, 209)

    controller.mutexes_data = [_held_lock()]
    held_win = _StubWin()
    view.render(held_win, 24, 209)

    free_cols = _row_columns(free_win, 4)
    held_cols = _row_columns(held_win, 4)

    # Type, name and the bar all start at the same column, and so does the
    # waiters column on the right; only the bar's centered caption moves with
    # the text it holds.
    assert free_cols[:4] == held_cols[:4]
    assert free_cols[-1] == held_cols[-1]
    assert _row_columns(free_win, 1) == _row_columns(held_win, 1)


def test_table_fills_a_wide_terminal(controller, theme):
    """The metric column absorbs the spare width instead of leaving a gap."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    view = KernelObjectListView(controller, theme)
    win = _StubWin(width=209)
    view.render(win, 24, 209)

    widest = max(x + len(text) for y, x, text in win.writes if y >= 4)
    assert widest > 150


def test_row_color_reaches_the_bar(controller, distinct_theme):
    """A contended mutex colors its bar, not just its cells."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    view = KernelObjectListView(controller, distinct_theme)
    controller.semaphores_data = []
    controller.msgqs_data = []
    controller.mem_slabs_data = []
    controller.mutexes_data = [_held_lock()]

    drawn: list[int | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        drawn.append(attr)
    )

    view.render(_StubWin(), 24, 209)

    # Aggregate row first, then the contended mutex row.
    assert drawn[-1] == view._info._contended_attr


def test_cursor_does_not_recolor_the_bar(controller, distinct_theme):
    """Selection marks the name, never the bar: its track must stay readable."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    view = KernelObjectListView(controller, distinct_theme)
    controller.semaphores_data = []
    controller.msgqs_data = []
    controller.mem_slabs_data = []
    controller.mutexes_data = [_held_lock()]
    view.cursor = 0

    drawn: list[int | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        drawn.append(attr)
    )

    view.render(_StubWin(), 24, 209)

    assert view._info._selected_attr not in drawn
    assert drawn[-1] == view._info._contended_attr


def test_default_sort_groups_by_type_then_name(controller, theme):
    """Rows open grouped by type, alphabetical inside each group."""
    view = KernelObjectListView(controller, theme)

    assert [(kind, obj.name) for kind, obj in view._rows()] == [
        (MSGQ, "sensor_q"),
        (MUTEX, "spi_bus"),
        (MUTEX, "uart_mutex"),
        (SEMAPHORE, "conn_pool"),
        (SEMAPHORE, "data_ready"),
        (MEM_SLAB, "frame_slab"),
    ]


def _lock_with(waiters: tuple[str, ...]) -> MutexInfo:
    return MutexInfo(
        name="bench_lock",
        address=0x3000,
        lock_count=1,
        owner_address=0x4000,
        owner_name="lock_owner_id",
        waiters=waiters,
    )


def test_detail_layout_holds_as_the_wait_queue_grows(controller, theme):
    """The strip and the state lines stay put however many threads are queued."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mutex_address = 0x3000
    controller.mutex_history = {0x3000: [MutexState.CONTENDED, MutexState.FREE, MutexState.LOCKED]}
    view = MutexDetailView(controller, theme)

    rows_by_size = []
    for size in (0, 1, 5):
        controller.mutexes_data = [_lock_with(tuple(f"waiter_{i}" for i in range(size)))]
        win = _StubWin()
        view.render(win, 24, 209)
        # Row of every line that is not part of the wait queue section.
        rows_by_size.append([y for y, _, _ in win.writes if y < MutexDetailView._QUEUE_ROW])

    assert rows_by_size[0] == rows_by_size[1] == rows_by_size[2]


def test_contention_strip_colors_contended_frames(controller, distinct_theme):
    """Contended frames are red in the strip, held-alone frames busy, free plain."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mutex_address = 0x3000
    controller.mutexes_data = [_lock_with(("lock_waiter_id",))]
    controller.mutex_history = {
        0x3000: [
            MutexState.FREE,
            MutexState.FREE,
            MutexState.CONTENDED,
            MutexState.CONTENDED,
            MutexState.LOCKED,
            MutexState.LOCKED,
        ]
    }
    view = MutexDetailView(controller, distinct_theme)

    win = _StubWin()
    view.render(win, 24, 209)

    strip = [
        (text, attr)
        for y, _, text, attr in win.styled
        if y == MutexDetailView._HISTORY_ROW + 1 and text.strip()
    ]
    assert (view.CONTENDED_MARK * 2, view._contended_attr) in strip
    assert (view.LOCKED_MARK * 2, view._busy_attr) in strip
    assert (view.FREE_MARK * 2, 0) in strip


def test_count_graph_is_colored_while_threads_wait(controller, distinct_theme):
    """The count graph turns busy-colored only when the queue is non-empty."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_semaphore_address = 0x2000
    controller.sem_history = {0x2000: [0, 1, 2]}
    view = SemaphoreDetailView(controller, distinct_theme)

    queued = SemaphoreInfo(name="s", address=0x2000, count=0, limit=4, waiters=("taker",))
    idle = SemaphoreInfo(name="s", address=0x2000, count=4, limit=4, waiters=())

    drawn: list[int] = []
    view._draw_count_graph = lambda stdscr, width, sem, graph_height: drawn.append(
        view._busy_attr if sem.waiters else 0
    )

    for sem in (queued, idle):
        controller.semaphores_data = [sem]
        view.render(_StubWin(), 24, 209)

    assert drawn == [view._busy_attr, 0]


def test_count_graph_plots_one_column_per_sample(controller, theme):
    """More points than columns would let the graph average and reshape them."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    width = 60
    controller.detailing_semaphore_address = 0x2000
    controller.semaphores_data = [
        SemaphoreInfo(name="s", address=0x2000, count=2, limit=4, waiters=())
    ]
    controller.sem_history = {0x2000: list(range(500))}
    view = SemaphoreDetailView(controller, theme)

    plotted: list[list[int]] = []
    with patch(
        "frontend.tui.views.semaphore_detail.TUIGraph.draw",
        lambda self, stdscr, y, x, h, w, **kwargs: plotted.append(kwargs["points"]),
    ):
        view.render(_StubWin(width=width), 24, width)

    assert len(plotted[0]) == width - 2
    # The tail of the history, in order, not a resampling of all of it.
    assert plotted[0] == list(range(500))[-(width - 2) :]


def test_history_legend_marks_match_the_strip_colors(controller, distinct_theme):
    """The legend's marks are the key to the strip, so they carry the same colors."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mutex_address = 0x3000
    controller.mutexes_data = [_lock_with(("lock_waiter_id",))]
    controller.mutex_history = {0x3000: [MutexState.CONTENDED]}
    view = MutexDetailView(controller, distinct_theme)

    win = _StubWin()
    view.render(win, 24, 209)

    legend_row = MutexDetailView._HISTORY_ROW + MutexDetailView._HISTORY_HEIGHT - 1
    for state, mark_x in view.legend_mark_columns().items():
        assert (legend_row, mark_x, view.MARKS[state], view._mark_attrs[state]) in win.styled


@pytest.mark.parametrize("queued", (0, 1, 40, 400))
@pytest.mark.parametrize(("height", "width"), ((14, 85), (24, 100), (57, 209)))
def test_views_hold_a_waiter_burst_at_any_terminal_size(controller, theme, queued, height, width):
    """A queue longer than its panel is truncated, never drawn past the frame."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    waiters = tuple(f"waiter_number_{i}" for i in range(queued))
    controller.detailing_mutex_address = 0x3000
    controller.detailing_semaphore_address = 0x2000
    controller.mutexes_data = [_lock_with(waiters)]
    controller.semaphores_data = [
        SemaphoreInfo(name="bench_slots", address=0x2000, count=2, limit=4, waiters=waiters)
    ]
    controller.mutex_history = {0x3000: [MutexState.CONTENDED] * 500}
    controller.sem_history = {0x2000: list(range(500))}

    for view_type in (MutexDetailView, SemaphoreDetailView, KernelObjectListView):
        win = _StubWin(height=height, width=width)
        view_type(controller, theme).render(win, height, width)

        assert win.writes, f"{view_type.__name__} drew nothing at {width}x{height}"
        assert max(y for y, _, _ in win.writes) < height


def test_wait_queue_truncation_covers_the_row_it_replaces(controller, theme):
    """The `... N more` line lands on a waiter row, so it is padded over it."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mutex_address = 0x3000
    controller.mutexes_data = [_lock_with(tuple(f"waiter_number_{i}" for i in range(40)))]
    controller.mutex_history = {}
    view = MutexDetailView(controller, theme)

    win = _StubWin(height=20, width=85)
    view.render(win, 20, 85)

    more = next(text for _, _, text in win.writes if text.startswith("... "))
    assert len(more) == 85 - 4


@pytest.mark.parametrize(
    ("presses", "expected_type", "expected_label"),
    ((0, "ALL", "All Objects"), (1, SEMAPHORE, "All Semaphores"), (2, MUTEX, "All Mutexes")),
)
def test_the_aggregate_row_is_labelled_by_the_filter(
    controller, theme, presses, expected_type, expected_label
):
    """The row sums the filtered objects, so it must not claim to sum them all."""
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    view = KernelObjectListView(controller, theme)
    for _ in range(presses):
        view.handle_input(SpecialCode.FILTER)

    win = _StubWin()
    view.render(win, 24, 209)

    row = [text for y, _, text in win.writes if y == 2]
    assert expected_type in row[0]
    assert expected_label in row[1]


def test_msgq_row_shows_fill_and_message_size(controller, theme):
    view = KernelObjectListView(controller, theme)

    fill, label, waiters, attr = view._info.row_values(MSGQ, controller.msgqs_data[0])

    assert fill == 25.0
    assert label == "2/8 × 16B"
    assert waiters == "-"
    assert attr == 0


def test_a_full_queue_reads_as_contention(controller, distinct_theme):
    """A full queue blocks its senders, so it carries the contended color."""
    view = KernelObjectListView(controller, distinct_theme)
    full = MsgqInfo(name="q", address=0x1, used_msgs=4, max_msgs=4, msg_size=8, waiters=())
    queued = MsgqInfo(name="q", address=0x1, used_msgs=0, max_msgs=4, msg_size=8, waiters=("rx",))
    idle = MsgqInfo(name="q", address=0x1, used_msgs=1, max_msgs=4, msg_size=8, waiters=())

    assert view._info.row_values(MSGQ, full)[3] == distinct_theme.ERROR
    assert view._info.row_values(MSGQ, queued)[3] == distinct_theme.PROGRESS_BAR_MEDIUM
    assert view._info.row_values(MSGQ, idle)[3] == 0


def test_a_zero_capacity_queue_row_does_not_divide_by_zero(controller, theme):
    view = KernelObjectListView(controller, theme)
    msgq = MsgqInfo(name="q", address=0x1, used_msgs=0, max_msgs=0, msg_size=4, waiters=())

    assert view._info.row_values(MSGQ, msgq)[0] == 0.0


def test_enter_on_a_queue_opens_its_detail_view(controller, theme):
    view = KernelObjectListView(controller, theme)
    view.handle_input(SpecialCode.FILTER)  # SEM
    view.handle_input(SpecialCode.FILTER)  # MTX
    view.handle_input(SpecialCode.FILTER)  # MSG
    view.cursor = 0

    target = view._rows()[0][1]
    state = view.handle_input(SpecialCode.NEWLINE)

    assert state == ZViewState.MSGQ_DETAIL_VIEW
    assert controller.detailing_msgq_address == target.address


def test_the_queue_detail_plots_its_depth_against_capacity(controller, distinct_theme):
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_msgq_address = 0x4000
    controller.msgq_history = {0x4000: [0, 1, 2, 3]}
    view = MsgqDetailView(controller, distinct_theme)

    win = _StubWin()
    view.render(win, 24, 209)

    drawn = " ".join(text for _, _, text in win.writes)
    assert "0 to 8 messages" in drawn
    assert "16 B" in drawn
    assert "0x4000" in drawn


def test_the_queue_detail_says_when_the_queue_is_gone(controller, theme):
    controller.detailing_msgq_address = 0x9999
    view = MsgqDetailView(controller, theme)

    win = _StubWin()
    view.render(win, 24, 209)

    assert any("no longer being reported" in text for _, _, text in win.writes)


def test_mem_slab_row_shows_usage_block_size_and_peak(controller, theme):
    view = KernelObjectListView(controller, theme)

    fill, label, waiters, attr = view._info.row_values(MEM_SLAB, controller.mem_slabs_data[0])

    assert fill == 37.5
    assert label == "6/16 × 256B"
    assert waiters == "-"
    assert attr == 0


def test_the_peak_is_marked_on_the_bar(controller, theme):
    """The peak is a position on the track, not a number beside the label."""
    view = KernelObjectListView(controller, theme)
    view._info.set_field_widths(5, 24, 22, 30)

    marks: list[float | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        marks.append(mark)
    )

    win = _StubWin()
    view._info.draw(win, 4, 0, MEM_SLAB, controller.mem_slabs_data[0])

    # 11 of 16 blocks at its peak.
    assert marks == [pytest.approx(68.75)]


def test_a_slab_that_never_allocated_gets_no_mark(controller, theme):
    """A peak of nothing sits at the track's start, where it reads as an artifact."""
    view = KernelObjectListView(controller, theme)
    view._info.set_field_widths(5, 24, 22, 30)
    untouched = MemSlabInfo(
        name="pool", address=0x1, num_blocks=4, block_size=44, num_used=0, max_used=0, waiters=()
    )

    marks: list[float | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        marks.append(mark)
    )

    view._info.draw(_StubWin(), 4, 0, MEM_SLAB, untouched)

    assert marks == [None]


def test_a_slab_without_a_tracked_peak_gets_no_mark(controller, theme):
    view = KernelObjectListView(controller, theme)
    view._info.set_field_widths(5, 24, 22, 30)
    slab = MemSlabInfo(name="s", address=0x1, num_blocks=8, block_size=64, num_used=2, waiters=())

    marks: list[float | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        marks.append(mark)
    )

    view._info.draw(_StubWin(), 4, 0, MEM_SLAB, slab)

    assert marks == [None]
    assert view._info.row_values(MEM_SLAB, slab)[1] == "2/8 × 64B"


def test_an_exhausted_slab_reads_as_contention(controller, distinct_theme):
    """Every block is out, so the next allocation blocks."""
    view = KernelObjectListView(controller, distinct_theme)
    out = MemSlabInfo(name="s", address=0x1, num_blocks=4, block_size=32, num_used=4, waiters=())
    waited = MemSlabInfo(
        name="s", address=0x1, num_blocks=4, block_size=32, num_used=2, waiters=("worker",)
    )
    idle = MemSlabInfo(name="s", address=0x1, num_blocks=4, block_size=32, num_used=1, waiters=())

    assert view._info.row_values(MEM_SLAB, out)[3] == distinct_theme.ERROR
    assert view._info.row_values(MEM_SLAB, waited)[3] == distinct_theme.PROGRESS_BAR_MEDIUM
    assert view._info.row_values(MEM_SLAB, idle)[3] == 0


def test_enter_on_a_slab_opens_its_detail_view(controller, theme):
    view = KernelObjectListView(controller, theme)
    for _ in range(4):
        view.handle_input(SpecialCode.FILTER)  # SEM, MTX, MSG, SLB
    view.cursor = 0

    target = view._rows()[0][1]
    state = view.handle_input(SpecialCode.NEWLINE)

    assert state == ZViewState.MEM_SLAB_DETAIL_VIEW
    assert controller.detailing_mem_slab_address == target.address


def test_the_slab_detail_plots_usage_and_names_the_peak(controller, distinct_theme):
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mem_slab_address = 0x5000
    controller.mem_slab_history = {0x5000: [1, 3, 6]}
    view = MemSlabDetailView(controller, distinct_theme)

    win = _StubWin()
    view.render(win, 24, 209)

    drawn = " ".join(text for _, _, text in win.writes)
    assert "0 to 16 blocks, peak 11" in drawn
    assert "11/16 blocks" in drawn
    assert "256 B (4096 B total)" in drawn


def test_the_slab_detail_says_when_the_peak_is_not_tracked(controller, distinct_theme):
    if not hasattr(curses, "ACS_S3"):
        curses.ACS_S3 = ord("-")

    controller.detailing_mem_slab_address = 0x5000
    controller.mem_slabs_data = [
        MemSlabInfo(name="frame_slab", address=0x5000, num_blocks=16, block_size=256, num_used=6)
    ]
    controller.mem_slab_history = {}
    view = MemSlabDetailView(controller, distinct_theme)

    win = _StubWin()
    view.render(win, 24, 209)

    drawn = " ".join(text for _, _, text in win.writes)
    assert MemSlabDetailView._UNTRACKED in drawn
    assert "peak" not in drawn


def _heap(
    name: str = "bench_heap",
    *,
    free: int = 1536,
    used: int = 2560,
    peak: int = 2816,
    waiters: tuple[str, ...] = (),
) -> HeapInfo:
    total = free + used
    return HeapInfo(
        name=name,
        address=0x6000,
        free_bytes=free,
        allocated_bytes=used,
        max_allocated_bytes=peak,
        usage_percent=(used / total * 100.0) if total else 0.0,
        chunks=None,
        waiters=waiters,
    )


def test_the_allocator_filter_shows_heaps_and_slabs(controller, theme):
    controller.heaps_data = [_heap()]
    view = KernelObjectListView(controller, theme)

    _filter_to(view, ALLOCATORS)

    assert {kind for kind, _ in view._rows()} == {HEAP, MEM_SLAB}
    assert view._AGGREGATE_LABELS[ALLOCATORS] == "All Allocators"


def test_an_allocator_filter_counts_exhaustion_not_contention(controller, theme):
    controller.heaps_data = [_heap(free=0)]
    view = KernelObjectListView(controller, theme)
    _filter_to(view, ALLOCATORS)

    summary = view._summary(view._rows())

    assert "1 exhausted" in summary
    assert "contended" not in summary


def test_a_heap_row_reads_bytes_out_over_its_size(controller, theme):
    view = KernelObjectListView(controller, theme)

    fill, label, waiters, attr = view._info.row_values(HEAP, _heap())

    assert label == "2560/4096B"
    assert fill == pytest.approx(62.5)
    assert waiters == "-"
    assert attr == 0


def test_a_heap_with_nothing_free_carries_the_contended_color(controller, theme):
    view = KernelObjectListView(controller, theme)

    _, _, _, attr = view._info.row_values(HEAP, _heap(free=0, used=4096))

    assert attr == view._info._contended_attr


def test_a_heap_with_a_thread_waiting_for_memory_reads_busy(controller, theme):
    view = KernelObjectListView(controller, theme)

    _, _, waiters, attr = view._info.row_values(HEAP, _heap(waiters=("worker",)))

    assert waiters.startswith("1")
    assert attr == view._info._busy_attr


def test_a_heap_carries_no_peak_mark(controller, theme):
    """``max_allocated_bytes`` peaks the allocated bytes; the bar is not drawn on that axis."""
    view = KernelObjectListView(controller, theme)
    view._info.set_field_widths(5, 24, 22, 30)

    marks: list[float | None] = []
    view._info.state_bar.draw = lambda stdscr, y, x, pct, label=None, attr=None, mark=None: (
        marks.append(mark)
    )

    view._info.draw(_StubWin(), 4, 0, HEAP, _heap())

    assert marks == [None]


def test_enter_on_a_heap_opens_its_own_view(controller, theme):
    controller.heaps_data = [_heap()]
    view = KernelObjectListView(controller, theme)
    _filter_to(view, HEAP)

    state = view.handle_input(SpecialCode.NEWLINE)

    assert state is ZViewState.HEAPS_DETAIL_VIEW
    assert controller.detailing_heap_address == 0x6000


def _render(view, height: int = 24, width: int = 209) -> _StubWin:
    """Render one frame into a stub window, with the theme's real attributes."""
    win = _StubWin(height, width)
    view.render(win, height, width)
    return win


def test_the_queue_is_drawn_in_its_own_column_beside_the_graph(controller, theme):
    controller.detailing_semaphore_address = 0x2000
    controller.sem_history = {0x2000: [0, 1, 2]}
    view = SemaphoreDetailView(controller, theme)

    win = _render(view)

    queue_col = 209 - view._QUEUE_WIDTH
    titles = [(y, x) for y, x, text in win.writes if text.startswith("┌Wait queue")]
    assert titles and all(x == queue_col for _, x in titles)
    # The waiter list sits in that column, not under the graph.
    assert any(x == queue_col + 2 and "sensor" in text for _, x, text in win.writes)


def test_a_narrow_terminal_draws_the_queue_under_the_graph(controller, theme):
    controller.detailing_semaphore_address = 0x2000
    controller.sem_history = {0x2000: [0, 1, 2]}
    view = SemaphoreDetailView(controller, theme)

    win = _render(view, height=24, width=70)

    titles = [(y, x) for y, x, text in win.writes if text.startswith("┌Wait queue")]
    assert titles and all(x == 0 for _, x in titles)
    assert titles[0][0] > view._GRAPH_ROW
