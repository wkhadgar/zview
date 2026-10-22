# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from zview.frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from zview.frontend.tui.widgets import TUIKernelObjectInfo
from zview.kinds import KINDS
from zview.kinds.kind import Kind, Level
from zview.selection import Selection


class KernelObjectListView(BaseStateView):
    """
    List of the semaphores, mutexes, message queues, memory slabs, heaps and
    work queues found in the ELF.

    Columns are type, name, a state bar and the wait queue. The bar shows
    count over limit for a semaphore, held or free for a mutex, used over
    capacity for a queue, blocks out over the block count for a slab, bytes
    out over the size for a heap, and busy or idle for a work queue, captioned
    with the count, the owner, the size or the backlog. ``f`` filters by type,
    ``Enter`` opens the detail view for the selected object.
    """

    # ``State`` is the bar column and absorbs the spare width.
    SCHEMA = {"Type": 5, "Name": 24, "State": 26, "Waiters": 30}
    COLUMNS: list[str] = list(SCHEMA.keys())
    _BAR_COLUMN = 2

    # Every group but the threads, which the thread views show.
    OBJECTS = Selection(skipped=frozenset({"threads"}))
    # The aggregate row sums the filtered rows, so it is titled by the filter.
    _ALL_TITLE = "All Objects"
    _MAX_NAME = 34
    _MAX_LISTED_WAITERS = 3
    _UNKNOWN = "?"

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self.top_line: int = 0
        # The kinds this view lists, and its filters: all of them, then each.
        self._kinds = tuple(kind for kind in KINDS if self.OBJECTS.reads_group(kind.key))
        self._filters: tuple[Kind | None, ...] = (None, *self._kinds)
        self._filter_idx: int = 0
        self._current_sort_idx: int = 0
        self._invert_sorting: bool = False

        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._info = TUIKernelObjectInfo(
            theme.CURSOR, theme.ERROR, theme.PROGRESS_BAR_MEDIUM, bar_theme
        )

        # Type first by default, with the name as tiebreak so each group is
        # alphabetical.
        self._sort_keys = [
            lambda row: (row[0].label, row[1].name),
            lambda row: row[1].name,
            lambda row: len(row[1].waiters or ()),
            # Whatever its bar counts: a count, a depth, blocks or bytes.
            lambda row: self._info.row_values(row[0], row[1])[0],
        ]
        # Column each sort key orders by, for the header indicator.
        self._sort_columns = (0, 1, 3, 2)

    def reads(self) -> Selection:
        return self.OBJECTS

    @property
    def _filter(self) -> Kind | None:
        """The kind the table is filtered to, or ``None`` for all of them."""
        return self._filters[self._filter_idx]

    @property
    def filter_name(self) -> str:
        return "ALL" if self._filter is None else self._filter.label

    def _rows(self) -> list[tuple[Kind, Any]]:
        """Typed, filtered and sorted rows backing the table."""
        rows = [
            (kind, obj)
            for kind in self._kinds
            if self._filter in (None, kind)
            for obj in self.controller.frames.latest(kind.key)
        ]

        return sorted(
            rows, key=self._sort_keys[self._current_sort_idx], reverse=self._invert_sorting
        )

    @classmethod
    def compute_widths(cls, terminal_width: int, names: list[str]) -> list[int]:
        """
        Column widths: name sized to the object names, the bar takes the rest.

        Only the names and the terminal width feed this, never a value that
        changes between polls, so the columns hold still. A classmethod
        because the detail views lay out the same columns for their title row.
        """
        widths = list(cls.SCHEMA.values())

        if names:
            widths[1] = max(widths[1], min(cls._MAX_NAME, max(len(n) for n in names) + 1))

        gaps = len(widths) - 1
        spare = terminal_width - 1 - gaps - widths[0] - widths[1] - widths[3]
        widths[cls._BAR_COLUMN] = max(widths[cls._BAR_COLUMN], spare)

        return widths

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        pairs = self._rows()
        widths = self.compute_widths(width, [obj.name for _, obj in pairs])
        self._info.set_field_widths(*widths)

        curr_x = 0
        order_symbol = " ▼" if self._invert_sorting else " ▲"
        sorted_column = self._sort_columns[self._current_sort_idx]
        for column, (col_header, col_width) in enumerate(zip(self.COLUMNS, widths, strict=True)):
            if curr_x >= width:
                break
            header = col_header + (order_symbol if column == sorted_column else "")
            stdscr.addstr(1, curr_x, f"{header:^{col_width}}"[: width - curr_x])
            curr_x += col_width + 1

        self._info.draw_cells(
            stdscr,
            2,
            0,
            self.filter_name,
            f"{self._ALL_TITLE if self._filter is None else self._filter.title:^{widths[1]}}",
            *self._aggregate(pairs),
            0,
        )
        stdscr.hline(3, 0, curses.ACS_S3, width)

        table_start = 4
        max_table_rows = max(0, height - 6)
        total = len(pairs)

        self.cursor = max(min(total - 1, self.cursor), 0)
        if self.cursor >= self.top_line + max_table_rows:
            self.top_line = self.cursor - max_table_rows + 1
        elif self.cursor < self.top_line:
            self.top_line = self.cursor

        if not pairs:
            stdscr.addstr(table_start, 0, self._empty_message()[:width])

        visible = pairs[self.top_line : self.top_line + max_table_rows]
        for idx, (kind, obj) in enumerate(visible):
            target_y = table_start + idx
            if target_y >= height - 2:
                break

            selected = (self.top_line + idx) == self.cursor
            self._info.draw(stdscr, target_y, 0, kind, obj, selected)

        start_num = self.top_line + 1 if total else 0
        end_num = min(self.top_line + max_table_rows, total)
        indicator = f" {self.filter_name} objects: {start_num}-{end_num} of {total} "
        stdscr.addstr(height - 1, 0, indicator[:width], self._frame_attr)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _aggregate(self, rows: list[tuple[Kind, Any]]) -> tuple[float, str, str]:
        """``(bar percentage, bar label, waiters cell)`` for the aggregate row."""
        pressured = sum(
            1
            for kind, obj in rows
            if obj.waiters or kind.in_use(obj) or self._is_blocked(kind, obj)
        )
        fill = (pressured / len(rows) * 100.0) if rows else 0.0

        walked = [obj.waiters for _, obj in rows if obj.waiters is not None]
        queued = sum(len(w) for w in walked)
        cell = str(queued) if len(walked) == len(rows) else f"{queued}+?"

        return fill, self._summary(rows).strip(), cell

    @staticmethod
    def _is_blocked(kind: Kind, obj: Any) -> bool:
        return kind.state(obj).level is Level.BLOCKED

    def _summary(self, rows: list[tuple[Kind, Any]]) -> str:
        contended = sum(1 for kind, obj in rows if self._is_blocked(kind, obj))
        queued = sum(1 for _, obj in rows if obj.waiters)

        word = "contended" if self._filter is None else self._filter.pressure
        summary = f" {len(rows)} objects | {contended} {word} | {queued} with waiters"
        if self.controller.waiters_unknown:
            # Stated once here instead of per row.
            summary += " | wait queues not observable (scalable waitq)"

        return summary

    def _empty_message(self) -> str:
        if not self.controller.has_kernel_objects():
            return " No statically declared kernel objects in this build."

        return f" No {self.filter_name} objects."

    def keybindings(self) -> list[Keybind]:
        return [
            Keybind("<Enter>", "Detail", "Open detail view for the selected object"),
            Keybind(
                "f",
                "Filter",
                f"Cycle the type filter ({', '.join(['ALL', *(k.label for k in self._kinds)])})",
            ),
            Keybind("k", "Threads", "Switch back to the threads view"),
            Keybind("s", "Sort", "Cycle through sort keys"),
            Keybind("i", "Invert", "Reverse the current sort order"),
        ]

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case curses.KEY_DOWN:
                self.cursor += 1
            case curses.KEY_UP:
                self.cursor -= 1
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                rows = self._rows()
                if not rows:
                    return None

                kind, obj = rows[max(min(self.cursor, len(rows) - 1), 0)]
                self.controller.detailing = (kind.key, obj.address)
                return ZViewState.OBJECT_DETAIL_VIEW

            case SpecialCode.FILTER:
                self._filter_idx = (self._filter_idx + 1) % len(self._filters)
                self.cursor = 0
                self.top_line = 0

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.KERNEL_OBJECTS:
                return ZViewState.THREAD_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None

        return None
