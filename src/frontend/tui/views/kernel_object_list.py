# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.widgets import MUTEX, SEMAPHORE, TUIKernelObjectInfo

__all__ = ["MUTEX", "SEMAPHORE", "KernelObjectListView"]


class KernelObjectListView(BaseStateView):
    """
    List of the semaphores and mutexes found in the ELF.

    Columns are type, name, a state bar and the wait queue. The bar shows
    count over limit for a semaphore and held or free for a mutex, captioned
    with the count or the owner. ``f`` filters by type, ``Enter`` opens the
    detail view for the selected object.
    """

    # ``State`` is the bar column and absorbs the spare width, mirroring the
    # stack and heap usage bars in the thread and heap views.
    SCHEMA = {"Type": 5, "Name": 24, "State": 26, "Waiters": 30}
    COLUMNS: list[str] = list(SCHEMA.keys())
    _BAR_COLUMN = 2

    _FILTERS = ("ALL", SEMAPHORE, MUTEX)
    _MAX_NAME = 34
    _MAX_LISTED_WAITERS = 3
    _UNKNOWN = "?"

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self.top_line: int = 0
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
            lambda row: (row[0], row[1].name),
            lambda row: row[1].name,
            lambda row: len(row[1].waiters or ()),
        ]
        # Column each sort key orders by, for the header indicator.
        self._sort_columns = (0, 1, 3)

    @property
    def filter_name(self) -> str:
        return self._FILTERS[self._filter_idx]

    def _rows(self) -> list[tuple[str, Any]]:
        """Typed, filtered and sorted rows backing the table."""
        rows: list[tuple[str, Any]] = []
        if self.filter_name in ("ALL", SEMAPHORE):
            rows += [(SEMAPHORE, sem) for sem in self.controller.semaphores_data]
        if self.filter_name in ("ALL", MUTEX):
            rows += [(MUTEX, mutex) for mutex in self.controller.mutexes_data]

        return sorted(
            rows, key=self._sort_keys[self._current_sort_idx], reverse=self._invert_sorting
        )

    @classmethod
    def compute_widths(cls, terminal_width: int, names: list[str]) -> list[int]:
        """
        Column widths: name sized to the object names, the bar takes the rest.

        Only the names and the terminal width feed this, never a value that
        changes between polls, so a row flipping state cannot shift the table
        sideways. A classmethod because the detail views lay out the same
        columns for their title row.
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
            "ALL",
            f"{'All Objects':^{widths[1]}}",
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

    def _aggregate(self, rows: list[tuple[str, Any]]) -> tuple[float, str, str]:
        """``(bar percentage, bar label, waiters cell)`` for the aggregate row."""
        pressured = sum(1 for kind, obj in rows if obj.waiters or (kind == MUTEX and obj.is_locked))
        fill = (pressured / len(rows) * 100.0) if rows else 0.0

        walked = [obj.waiters for _, obj in rows if obj.waiters is not None]
        queued = sum(len(w) for w in walked)
        cell = str(queued) if len(walked) == len(rows) else f"{queued}+?"

        return fill, self._summary(rows).strip(), cell

    def _summary(self, rows: list[tuple[str, Any]]) -> str:
        contended = sum(1 for kind, obj in rows if kind == MUTEX and obj.is_locked and obj.waiters)
        queued = sum(1 for kind, obj in rows if kind == SEMAPHORE and obj.waiters)

        summary = f" {len(rows)} objects | {contended} contended | {queued} with waiters"
        if self.controller.waiters_unknown:
            # Stated once here instead of per row.
            summary += " | wait queues not observable (scalable waitq)"

        return summary

    def _empty_message(self) -> str:
        if not (self.controller.scraper.has_semaphores or self.controller.scraper.has_mutexes):
            return " No statically declared semaphores or mutexes in this build."

        return f" No {self.filter_name} objects."

    def keybindings(self) -> list[Keybind]:
        return [
            Keybind("<Enter>", "Detail", "Open detail view for the selected object"),
            Keybind("f", "Filter", "Cycle the type filter (ALL, SEM, MTX)"),
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
                if kind == MUTEX:
                    self.controller.detailing_mutex_address = obj.address
                    return ZViewState.MUTEX_DETAIL_VIEW

                self.controller.detailing_semaphore_address = obj.address
                return ZViewState.SEMAPHORE_DETAIL_VIEW

            case SpecialCode.FILTER:
                self._filter_idx = (self._filter_idx + 1) % len(self._FILTERS)
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
