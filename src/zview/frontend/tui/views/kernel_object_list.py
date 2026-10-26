# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from zview.frontend.tui.views.base import (
    DETAIL_KEYS,
    INVERT_KEYS,
    MUTE_KEYS,
    SORT_KEYS,
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
    with the count, the owner, the size or the backlog. ``Enter`` opens the
    detail view for the selected object.
    """

    # ``State`` is the bar column and absorbs the spare width.
    SCHEMA = {"Type": 5, "Name": 24, "State": 26, "Waiters": 30}
    COLUMNS: list[str] = list(SCHEMA.keys())
    _BAR_COLUMN = 2

    # Every group but the threads, which the thread views show.
    OBJECTS = Selection(skipped=frozenset({"threads"}))
    # The aggregate row sums the rows listed, so a single kind listed titles it.
    _ALL_TITLE = "All Objects"
    _MAX_NAME = 34
    _MAX_LISTED_WAITERS = 3
    _UNKNOWN = "?"

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self.top_line: int = 0
        # The kinds this view lists.
        self._kinds = tuple(kind for kind in KINDS if self.OBJECTS.reads_group(kind.key))
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

    def _only_kind(self) -> Kind | None:
        """The kind listed while it is the only one read and available, or ``None``."""
        listed = [
            kind
            for kind in self._kinds
            if self.controller.scraper.available.get(kind.key)
            and self.controller.selection.reads_group(kind.key)
        ]
        return listed[0] if len(listed) == 1 else None

    def _rows(self) -> list[tuple[Kind, Any]]:
        """Typed, filtered and sorted rows backing the table."""
        rows = [
            (kind, obj)
            for kind in self._kinds
            for obj in self.controller.frames.latest(kind.key, self.controller.selection)
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
        hint = self._footer_hint()
        self._render_frame(stdscr, hint, height, width)

        pairs = self._rows()
        only = self._only_kind()
        name = "ALL" if only is None else only.label
        muted = self.controller.selection.muted
        # A muted object's last values would add stale numbers to the aggregate.
        read = [(kind, obj) for kind, obj in pairs if (kind.key, obj.name) not in muted]
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
            name,
            f"{self._ALL_TITLE if only is None else only.title:^{widths[1]}}",
            *self._aggregate(read, only),
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
            stdscr.addstr(table_start, 0, self._empty_message(name)[:width])

        visible = pairs[self.top_line : self.top_line + max_table_rows]
        for idx, (kind, obj) in enumerate(visible):
            target_y = table_start + idx
            if target_y >= height - 2:
                break

            selected = (self.top_line + idx) == self.cursor
            self._info.draw(
                stdscr, target_y, 0, kind, obj, selected, muted=(kind.key, obj.name) in muted
            )

        start_num = self.top_line + 1 if total else 0
        end_num = min(self.top_line + max_table_rows, total)
        indicator = self._bar_count(
            f"{name} objects: {start_num}-{end_num} of {total}", width - len(hint)
        )
        stdscr.addstr(height - 1, 0, indicator, self._frame_attr)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _aggregate(self, rows: list[tuple[Kind, Any]], only: Kind | None) -> tuple[float, str, str]:
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

        return fill, self._summary(rows, only).strip(), cell

    @staticmethod
    def _is_blocked(kind: Kind, obj: Any) -> bool:
        state = kind.state(obj)
        return state is not None and state.level is Level.BLOCKED

    def _summary(self, rows: list[tuple[Kind, Any]], only: Kind | None) -> str:
        contended = sum(1 for kind, obj in rows if self._is_blocked(kind, obj))
        queued = sum(1 for _, obj in rows if obj.waiters)

        word = "contended" if only is None else only.pressure
        summary = f" {len(rows)} objects | {contended} {word} | {queued} with waiters"
        if self.controller.waiters_unknown:
            # Stated once here instead of per row.
            summary += " | wait queues not observable (scalable waitq)"

        return summary

    def _empty_message(self, name: str) -> str:
        if not self.controller.has_kernel_objects():
            return " No kernel object group is read, / edits the patterns."

        return f" No {name} objects."

    def keybindings(self) -> list[Keybind]:
        return [
            DETAIL_KEYS,
            Keybind("k", "Threads", "Switch back to the threads view"),
            SORT_KEYS,
            INVERT_KEYS,
            MUTE_KEYS,
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

            case SpecialCode.MUTE:
                if rows := self._rows():
                    kind, obj = rows[max(min(self.cursor, len(rows) - 1), 0)]
                    self.controller.toggle_mute(kind.key, obj.name)

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
