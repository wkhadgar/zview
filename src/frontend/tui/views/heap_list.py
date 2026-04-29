# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import HeapInfo
from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
    compute_flex_widths,
)
from frontend.tui.widgets import TUIHeapInfo


class HeapListView(BaseStateView):
    SCHEMA = {
        "Heap": 25,
        "Free B": 7,
        "Used B": 7,
        "Heap Usage %": 27,
        "Watermark Bytes ": 15,
    }
    COLLUMS: list[str] = list(SCHEMA.keys())
    COLLUM_WIDTHS: list[int] = list(SCHEMA.values())

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._current_sort_idx = 0
        self._invert_sorting = False

        self._sort_keys = [
            lambda h: h.name,
            lambda h: h.free_bytes,
            lambda h: h.allocated_bytes,
            lambda h: h.max_allocated_bytes,
            lambda h: (
                (h.allocated_bytes / (h.allocated_bytes + h.free_bytes))
                if (h.allocated_bytes + h.free_bytes) > 0
                else 0
            ),
        ]

        self.top_line: int = 0

        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._tui_heap_info: TUIHeapInfo = TUIHeapInfo(theme.CURSOR, theme.ACTIVE, bar_theme)
        self._tui_heap_info.set_field_widths(
            self.COLLUM_WIDTHS[0],
            self.COLLUM_WIDTHS[1],
            self.COLLUM_WIDTHS[2],
            self.COLLUM_WIDTHS[3],
            self.COLLUM_WIDTHS[4],
        )

    def _compute_widths(self, terminal_width: int) -> list[int]:
        return compute_flex_widths(
            list(self.SCHEMA.values()), terminal_width, self.controller.heaps_data
        )

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draws the heap data table and its aggregate information.
        """
        stdscr.erase()

        self._render_frame(stdscr, self._footer_hint(), height, width)

        widths = self._compute_widths(width)
        self._tui_heap_info.set_field_widths(*widths)

        max_table_rows = height - 6
        total_heaps = len(self.controller.heaps_data)
        start_num = self.top_line + 1 if total_heaps > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_heaps)
        heap_column_width = widths[0]

        self.cursor = max(min(total_heaps - 1, self.cursor), 0)
        if self.cursor >= self.top_line + max_table_rows:
            self.top_line = self.cursor - max_table_rows + 1
        elif self.cursor < self.top_line:
            self.top_line = self.cursor

        order_symbol = " ▼" if self._invert_sorting else " ▲"
        sorting_header = self.COLLUMS[self._current_sort_idx]

        curr_x = 0
        for col_header, h_width in zip(self.SCHEMA.keys(), widths, strict=True):
            if curr_x >= width:
                break

            if col_header == sorting_header:
                col_header += order_symbol

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        scroll_indicator = f" Heaps: {start_num}-{end_num} of {total_heaps} "
        stdscr.addstr(height - 1, 0, scroll_indicator[:width], self._frame_attr)

        table_start = 4

        free_bytes_sum = sum(h.free_bytes for h in self.controller.heaps_data)
        allocated_bytes_sum = sum(h.allocated_bytes for h in self.controller.heaps_data)
        max_allocated_bytes_sum = sum(h.max_allocated_bytes for h in self.controller.heaps_data)
        total_heap_bytes = free_bytes_sum + allocated_bytes_sum

        aggregate_usage_pct = (
            (allocated_bytes_sum / total_heap_bytes * 100) if total_heap_bytes > 0 else 0.0
        )

        all_heaps_info = HeapInfo(
            name="All Heaps".center(heap_column_width - 1),
            address=0,
            free_bytes=free_bytes_sum,
            allocated_bytes=allocated_bytes_sum,
            max_allocated_bytes=max_allocated_bytes_sum,
            usage_percent=aggregate_usage_pct,
            chunks=None,
        )

        self._tui_heap_info.draw(stdscr, 2, 0, all_heaps_info, False)
        stdscr.hline(3, 0, curses.ACS_S3, width)

        key_func = self._sort_keys[self._current_sort_idx]
        sorted_heaps = sorted(
            self.controller.heaps_data, key=key_func, reverse=self._invert_sorting
        )

        for idx, heap in enumerate(sorted_heaps[self.top_line : self.top_line + max_table_rows]):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            self._tui_heap_info.draw(
                stdscr,
                target_y,
                0,
                heap,
                selected=(absolute_idx == self.cursor),
            )

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def keybindings(self) -> list[Keybind]:
        return [
            Keybind("<Enter>", "Detail", "Open detail view for the selected heap"),
            Keybind("h", "Threads", "Switch to the threads view"),
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
                if not self.controller.heaps_data:
                    return None

                key_func = self._sort_keys[self._current_sort_idx]
                sorted_heaps = sorted(
                    self.controller.heaps_data, key=key_func, reverse=self._invert_sorting
                )

                self.controller.detailing_heap_address = sorted_heaps[self.cursor].address
                return ZViewState.HEAPS_DETAIL_VIEW

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.HEAPS:
                return ZViewState.THREAD_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None
