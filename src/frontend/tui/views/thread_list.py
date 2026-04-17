# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.z_scraper import ThreadInfo, ThreadRuntime
from frontend.tui.views.base import Any, BaseStateView, SpecialCode, ZViewState, ZViewTUIAttributes
from frontend.tui.widgets import TUIThreadInfo


class ThreadListView(BaseStateView):
    SCHEMA: dict[str, int] = {
        "Thread": 30,
        "CPU %": 8,
        "Load %": 8,
        "Stack Usage % (Watermark)": 32,
        "Watermark Bytes": 18,
    }
    COLLUMS: list[str] = list(SCHEMA.keys())
    COLLUM_WIDTHS: list[int] = list(SCHEMA.values())

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._current_sort_idx = 0
        self._invert_sorting = False
        self._sort_keys = [
            lambda t: t.name,
            lambda t: t.runtime.cpu if t.runtime else -1,
            lambda t: t.runtime.cpu_normalized if t.runtime else -1,
            lambda t: t.runtime.stack_watermark_percent if t.runtime else -1,
            lambda t: t.runtime.stack_watermark if t.runtime else -1,
        ]

        self.top_line: int = 0

        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._tui_thread_info: TUIThreadInfo = TUIThreadInfo(
            theme.CURSOR, theme.ACTIVE, theme.INACTIVE, bar_theme
        )
        self._tui_thread_info.set_field_widths(
            self.COLLUM_WIDTHS[0],
            self.COLLUM_WIDTHS[1],
            self.COLLUM_WIDTHS[2],
            self.COLLUM_WIDTHS[3],
            self.COLLUM_WIDTHS[4],
        )

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draws the thread data table and its general informations.
        """

        stdscr.erase()

        self._render_frame(
            stdscr,
            "Quit: q | Sort: s | Invert: i | Refresh: r | Details: <Enter> "
            + ("| Heaps: h " if self.controller.scraper.has_heaps else ""),
            height,
            width,
        )

        max_table_rows = height - 6
        total_threads = len(self.controller.threads_data)
        start_num = self.top_line + 1 if total_threads > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_threads)
        thread_column_width = self.COLLUM_WIDTHS[0]

        self.cursor = max(min(total_threads - 1, self.cursor), 0)
        if self.cursor >= self.top_line + max_table_rows:
            self.top_line = self.cursor - max_table_rows + 1
        elif self.cursor < self.top_line:
            self.top_line = self.cursor

        order_symbol = " ▼" if self._invert_sorting else " ▲"
        sorting_header = self.COLLUMS[self._current_sort_idx]

        curr_x = 0
        for col_header, h_width in self.SCHEMA.items():
            if curr_x >= width:
                break

            if col_header == sorting_header:
                col_header += order_symbol

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        scroll_indicator = f" Threads: {start_num}-{end_num} of {total_threads} "
        stdscr.addstr(height - 1, 0, scroll_indicator[:width], self._frame_attr)

        table_start = 4

        stack_size_sum = sum(t.stack_size for t in self.controller.threads_data)
        stack_watermark_sum = sum(
            t.runtime.stack_watermark if t.runtime else 0 for t in self.controller.threads_data
        )
        is_any_thread_active = any(
            t.runtime.active if t.runtime else False for t in self.controller.threads_data
        )
        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )
        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )
        aggregate_load = sum(
            t.runtime.cpu for t in self.controller.threads_data if t.runtime and t.runtime.cpu > 0
        )
        aggregate_cpu = sum(
            t.runtime.cpu_normalized
            for t in self.controller.threads_data
            if t.runtime and t.runtime.cpu_normalized > 0
        )

        all_threads_info = ThreadInfo(
            address=0,
            stack_start=0,
            stack_size=stack_size_sum,
            name="All Threads".center(thread_column_width - 1),
            runtime=ThreadRuntime(
                cpu=aggregate_load,
                cpu_normalized=aggregate_cpu,
                active=is_any_thread_active,
                stack_watermark=stack_watermark_sum,
                stack_watermark_percent=aggregate_stack_usage_pct,
            ),
        )

        self._tui_thread_info.draw(stdscr, 2, 0, all_threads_info, False)
        stdscr.addstr(3, 0, "─" * width)

        key_func = self._sort_keys[self._current_sort_idx]

        sorted_threads = sorted(
            self.controller.threads_data, key=key_func, reverse=self._invert_sorting
        )

        for idx, thread in enumerate(
            sorted_threads[self.top_line : self.top_line + max_table_rows]
        ):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            self._tui_thread_info.draw(
                stdscr,
                target_y,
                0,
                thread,
                selected=(absolute_idx == self.cursor),
            )

        if not sorted_threads:
            no_threads_strs = (
                "No threads found on kernel.",
                "If you had any, they are probably dead.",
                "You may try to refresh (r) the thread list.",
            )
            for i, _str in enumerate(no_threads_strs):
                stdscr.addstr(height // 2 + i, width // 2 - len(_str) // 2, _str)

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case curses.KEY_DOWN:
                self.cursor += 1
            case curses.KEY_UP:
                self.cursor -= 1
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                if not self.controller.threads_data:
                    return None

                key_func = self._sort_keys[self._current_sort_idx]
                sorted_threads = sorted(
                    self.controller.threads_data, key=key_func, reverse=self._invert_sorting
                )

                self.controller.detailing_thread = sorted_threads[self.cursor].name

                return ZViewState.THREAD_DETAIL_VIEW

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.RECONNECT:
                self.controller.status_message = "Refreshing thread list..."

                # Force the scraper to re-read the kernel's thread linked-list
                try:
                    self.controller.scraper.update_available_threads()
                    self.controller.scraper.reset_thread_pool()
                    self.controller.scraper.reset_runtime_state()
                    self.controller.purge_queue()
                except Exception as e:
                    self.controller.status_message = f"Error refreshing threads: {e}"

                return None  # Stay in the list view

            case SpecialCode.HEAPS:
                if not self.controller.scraper.has_heaps:
                    return None

                return ZViewState.HEAP_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None
