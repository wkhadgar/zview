# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses
from dataclasses import replace

from zview.backend.base import ThreadInfo, ThreadRuntime
from zview.frontend.tui.views.base import (
    DETAIL_KEYS,
    INVERT_KEYS,
    MUTE_KEYS,
    SORT_KEYS,
    USAGE_WINDOW_KEYS,
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
    compute_flex_widths,
)
from zview.frontend.tui.widgets import TUIThreadInfo
from zview.kinds import BY_KEY
from zview.selection import Selection


def _runtime_key(field: str):
    """Sort key on a runtime field, with an unknown value at ``-1``."""
    return lambda t: -1 if (value := getattr(t.runtime, field, None)) is None else value


def _total(values: list) -> float | None:
    """The threads' values summed, or None when the build does not measure them."""
    return None if None in values else sum(values)


def listed_threads(controller: Any) -> list[ThreadInfo]:
    """The threads the thread kind lists, as the selection shows them."""
    return [t for t in controller.rows("threads") if BY_KEY["threads"].listed(t)]


class ThreadListView(BaseStateView):
    SCHEMA: dict[str, int] = {
        "Thread": 25,
        "CPU %": 7,
        "Load %": 7,
        "Stack Usage % (Watermark)": 27,
        "Watermark Bytes": 15,
    }
    COLLUMS: list[str] = list(SCHEMA.keys())
    COLLUM_WIDTHS: list[int] = list(SCHEMA.values())
    THREADS = Selection(groups=frozenset({"threads"}))

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._current_sort_idx = 0
        self._invert_sorting = False
        self._sort_keys = [
            lambda t: t.name,
            _runtime_key("cpu"),
            _runtime_key("cpu_normalized"),
            _runtime_key("stack_watermark_percent"),
            _runtime_key("stack_watermark"),
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

    def reads(self, patterns: Selection) -> Selection:
        groups = (
            self.THREADS.groups
            if patterns.groups is None
            else patterns.groups & self.THREADS.groups
        )
        return replace(patterns, groups=groups)

    def _compute_widths(self, terminal_width: int, threads: list[ThreadInfo]) -> list[int]:
        return compute_flex_widths(list(self.SCHEMA.values()), terminal_width, threads)

    def _sorted_threads(self) -> list[ThreadInfo]:
        """The listed threads in the table's order."""
        key_func = self._sort_keys[self._current_sort_idx]
        return sorted(listed_threads(self.controller), key=key_func, reverse=self._invert_sorting)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draws the thread data table and its general informations.
        """

        stdscr.erase()

        hint = self._footer_hint()
        self._render_frame(stdscr, hint, height, width)

        threads = self._sorted_threads()
        muted = self.controller.muted
        widths = self._compute_widths(width, threads)
        self._tui_thread_info.set_field_widths(*widths)

        max_table_rows = height - 6
        total_threads = len(threads)
        start_num = self.top_line + 1 if total_threads > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_threads)
        thread_column_width = widths[0]

        self.cursor = max(min(total_threads - 1, self.cursor), 0)
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

            self._draw_header(stdscr, curr_x, col_header, h_width, width)
            curr_x += h_width + 1

        scroll_indicator = self._bar_count(
            f"Threads: {start_num}-{end_num} of {total_threads}", width - len(hint)
        )
        stdscr.addstr(height - 1, 0, scroll_indicator, self._frame_attr)

        table_start = 4

        # A muted thread's last values would add stale numbers to the totals.
        read = [t for t in threads if ("threads", t.name) not in muted]
        runtimes = [t.runtime for t in read if t.runtime]
        stack_size_sum = _total([t.stack_size for t in read])
        stack_watermark_sum = _total([r.stack_watermark for r in runtimes])
        is_any_thread_active = any(r.active for r in runtimes)
        aggregate_stack_usage_pct = None
        if stack_watermark_sum is not None:
            aggregate_stack_usage_pct = (
                (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
            )
        aggregate_load = _total([r.cpu for r in runtimes])
        aggregate_cpu = _total([r.cpu_normalized for r in runtimes])

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

        for idx, thread in enumerate(threads[self.top_line : self.top_line + max_table_rows]):
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
                muted=("threads", thread.name) in muted,
            )

        if not threads:
            no_threads_strs = (
                "No threads found on kernel.",
                "If you had any, they are probably dead.",
                "You may try to refresh (r) the thread list.",
            )
            if not self.controller.selection.reads_group("threads"):
                no_threads_strs = ("Threads are not read.", "/ edits the patterns.")
            for i, _str in enumerate(no_threads_strs):
                stdscr.addstr(height // 2 + i, width // 2 - len(_str) // 2, _str)

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def keybindings(self) -> list[Keybind]:
        bindings = [
            DETAIL_KEYS,
            Keybind("r", "Refresh", "Re-walk the kernel thread list and resync baselines"),
            SORT_KEYS,
            INVERT_KEYS,
            MUTE_KEYS,
            USAGE_WINDOW_KEYS,
        ]
        # The footer shows the first three bindings only, so the view switches
        # come ahead of Refresh.
        if self.controller.has_kernel_objects():
            bindings[1:1] = [Keybind("k", "Kernel objects", "Switch to the kernel objects view")]
        return bindings

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case curses.KEY_DOWN:
                self.cursor += 1
            case curses.KEY_UP:
                self.cursor -= 1
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                threads = self._sorted_threads()
                if not threads:
                    return None

                self.controller.detailing_thread = threads[self.cursor].name

                return ZViewState.THREAD_DETAIL_VIEW

            case SpecialCode.MUTE:
                if threads := self._sorted_threads():
                    self.controller.toggle_mute("threads", threads[self.cursor].name)

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.WIDER | SpecialCode.NARROWER:
                self.controller.scale_usage_window(key == SpecialCode.WIDER)

            case SpecialCode.REFRESH:
                if not self.controller.scraper.is_live:
                    self.controller.report("Refresh is not available in replay mode.")
                    return None

                self.controller.report("Refreshing thread list...")

                try:
                    self.controller.core.refresh()
                    self.controller.purge_queue()
                except Exception as e:
                    self.controller.report(f"Error refreshing threads: {e}")

                return None

            case SpecialCode.KERNEL_OBJECTS:
                if not self.controller.has_kernel_objects():
                    return None

                return ZViewState.KERNEL_OBJECT_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None
