# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import ThreadInfo, decode_thread_state
from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
    compute_flex_widths,
)
from frontend.tui.views.thread_list import ThreadListView
from frontend.tui.widgets import TUIGraph, TUIThreadInfo


def _format_thread_meta(thread: ThreadInfo) -> str:
    """One-line shell-style metadata: ``prio | state | options | entry``."""
    parts: list[str] = []
    if thread.priority is not None:
        parts.append(f"prio: {thread.priority}")
    state_label = decode_thread_state(thread.state)
    if state_label is not None:
        parts.append(f"state: {state_label}")
    if thread.user_options is not None:
        parts.append(f"options: 0x{thread.user_options:02x}")
    if thread.entry_point:
        if thread.entry_symbol:
            parts.append(f"entry: {thread.entry_symbol} (0x{thread.entry_point:08x})")
        else:
            parts.append(f"entry: 0x{thread.entry_point:08x}")
    return " | ".join(parts)


class ThreadDetailView(BaseStateView):
    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)
        self._cpu_graph: TUIGraph = TUIGraph(
            "CPU %", "Thread cycles / Cycles", (0, 100), theme.GRAPH_B
        )
        self._load_graph: TUIGraph = TUIGraph(
            "Load %", "Thread cycles / Non-idle cycles", (0, 100), theme.GRAPH_A
        )

        self._scheme: dict[str, int] = ThreadListView.SCHEMA
        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._tui_thread_info: TUIThreadInfo = TUIThreadInfo(
            theme.CURSOR, theme.ACTIVE, theme.INACTIVE, bar_theme
        )
        self._tui_thread_info.set_field_widths(
            ThreadListView.COLLUM_WIDTHS[0],
            ThreadListView.COLLUM_WIDTHS[1],
            ThreadListView.COLLUM_WIDTHS[2],
            ThreadListView.COLLUM_WIDTHS[3],
            ThreadListView.COLLUM_WIDTHS[4],
        )
        self._current_thread_name: str | None = None
        self._usages: dict[str, list[int]] = {"cpu": [], "load": []}

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()

        self._render_frame(stdscr, self._footer_hint(), height, width)

        widths = compute_flex_widths(
            list(self._scheme.values()), width, self.controller.threads_data
        )
        self._tui_thread_info.set_field_widths(*widths)

        curr_x = 0
        for col_header, h_width in zip(self._scheme.keys(), widths, strict=True):
            if curr_x >= width:
                break

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        thread = next(
            (t for t in self.controller.threads_data if t.name == self.controller.detailing_thread),
            None,
        )
        if not thread or thread.runtime is None:
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        # Reset history if switching to a new thread
        if self._current_thread_name != thread.name:
            self._current_thread_name = thread.name
            self._usages = {"cpu": [], "load": []}

        self._tui_thread_info.draw(stdscr, 2, 0, thread)

        meta_line = _format_thread_meta(thread)
        if meta_line:
            stdscr.addstr(3, 0, meta_line[: width - 1])

        self._usages["cpu"].append(int(thread.runtime.cpu_normalized))
        self._usages["load"].append(int(thread.runtime.cpu))

        graph_top = 5 if meta_line else 4
        graph_height = max(self.controller.min_dimensions[0] - 6, height - 3 - graph_top)
        graph_width = width // 2

        if len(self._usages["load"]) > graph_width - 2:
            self._usages["load"].pop(0)
            self._usages["cpu"].pop(0)

        self._cpu_graph.draw(
            stdscr,
            graph_top,
            0,
            graph_height,
            graph_width,
            points=self._usages["cpu"],
        )

        self._load_graph.draw(
            stdscr,
            graph_top,
            graph_width,
            graph_height,
            graph_width,
            points=self._usages["load"],
        )

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def keybindings(self) -> list[Keybind]:
        return [Keybind("<Enter>", "Back", "Return to the thread list")]

    def handle_input(self, key: int) -> ZViewState | None:
        if key in (curses.KEY_ENTER, SpecialCode.NEWLINE, SpecialCode.RETURN):
            return ZViewState.THREAD_LIST_VIEW
        elif key == SpecialCode.QUIT:
            self.controller.running = False
        return None
