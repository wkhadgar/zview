# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import ThreadInfo
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
from frontend.tui.widgets import TUIBox, TUIGraph, TUIThreadInfo


def _format_priority(thread: ThreadInfo) -> str:
    return str(thread.priority) if thread.priority is not None else "-"


def _format_options(thread: ThreadInfo) -> str:
    return f"0x{thread.user_options:02x}" if thread.user_options is not None else "-"


def _format_entry(thread: ThreadInfo) -> str:
    if not thread.entry_point:
        return "-"
    if thread.entry_symbol:
        return f"{thread.entry_symbol} (0x{thread.entry_point:08x})"
    return f"0x{thread.entry_point:08x}"


class ThreadDetailView(BaseStateView):
    _INFO_BOX_HEIGHT = 3

    # ``(title, weight, formatter)``: weight is the relative horizontal share
    # used when laying the boxes side-by-side; entry needs the most room.
    _INFO_BOXES: tuple[tuple[str, int, Any], ...] = (
        ("Priority", 1, _format_priority),
        ("Options", 1, _format_options),
        ("Entry", 3, _format_entry),
    )

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)
        self._cpu_graph: TUIGraph = TUIGraph(
            "CPU %", "Thread cycles / Cycles", (0, 100), theme.GRAPH_B
        )
        self._load_graph: TUIGraph = TUIGraph(
            "Load %", "Thread cycles / Non-idle cycles", (0, 100), theme.GRAPH_A
        )
        self._info_widgets: list[TUIBox] = [
            TUIBox(title, "", theme.GRAPH_B) for title, _, _ in self._INFO_BOXES
        ]

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

    def _draw_info_boxes(
        self, stdscr: curses.window, y: int, width: int, thread: ThreadInfo
    ) -> None:
        """Lay the identity boxes side-by-side, sized by their relative weights."""
        weights = [w for _, w, _ in self._INFO_BOXES]
        total_weight = sum(weights)
        x = 0
        for idx, ((title, weight, fmt), box) in enumerate(
            zip(self._INFO_BOXES, self._info_widgets, strict=True)
        ):
            del title  # unused here; widgets carry their own titles
            # The last box takes any rounding remainder so the row ends flush at ``width``.
            is_last = idx == len(self._INFO_BOXES) - 1
            box_w = (width - x) if is_last else (width * weight) // total_weight
            box.draw(stdscr, y, x, self._INFO_BOX_HEIGHT, box_w)
            value = fmt(thread)
            inner_w = box_w - 4  # 2 border cells + 2 padding cells
            if inner_w > 0:
                stdscr.addstr(y + 1, x + 2, value.ljust(inner_w)[:inner_w])
            x += box_w

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

        info_top = 3
        self._draw_info_boxes(stdscr, info_top, width, thread)

        self._usages["cpu"].append(int(thread.runtime.cpu_normalized))
        self._usages["load"].append(int(thread.runtime.cpu))

        graph_top = info_top + self._INFO_BOX_HEIGHT
        graph_height = max(
            self.controller.min_dimensions[0] - 5 - self._INFO_BOX_HEIGHT, height - 3 - graph_top
        )
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
