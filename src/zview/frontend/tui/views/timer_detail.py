# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from zview.backend.base import TimerInfo
from zview.frontend.tui.views.base import Any, ZViewTUIAttributes
from zview.frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from zview.frontend.tui.widgets import SiteValue, TUIBox, TUIGraph
from zview.kinds import BY_KEY


def _function_cell(
    pointer: int | None, symbol: str | None, site: tuple[str, int] | None
) -> str | SiteValue:
    """A callback with the source site the ELF gives for it, or ``-`` for none."""
    if pointer is None:
        return "-"
    return SiteValue(site, pointer, symbol)


class TimerDetailView(KernelObjectDetailView):
    """
    One timer: its period and callbacks, a running strip, its unread expiries
    over time, and the thread waiting on it.

    The strip carries one mark per polled frame, as the mutex's does: running
    or stopped. A one-shot that runs and stops between two frames leaves no mark.
    """

    kind = BY_KEY["timers"]
    RUNNING_MARK = "●"
    STOPPED_MARK = "·"

    # A callback's site needs the room a thread's entry has.
    _INFO_TITLES = (("Location", 2), ("Period", 1), ("Expiry fn", 2), ("Stop fn", 2))
    _STRIP_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT
    _STRIP_HEIGHT = 3
    _GRAPH_ROW = _STRIP_ROW + _STRIP_HEIGHT

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)
        legend = f"one mark per frame: {self.RUNNING_MARK} running  {self.STOPPED_MARK} stopped"
        self._strip_box = TUIBox("Running history", legend, 0)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        timer = self._target()
        if timer is None:
            stdscr.addstr(1, 0, " Timer is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, timer)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [
                self._site(timer),
                "oneshot" if timer.is_oneshot else f"{timer.period_ticks} ticks",
                _function_cell(timer.expiry_fn, timer.expiry_fn_symbol, timer.expiry_fn_site),
                _function_cell(timer.stop_fn, timer.stop_fn_symbol, timer.stop_fn_site),
            ],
        )

        history = self.controller.frames.of("timers", timer.address)
        self._draw_strip(stdscr, width, history)
        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_unread_graph(stdscr, panels.graph_w, panels.graph_h, history)
        self._draw_wait_queue(stdscr, panels, timer.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_strip(self, stdscr: curses.window, width: int, history: list[TimerInfo]) -> None:
        self._strip_box.draw(stdscr, self._STRIP_ROW, 0, self._STRIP_HEIGHT, width)
        inner_w = width - 4
        if not history or inner_w <= 0:
            return

        marks = "".join(
            self.RUNNING_MARK if info.is_running else self.STOPPED_MARK for info in history
        )
        stdscr.addstr(self._STRIP_ROW + 1, 2, marks[-inner_w:])

    def _draw_unread_graph(
        self, stdscr: curses.window, width: int, graph_height: int, history: list[TimerInfo]
    ) -> None:
        # One column per sample: extra points are averaged into moving buckets.
        unread = [info.status for info in history][-max(1, width - 2) :]
        # The scale is the highest count drawn, as an unlimited semaphore's is.
        top = max(unread, default=0) or 1
        graph = TUIGraph("Unread expiries", "since a status read", (0, top), 0)
        graph.draw(stdscr, self._GRAPH_ROW, 0, graph_height, width, points=unread)
