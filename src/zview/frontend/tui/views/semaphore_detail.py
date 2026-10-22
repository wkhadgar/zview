# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from zview.backend.base import SemaphoreInfo
from zview.frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from zview.frontend.tui.widgets import TUIGraph
from zview.kinds import BY_KEY


class SemaphoreDetailView(KernelObjectDetailView):
    """
    One semaphore: count, limit, waiter count, a count graph and the wait queue.

    The graph plots one sample per polled frame. A take and a give between two
    frames cancel out and leave no trace.
    """

    kind = BY_KEY["semaphores"]
    # The title row already carries count over limit and the waiters, so the
    # boxes add what it cannot show.
    _INFO_TITLES = (("Location", 2), ("Limit", 1))
    _GRAPH_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        sem = self._target()
        if sem is None:
            stdscr.addstr(1, 0, " Semaphore is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, sem)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [self._site(sem.address), self._limit_text(sem)],
        )

        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_count_graph(stdscr, panels.graph_w, sem, panels.graph_h)
        self._draw_wait_queue(stdscr, panels, sem.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    @staticmethod
    def _limit_text(sem: SemaphoreInfo) -> str:
        # First match wins.
        return next(
            text
            for applies, text in (
                (not sem.is_initialized, "not initialized"),
                (sem.is_unbounded, "unlimited"),
                (True, str(sem.limit)),
            )
            if applies
        )

    def _draw_count_graph(
        self, stdscr: curses.window, width: int, sem: SemaphoreInfo, graph_height: int
    ) -> None:
        # Yellow while threads are queued on it, plain otherwise.
        attr = self._busy_attr if sem.waiters else 0

        # One column per sample: extra points are averaged into moving buckets.
        columns = max(1, width - 2)
        history = [s.count for s in self.controller.frames.of("semaphores", sem.address)][-columns:]

        # Built per frame: the y scale is the semaphore's own limit, or the
        # highest count drawn when it has none.
        if sem.is_unbounded:
            top, subtitle = max(history, default=0) or 1, "unlimited"
        else:
            top, subtitle = sem.limit or 1, f"0 to {sem.limit} available"
        graph = TUIGraph("Count", subtitle, (0, top), attr)

        graph.draw(
            stdscr,
            self._GRAPH_ROW,
            0,
            graph_height,
            width,
            points=history,
        )
