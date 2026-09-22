# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import SemaphoreInfo
from frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from frontend.tui.widgets import SEMAPHORE, TUIGraph


class SemaphoreDetailView(KernelObjectDetailView):
    """
    One semaphore: count, limit, waiter count, a count graph and the wait queue.

    The graph plots one sample per polled frame. A take and a give between two
    frames cancel out and leave no trace.
    """

    # The title row already carries count over limit and the waiters, so the
    # boxes add what it cannot show.
    _INFO_TITLES = (("Location", 2), ("Limit", 1))
    _GRAPH_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT

    def _target(self) -> SemaphoreInfo | None:
        address = self.controller.detailing_semaphore_address
        if address is None:
            return None

        return next((s for s in self.controller.semaphores_data if s.address == address), None)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        sem = self._target()
        if sem is None:
            stdscr.addstr(1, 0, " Semaphore is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, SEMAPHORE, sem)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [self._site(sem.address), str(sem.limit)],
        )

        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_count_graph(stdscr, panels.graph_w, sem, panels.graph_h)
        self._draw_wait_queue(stdscr, panels, sem.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_count_graph(
        self, stdscr: curses.window, width: int, sem: SemaphoreInfo, graph_height: int
    ) -> None:
        # Yellow while threads are queued on it, plain otherwise.
        attr = self._busy_attr if sem.waiters else 0

        # Built per frame: the y scale is the semaphore's own limit.
        graph = TUIGraph(
            "Count",
            f"0 to {sem.limit} available",
            (0, sem.limit or 1),
            attr,
        )
        # One column per sample: extra points are averaged into moving buckets.
        columns = max(1, width - 2)
        history = list(self.controller.sem_history.get(sem.address, ()))[-columns:]

        graph.draw(
            stdscr,
            self._GRAPH_ROW,
            0,
            graph_height,
            width,
            points=history,
        )
