# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import WorkqInfo
from frontend.tui.views.base import Any, ZViewTUIAttributes
from frontend.tui.views.kernel_object_detail import KernelObjectDetailView, Panels
from frontend.tui.widgets import WORKQ, TUIGraph, TUIWaitQueue


class WorkqDetailView(KernelObjectDetailView):
    """
    One work queue: where it is declared, its thread and state flags, a
    backlog graph, the items waiting to run and the threads draining it.
    """

    # The title row already carries busy or idle, the backlog and the drain
    # waiters, so the boxes add what it cannot show.
    _INFO_TITLES = (("Location", 2), ("Thread", 1), ("Flags", 1))
    _GRAPH_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT
    # Borders and one line of content.
    _MIN_BOX_HEIGHT = 3
    # Wide enough for an item named with its handler.
    _QUEUE_WIDTH = 40

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._pending = TUIWaitQueue(theme.INACTIVE, "Pending")
        self._draining = TUIWaitQueue(theme.INACTIVE, "Drain waiters")

    def _target(self) -> WorkqInfo | None:
        address = self.controller.detailing_workq_address
        if address is None:
            return None

        return next((q for q in self.controller.workqs_data if q.address == address), None)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        workq = self._target()
        if workq is None:
            stdscr.addstr(
                1, 0, " Work queue is no longer being reported."[:width], self._error_attr
            )
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, WORKQ, workq)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [self._site(workq.address), workq.thread_name or "-", ",".join(workq.states) or "-"],
        )

        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_backlog_graph(stdscr, panels.graph_w, workq, panels.graph_h)
        self._draw_side_panels(stdscr, panels, workq)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_backlog_graph(
        self, stdscr: curses.window, width: int, workq: WorkqInfo, graph_height: int
    ) -> None:
        columns = max(1, width - 2)
        history = list(self.controller.workq_history.get(workq.address, ()))[-columns:]
        # A work queue has no capacity, so the scale is the deepest backlog shown.
        deepest = max(history, default=0) or 1

        graph = TUIGraph(
            "Backlog",
            f"0 to {deepest} pending",
            (0, deepest),
            self._info.row_values(WORKQ, workq)[3],
        )
        graph.draw(stdscr, self._GRAPH_ROW, 0, graph_height, width, points=history)

    def _draw_side_panels(self, stdscr: curses.window, panels: Panels, workq: WorkqInfo) -> None:
        """Pending items over the drain waiters, which take a third of the column."""
        pending = workq.pending + (("...",) if workq.pending_truncated else ())
        drain_h = max(self._MIN_BOX_HEIGHT, panels.queue_h // 3)
        pending_h = panels.queue_h - drain_h
        x, y, w = panels.queue_x, panels.queue_y, panels.queue_w

        if pending_h < self._MIN_BOX_HEIGHT:
            # Too short for two boxes, and the header row still names the drain waiters.
            self._pending.draw(stdscr, y, x, panels.queue_h, w, pending)
            return

        self._pending.draw(stdscr, y, x, pending_h, w, pending)
        self._draining.draw(stdscr, y + pending_h, x, drain_h, w, workq.waiters)
