# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from zview.backend.base import MsgqInfo
from zview.frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from zview.frontend.tui.widgets import TUIGraph
from zview.kinds import BY_KEY


class MsgqDetailView(KernelObjectDetailView):
    """
    One message queue: fill level, capacity, message size, a depth graph and
    the wait queue.

    The graph plots one sample per polled frame. A message written and read
    between two frames cancels out and leaves no trace.
    """

    kind = BY_KEY["msgqs"]
    # The title row already carries used over capacity and the waiters, so the
    # boxes add what it cannot show.
    _INFO_TITLES = (("Location", 2), ("Capacity", 1), ("Message size", 1))
    _GRAPH_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        msgq = self._target()
        if msgq is None:
            stdscr.addstr(1, 0, " Queue is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, msgq)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [self._site(msgq), str(msgq.max_msgs), f"{msgq.msg_size} B"],
        )

        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_depth_graph(stdscr, panels.graph_w, msgq, panels.graph_h)
        self._draw_wait_queue(stdscr, panels, msgq.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_depth_graph(
        self, stdscr: curses.window, width: int, msgq: MsgqInfo, graph_height: int
    ) -> None:
        # Red at capacity, yellow while threads are queued on it, plain otherwise.
        if msgq.is_full:
            attr = self._contended_attr
        else:
            attr = self._busy_attr if msgq.waiters else 0

        # Built per frame: the y scale is the queue's own capacity.
        graph = TUIGraph(
            "Depth",
            f"0 to {msgq.max_msgs} messages",
            (0, msgq.max_msgs or 1),
            attr,
        )
        # One column per sample: extra points are averaged into moving buckets.
        columns = max(1, width - 2)
        history = [q.used_msgs for q in self.controller.frames.of("msgqs", msgq.address)][-columns:]

        graph.draw(
            stdscr,
            self._GRAPH_ROW,
            0,
            graph_height,
            width,
            points=history,
        )
