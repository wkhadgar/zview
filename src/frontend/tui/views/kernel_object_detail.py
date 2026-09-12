# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Shared layout for the per-object detail views."""

import curses
from typing import NamedTuple

from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.views.kernel_object_list import KernelObjectListView
from frontend.tui.widgets import TUIBox, TUIKernelObjectInfo, TUIWaitQueue


class Panels(NamedTuple):
    """Geometry of the graph and the wait queue for one frame."""

    graph_w: int
    graph_h: int
    queue_x: int
    queue_y: int
    queue_w: int
    queue_h: int


class KernelObjectDetailView(BaseStateView):
    """
    Identity line, a row of info boxes, a graph and a wait queue panel.

    Subclasses declare ``_INFO_TITLES`` as ``(title, weight)`` pairs and pass
    the matching values to ``_draw_info_boxes``. ``_panels`` puts the queue
    beside the graph, or under it on a terminal too narrow to hold both.
    """

    _INFO_BOX_HEIGHT = 3
    _INFO_TITLES: tuple[tuple[str, int], ...] = ()
    # The wait queue column, and the graph width it may not eat into.
    _QUEUE_WIDTH = 30
    _MIN_GRAPH_WIDTH = 44
    # Row 1 holds the list column headers and row 2 the object's own row, as
    # in the thread and heap detail views.
    _HEADER_ROW = 1
    _OBJECT_ROW = 2
    _INFO_ROW = 3
    _GRAPH_HEIGHT = 9

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._contended_attr = theme.ERROR
        self._busy_attr = theme.PROGRESS_BAR_MEDIUM
        self._label_attr = theme.INACTIVE
        self._graph_attr = theme.GRAPH_A

        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._info = TUIKernelObjectInfo(
            theme.CURSOR, theme.ERROR, theme.PROGRESS_BAR_MEDIUM, bar_theme
        )

        # Frames stay plain: color is reserved for state.
        self._info_boxes = [TUIBox(title, "", 0) for title, _ in self._INFO_TITLES]
        self._queue = TUIWaitQueue(theme.INACTIVE)

    def _draw_title(self, stdscr: curses.window, width: int, kind: str, obj: Any) -> None:
        """Draw the list column headers and the object's row beneath them."""
        names = [obj.name]
        widths = KernelObjectListView.compute_widths(width, names)
        self._info.set_field_widths(*widths)

        curr_x = 0
        for header, col_width in zip(KernelObjectListView.COLUMNS, widths, strict=True):
            if curr_x >= width:
                break
            stdscr.addstr(self._HEADER_ROW, curr_x, f"{header:^{col_width}}"[: width - curr_x])
            curr_x += col_width + 1

        self._info.draw(stdscr, self._OBJECT_ROW, 0, kind, obj)

    def _draw_info_boxes(
        self, stdscr: curses.window, y: int, width: int, values: list[str]
    ) -> None:
        """Lay the info boxes side by side, sized by their relative weights."""
        weights = [weight for _, weight in self._INFO_TITLES]
        total_weight = sum(weights) or 1

        x = 0
        for idx, (box, weight, value) in enumerate(
            zip(self._info_boxes, weights, values, strict=True)
        ):
            is_last = idx == len(self._info_boxes) - 1
            box_w = (width - x) if is_last else (width * weight) // total_weight
            box.draw(stdscr, y, x, self._INFO_BOX_HEIGHT, box_w)

            inner_w = box_w - 4  # 2 border cells + 2 padding cells
            if inner_w > 0:
                stdscr.addstr(y + 1, x + 2, value.ljust(inner_w)[:inner_w])
            x += box_w

    def _panels(self, height: int, width: int, top: int) -> Panels:
        """
        Place the graph and the wait queue in the space below ``top``.

        Side by side both take the full height left. Below ``_MIN_GRAPH_WIDTH``
        of graph, the queue goes under it instead.
        """
        room = max(3, height - 2 - top)

        if width - self._QUEUE_WIDTH >= self._MIN_GRAPH_WIDTH:
            graph_w = width - self._QUEUE_WIDTH
            return Panels(graph_w, room, graph_w, top, self._QUEUE_WIDTH, room)

        graph_h = max(3, min(self._GRAPH_HEIGHT, room - 5))
        queue_h = room - graph_h
        if queue_h < 3:
            # Three rows is the least a box draws in; the graph gives first.
            graph_h = max(3, room - 3)
            queue_h = max(3, room - graph_h)

        return Panels(width, graph_h, 0, top + graph_h, width, queue_h)

    def _draw_wait_queue(
        self, stdscr: curses.window, panels: Panels, waiters: tuple[str, ...] | None
    ) -> None:
        self._queue.draw(
            stdscr, panels.queue_y, panels.queue_x, panels.queue_h, panels.queue_w, waiters
        )

    def keybindings(self) -> list[Keybind]:
        return [
            Keybind("<Esc>", "Back", "Return to the kernel objects list"),
            Keybind("k", "Objects", "Return to the kernel objects list"),
        ]

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case (
                curses.KEY_LEFT
                | curses.KEY_EXIT
                | SpecialCode.KERNEL_OBJECTS
                | SpecialCode.NEWLINE
                | SpecialCode.RETURN
                | SpecialCode.ESCAPE
            ):
                return ZViewState.KERNEL_OBJECT_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None

        return None
