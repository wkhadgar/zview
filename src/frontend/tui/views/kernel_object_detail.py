# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Shared layout for the per-object detail views."""

import curses

from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.views.kernel_object_list import KernelObjectListView
from frontend.tui.widgets import TUIBox, TUIKernelObjectInfo


class KernelObjectDetailView(BaseStateView):
    """
    Identity line, a row of info boxes, and a wait queue panel.

    Subclasses declare ``_INFO_TITLES`` as ``(title, weight)`` pairs and pass
    the matching values to ``_draw_info_boxes``. The wait queue panel is drawn
    last and takes the remaining height, so its length never moves anything
    above it.
    """

    _INFO_BOX_HEIGHT = 3
    _INFO_TITLES: tuple[tuple[str, int], ...] = ()
    # Row 1 holds the list column headers and row 2 the object's own row, as
    # in the thread and heap detail views.
    _HEADER_ROW = 1
    _OBJECT_ROW = 2
    _INFO_ROW = 3

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

        # Frames stay plain: color is reserved for state (yellow busy, red
        # contended), so it always means something.
        self._info_boxes = [TUIBox(title, "", 0) for title, _ in self._INFO_TITLES]
        self._queue_box = TUIBox("Wait queue", "", 0)

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

    def _draw_wait_queue(
        self,
        stdscr: curses.window,
        y: int,
        height: int,
        width: int,
        waiters: tuple[str, ...] | None,
    ) -> None:
        """Draw the wait queue panel, filling the height left below ``y``."""
        box_h = max(3, height - 2 - y)
        self._queue_box.draw(stdscr, y, 0, box_h, width)

        inner_w = width - 4
        if inner_w <= 0:
            return

        if waiters is None:
            text = "not observable on this build (CONFIG_WAITQ_SCALABLE)"
            stdscr.addstr(y + 1, 2, text[:inner_w], self._label_attr)
            return

        if not waiters:
            stdscr.addstr(y + 1, 2, "empty"[:inner_w], self._label_attr)
            return

        last_row = y + box_h - 2
        for idx, waiter in enumerate(waiters):
            row = y + 1 + idx
            if row > last_row:
                stdscr.addstr(last_row, 2, f"... {len(waiters) - idx + 1} more"[:inner_w])
                break

            branch = "└─" if idx == len(waiters) - 1 else "├─"
            stdscr.addstr(row, 2, f"{branch} {waiter}"[:inner_w])

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
