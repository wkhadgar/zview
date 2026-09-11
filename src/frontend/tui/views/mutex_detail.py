# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses
from itertools import groupby

from backend.base import MutexInfo, MutexState
from frontend.tui.views.base import Any, ZViewTUIAttributes
from frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from frontend.tui.widgets import MUTEX, TUIBox


class MutexDetailView(KernelObjectDetailView):
    """
    One mutex: state, lock depth, waiter count, owner, contention strip and
    wait queue.

    The strip carries one mark per polled frame: free, held alone, or held
    with threads queued on it.
    """

    CONTENDED_MARK = "●"
    LOCKED_MARK = "○"
    FREE_MARK = "·"

    MARKS = {
        MutexState.CONTENDED: CONTENDED_MARK,
        MutexState.LOCKED: LOCKED_MARK,
        MutexState.FREE: FREE_MARK,
    }

    # The title row already carries state and waiters, so the boxes add what
    # it cannot show.
    _INFO_TITLES = (("Address", 2), ("Depth", 1), ("Owner", 3))
    _HISTORY_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT
    _HISTORY_HEIGHT = 3
    _QUEUE_ROW = _HISTORY_ROW + _HISTORY_HEIGHT

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)

        self._mark_attrs = {
            MutexState.CONTENDED: self._contended_attr,
            MutexState.LOCKED: self._busy_attr,
            MutexState.FREE: 0,
        }
        self._attr_by_mark = {self.MARKS[state]: attr for state, attr in self._mark_attrs.items()}

        self._history_legend = (
            f"one mark per frame: {self.CONTENDED_MARK} contended  "
            f"{self.LOCKED_MARK} locked  {self.FREE_MARK} free"
        )
        self._history_box = TUIBox("Contention history", self._history_legend, 0)

    def _target(self) -> MutexInfo | None:
        address = self.controller.detailing_mutex_address
        if address is None:
            return None

        return next((m for m in self.controller.mutexes_data if m.address == address), None)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        mutex = self._target()
        if mutex is None:
            stdscr.addstr(1, 0, " Mutex is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, MUTEX, mutex)

        waiters = mutex.waiters
        owner = "-"
        if mutex.is_locked:
            owner = mutex.owner_name or f"thread @ 0x{mutex.owner_address:X}"

        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [f"0x{mutex.address:X}", str(mutex.lock_count), owner],
        )

        self._draw_history(stdscr, width, mutex.address)
        self._draw_wait_queue(stdscr, self._QUEUE_ROW, height, width, waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_history(self, stdscr: curses.window, width: int, address: int) -> None:
        self._history_box.draw(stdscr, self._HISTORY_ROW, 0, self._HISTORY_HEIGHT, width)

        # The box paints its whole bottom border in one attribute, so the
        # legend's marks arrive plain. Repaint them in the colors they stand for.
        legend_row = self._HISTORY_ROW + self._HISTORY_HEIGHT - 1
        for state, mark_x in self.legend_mark_columns().items():
            if width - 2 > mark_x:
                stdscr.addstr(legend_row, mark_x, self.MARKS[state], self._mark_attrs[state])

        history = self.controller.mutex_history.get(address, ())
        inner_w = width - 4
        if not history or inner_w <= 0:
            return

        marks = "".join(self.MARKS[state] for state in history)[-inner_w:]

        # One color per state, the same color language the list rows use:
        # contended red, held alone yellow, free plain.
        x = 2
        for mark, group in groupby(marks):
            run = "".join(group)
            stdscr.addstr(self._HISTORY_ROW + 1, x, run, self._attr_by_mark[mark])
            x += len(run)

    def legend_mark_columns(self) -> dict[MutexState, int]:
        """Screen column of each colored legend mark, keyed by the state it stands for."""
        return {
            state: 1 + self._history_legend.index(self.MARKS[state])
            for state in (MutexState.CONTENDED, MutexState.LOCKED)
        }
