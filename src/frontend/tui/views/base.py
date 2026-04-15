# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ZViewTUIAttributes:
    ACTIVE: int
    INACTIVE: int
    PROGRESS_BAR_LOW: int
    PROGRESS_BAR_MEDIUM: int
    PROGRESS_BAR_HIGH: int
    HEADER_FOOTER: int
    ERROR: int
    CURSOR: int
    GRAPH_A: int
    GRAPH_B: int

    @classmethod
    def create_mono(cls):
        """Returns a default monochromatic theme for featureless consoles."""
        return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)


class SpecialCode:
    QUIT = ord("q")
    NEWLINE = ord("\n")
    RETURN = ord("\r")
    SORT = ord("s")
    INVERSE = ord("i")
    HEAPS = ord("h")
    RECONNECT = ord("r")


class ZViewState(enum.Enum):
    FATAL_ERROR = 1
    THREAD_LIST_VIEW = 2
    THREAD_DETAIL_VIEW = 3
    HEAP_LIST_VIEW = 4
    HEAPS_DETAIL_VIEW = 5


class BaseStateView:
    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        """
        The controller reference allows the view to access global state
        (like colors or the max threads limit) without owning it.
        """
        self.controller = controller
        self.cursor: int = 0
        self._frame_attr: int = theme.HEADER_FOOTER
        self._error_attr: int = theme.ERROR

    def _render_status(
        self,
        stdscr: curses.window,
        width: int,
        y: int,
    ):
        is_error = self.controller.status_message.startswith("Error")
        attr = stdscr.getbkgd()
        attr = attr & ~0xFF if isinstance(attr, int) else attr[0]
        status_row = y
        stdscr.addstr(
            status_row,
            0,
            self.controller.status_message[:width],
            (attr | self._error_attr) if is_error else attr,
        )

    def _render_frame(
        self,
        stdscr: curses.window,
        footer_hint: str,
        height: int,
        width: int,
    ):
        header_text = "ZView - Zephyr RTOS Runtime Viewer"

        stdscr.move(0, 0)
        stdscr.clrtoeol()
        stdscr.addstr(0, 0, f"{header_text:^{width}}", self._frame_attr)
        footer_row = height - 1
        stdscr.move(footer_row, 0)
        stdscr.clrtoeol()
        with contextlib.suppress(curses.error):
            # This is needed since curses try to advance the cursor to the next
            # position, wich is outside the terminal, we safely ignore this.
            stdscr.addstr(footer_row, 0, f"{footer_hint:>{width}}", self._frame_attr)

    @abstractmethod
    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draw the state to the provided curses window.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def handle_input(self, key: int) -> ZViewState | None:
        """
        Process navigation and state-specific inputs.

        Returns:
            A state enum (e.g., ZViewState.THREAD_DETAIL) to request a
            context switch from the Router, or None if the state remains unchanged.
        """
        pass
