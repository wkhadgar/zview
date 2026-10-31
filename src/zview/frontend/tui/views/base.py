# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import enum
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

from zview.selection import EVERY_ROW, Selection


class Keybind(NamedTuple):
    """One keybinding entry; ``label`` is footer-sized, ``help_text`` is overlay-sized."""

    key: str
    label: str
    help_text: str


def compute_flex_widths(
    schema_widths: list[int],
    terminal_width: int,
    items: list,
    name_idx: int = 0,
    bar_idx: int = 3,
) -> list[int]:
    """Schema widths are floor; extra goes to name (only as needed) then bar."""
    base = list(schema_widths)
    floor = sum(base) + len(base) - 1
    # Reserve one terminal column (curses errors when the cursor advances off-screen).
    extra = terminal_width - floor - 1
    if extra <= 0:
        return base
    max_name = max((len(item.name) for item in items), default=0)
    name_grow = min(extra, max(0, max_name - base[name_idx]))
    base[name_idx] += name_grow
    extra -= name_grow
    base[bar_idx] += extra
    return base


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
    ESCAPE = 27
    KERNEL_OBJECTS = ord("k")
    REFRESH = ord("r")
    RECONNECT = ord("R")
    HELP = ord("?")
    MESSAGES = ord("m")
    MUTE = ord("x")
    PATTERNS = ord("/")
    WIDER = ord("+")
    NARROWER = ord("-")
    BACKSPACE = ord("\b")
    DELETE = 127


# The bindings several views share.
DETAIL_KEYS = Keybind("<Enter>", "Detail", "Open the selected row's detail view")
BACK_KEYS = Keybind("<Esc>", "Back", "Return to the list, as <Enter> does")
BACK_CODES = (curses.KEY_ENTER, SpecialCode.NEWLINE, SpecialCode.RETURN, SpecialCode.ESCAPE)
SORT_KEYS = Keybind("s", "Sort", "Cycle through sort keys")
INVERT_KEYS = Keybind("i", "Invert", "Reverse the current sort order")
MUTE_KEYS = Keybind("x", "Mute", "Mute or unmute the selected row, kept dimmed as last read")
USAGE_WINDOW_KEYS = Keybind(
    "+/-", "Window", "Sum the CPU and load shares over twice or half the polls"
)


class ZViewState(enum.Enum):
    FATAL_ERROR = 1
    THREAD_LIST_VIEW = 2
    THREAD_DETAIL_VIEW = 3
    KERNEL_OBJECT_LIST_VIEW = 6
    OBJECT_DETAIL_VIEW = 7


class BaseStateView:
    _MAX_FOOTER_HINTS = 3

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

    def keybindings(self) -> list[Keybind]:
        """View-specific bindings, ordered most-important first."""
        return []

    def reads(self, patterns: Selection) -> Selection:
        """What the view reads of the rows ``patterns`` read, as the core is told to."""
        return patterns

    def can_open(self) -> bool:
        return True

    def row(self) -> tuple[str, str] | None:
        """The ``(group, name)`` of the one row the view shows, if it shows one."""
        return None

    def opened(self) -> None:  # noqa: B027
        """Called as the view opens on a live target."""

    def closed(self) -> None:  # noqa: B027
        """Called as another view replaces it on a live target."""

    @staticmethod
    def _draw_header(stdscr: curses.window, x: int, header: str, room: int, width: int) -> None:
        """
        ``header`` centred in ``room`` columns at ``x`` on the header row, moved
        left into the space before it when a sort mark would pass the screen's edge.
        """
        cell = f"{header:^{room}}"
        x = max(0, min(x, width - len(cell)))
        stdscr.addstr(1, x, cell[: width - x])

    def _bar_count(self, count: str, room: int) -> str:
        """
        ``count``, then the groups off, the patterns, the rows muted in this
        view and the polls its CPU shares are summed over, cut to ``room`` columns.
        """
        selection = self.controller.selection
        # What the view reads of every row.
        reads = self.reads(Selection())
        window = self.controller.core.usage_window
        off = [
            group
            for group in self.controller.core.groups
            if reads.reads_group(group) and not selection.reads_group(group)
        ]
        muted = sum(1 for group, _ in self.controller.muted if reads.reads_group(group))
        notes = [
            note
            for shown, note in (
                (off, f"Off: {', '.join(off)}"),
                (
                    selection.include != EVERY_ROW or selection.exclude,
                    f"Patterns: {selection.pattern_line()}",
                ),
                (muted, f"{muted} muted"),
                (reads.reads_group("threads") and window > 1, f"CPU over {window} polls"),
            )
            if shown
        ]
        text = f" {' | '.join([count, *notes])} "
        return text if len(text) <= room else text[: max(0, room - 2)] + "… "

    def _footer_hint(self) -> str:
        """View bindings (up to ``_MAX_FOOTER_HINTS``, ``...`` if truncated) + ``Help: ?``."""
        bindings = self.keybindings()
        parts = [f"{b.label}: {b.key}" for b in bindings[: self._MAX_FOOTER_HINTS]]
        if len(bindings) > self._MAX_FOOTER_HINTS:
            parts.append("…")
        parts.insert(0, "Help: ?")
        return " | ".join(parts) + " "

    @abstractmethod
    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """Draw the state to the provided curses window."""
        pass

    @abstractmethod
    def handle_input(self, key: int) -> ZViewState | None:
        """Process navigation and state-specific inputs; returns a target state or ``None``."""
        pass
