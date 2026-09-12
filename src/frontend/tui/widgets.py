# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import textwrap
from typing import NamedTuple

from backend.base import HeapInfo, MutexState, ThreadInfo, ThreadRuntime


def _truncate_str(text: str, max_size: int) -> str:
    return text if len(text) <= max_size else text[: max_size - 3] + "..."


def _fit_str(text: str, width: int, align: str = "^") -> str:
    """Pad ``text`` to ``width`` (``align`` is one of ``^<>``), then hard-clip to ``width``."""
    return f"{text:{align}{width}}"[:width]


def _addstr_clipped(
    stdscr: curses.window,
    y: int,
    x: int,
    text: str,
    screen_w: int,
    attr: int | None = None,
) -> None:
    """``addstr`` that hard-clips to ``screen_w - x`` so writes never wrap."""
    available = screen_w - x
    if available <= 0:
        return
    text = text[:available]
    with contextlib.suppress(curses.error):
        if attr is None:
            stdscr.addstr(y, x, text)
        else:
            stdscr.addstr(y, x, text, attr)


class TUIProgressBar:
    def __init__(
        self,
        width: int,
        std_attribute: int,
        medium_threshold: tuple[float, int],
        high_threshold: tuple[float, int],
    ):
        self._width = width
        self._bar_width = width - 2

        self._low_threshold_attr: int = std_attribute

        self._medium_threshold: float = medium_threshold[0]
        self._medium_threshold_attr: int = medium_threshold[1]

        self._high_threshold: float = high_threshold[0]
        self._high_threshold_attr: int = high_threshold[1]

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, new: int) -> None:
        self._width = new
        self._bar_width = new - 2

    HIGH_WATER_MARK = "┃"

    def draw(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        percentage: float,
        label: str | None = None,
        attr: int | None = None,
        mark: float | None = None,
    ):
        """
        Draw the bar at ``percentage``, captioned with ``label`` or the percentage.

        ``attr`` overrides the threshold colors, for callers that color a bar by
        state rather than by how full it is. ``mark`` is a second percentage
        drawn as a line across the track, for a high-water mark.
        """
        if attr is not None:
            bar_color_attr = attr
        elif percentage > self._high_threshold:
            bar_color_attr = self._high_threshold_attr
        elif percentage > self._medium_threshold:
            bar_color_attr = self._medium_threshold_attr
        else:
            bar_color_attr = self._low_threshold_attr

        completed_chars = int(self._bar_width * (percentage / 100))
        stdscr.addstr(y, x, "│" + "·" * self._bar_width + "│")
        x += 1

        stdscr.attron(bar_color_attr)
        stdscr.addstr(y, x, "█" * completed_chars)

        percent_display = _truncate_str(
            f"{percentage:.1f}%" if label is None else label, self._bar_width
        )
        percent_start_x = x + (self.width // 2) - (len(percent_display) // 2)
        bar_end_x = x + completed_chars

        split_point = max(0, min(len(percent_display), bar_end_x - percent_start_x))

        text_over_bar = percent_display[:split_point]
        if text_over_bar:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(y, percent_start_x, text_over_bar)
            stdscr.attroff(curses.A_REVERSE)

        text_outside_bar = percent_display[split_point:]
        if text_outside_bar:
            stdscr.addstr(y, percent_start_x + split_point, text_outside_bar)

        # After the caption, which would otherwise cover the cell.
        if mark is not None and self._bar_width > 0:
            cell = min(self._bar_width - 1, int(self._bar_width * (mark / 100)))
            stdscr.addstr(y, x + max(0, cell), self.HIGH_WATER_MARK)

        stdscr.attroff(bar_color_attr)


class TUIBox:
    def __init__(self, title: str, description: str, attribute: int):
        self._title: str = title
        self._description: str = description
        self._attr: int = attribute

    def draw(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        height: int,
        width: int,
        **kwargs,
    ):
        title = _truncate_str(self._title, width - 2)
        description = _truncate_str(self._description, width - 2)

        horizontal_bar = "─" * (width - 2)
        top_str = "┌" + title + horizontal_bar[len(title) :] + "┐"
        bottom_str = "└" + description + horizontal_bar[len(description) :] + "┘"
        side_str = "│"

        stdscr.attron(self._attr)

        stdscr.addstr(y, x, top_str)
        for row in range(1, height - 1):
            stdscr.addstr(y + row, x, side_str)
            stdscr.addstr(y + row, x + width - 1, side_str)
        stdscr.addstr(y + height - 1, x, bottom_str)

        stdscr.attroff(self._attr)


class PopupRow(NamedTuple):
    """
    One row of a ``TUIPopup``.

    ``label`` and ``text`` are the two columns; ``level`` picks the text
    color. A heading row carries its title in ``label``, and a row with both
    columns empty is a gap.
    """

    label: str
    text: str = ""
    level: str = "info"
    heading: bool = False


class TUIPopup:
    """
    Centered popup listing two-column rows, blanking the area it covers.

    ``keep_tail`` picks which rows survive when they do not all fit: the last
    ones for a log, the first ones for a list that reads top down.
    """

    _BORDER_THICKNESS = 2  # top + bottom
    _PADDING_ROWS = 2  # blank row above + below content
    _PADDING_COLS = 4  # left + right inner padding (2 each side)
    _COLUMN_GAP = 2  # spaces between the label column and the text column
    _MIN_BOX_WIDTH = 34

    def __init__(
        self,
        title: str,
        frame_attr: int,
        label_attr: int,
        level_attrs: dict[str, int],
        keep_tail: bool = False,
    ):
        self._title = title
        self._frame_attr = frame_attr
        self._label_attr = label_attr
        self._level_attrs = level_attrs
        self._keep_tail = keep_tail

    def _wrap(self, rows: list[PopupRow], text_room: int) -> list[PopupRow]:
        """
        Split rows whose text is wider than its column.

        The continuation lines carry no label, so the popup reads as one entry
        per label with its text below it.
        """
        if text_room < 1:
            return rows

        wrapped: list[PopupRow] = []
        for row in rows:
            if row.heading or len(row.text) <= text_room:
                wrapped.append(row)
                continue

            pieces = textwrap.wrap(row.text, text_room) or [row.text[:text_room]]
            wrapped.append(PopupRow(row.label, pieces[0], row.level))
            wrapped += [PopupRow("", piece, row.level) for piece in pieces[1:]]

        return wrapped

    def draw(self, stdscr: curses.window, height: int, width: int, rows: list[PopupRow]) -> None:
        if not rows:
            return

        # The last row and column stay free.
        max_rows = height - self._BORDER_THICKNESS - self._PADDING_ROWS - 1
        max_width = width - 1
        if max_rows < 1 or max_width < self._MIN_BOX_WIDTH:
            return

        label_w = max(len(row.label) for row in rows if not row.heading)
        text_w = max(len(row.text) for row in rows)
        needed = label_w + self._COLUMN_GAP + text_w + self._PADDING_COLS
        box_w = min(max(needed, self._MIN_BOX_WIDTH), max_width)

        # Wrap before trimming: what has to fit the height is lines, not rows.
        text_room = box_w - self._PADDING_COLS - label_w - self._COLUMN_GAP
        rows = self._wrap(rows, text_room)
        rows = rows[-max_rows:] if self._keep_tail else rows[:max_rows]
        box_h = len(rows) + self._BORDER_THICKNESS + self._PADDING_ROWS

        y0 = (height - box_h) // 2
        x0 = (width - box_w) // 2

        blank = " " * box_w
        for row_offset in range(box_h):
            with contextlib.suppress(curses.error):
                stdscr.addstr(y0 + row_offset, x0, blank)

        TUIBox(self._title, " Press any key to dismiss ", self._frame_attr).draw(
            stdscr, y0, x0, box_h, box_w
        )

        inner_w = box_w - self._PADDING_COLS
        for i, row in enumerate(rows):
            line_y = y0 + 2 + i
            with contextlib.suppress(curses.error):
                if row.heading:
                    stdscr.addstr(
                        line_y, x0 + 2, row.label[:inner_w], self._frame_attr | curses.A_BOLD
                    )
                    continue

                if row.label:
                    stdscr.addstr(line_y, x0 + 2, row.label[:inner_w], self._label_attr)
                if row.text and text_room > 0:
                    stdscr.addstr(
                        line_y,
                        x0 + 2 + label_w + self._COLUMN_GAP,
                        row.text[:text_room],
                        self._level_attrs.get(row.level, 0),
                    )


class TUIGraph(TUIBox):
    def __init__(self, title: str, description: str, limits: tuple[int, int], attribute: int):
        super().__init__(title, description, attribute)

        self._max_limit: int = max(limits) or 1
        self._min_limit: int = min(limits)
        self._max_limit_str = f"{self._max_limit}"
        self._min_limit_str = f"{self._min_limit}"

        self._blocks = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        self._blocks_res = len(self._blocks)

    def _process_points(self, points: list[int | float], target_len: int):
        n = len(points)

        if n < target_len:
            return [0.0] * (target_len - n) + points

        # n >= target_len
        res = [0.0] * target_len
        for i in range(target_len):
            start = (i * n) // target_len
            end = ((i + 1) * n) // target_len

            bucket = points[start:end]
            res[i] = sum(bucket) // len(bucket) if bucket else 0

        return res

    def draw(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        height: int,
        width: int,
        **kwargs,
    ):
        super().draw(stdscr, y, x, height, width)

        all_points: list[float | int] = kwargs.get("points", [])
        if not all_points:
            return

        norm_points = self._process_points(all_points, width - 2)

        internal_height = height - 2
        internal_width = width - 2
        stdscr.attron(self._attr)
        for x_step in range(internal_width):
            x_pos = x + x_step + 1
            full_blocks_f = (norm_points[x_step] / self._max_limit) * internal_height
            full_blocks_count = int(full_blocks_f)
            last_block_idx = int((full_blocks_f - full_blocks_count) * (self._blocks_res - 1))

            for y_step in range(internal_height):
                y_pos = y + internal_height - y_step
                if y_step < full_blocks_count:
                    stdscr.addstr(y_pos, x_pos, self._blocks[-1])
                elif y_step == full_blocks_count:
                    stdscr.addstr(y_pos, x_pos, self._blocks[last_block_idx])
                else:
                    stdscr.addstr(y_pos, x_pos, " ")

        stdscr.addstr(y + 1, x + width - len(self._max_limit_str), self._max_limit_str)
        stdscr.addstr(
            y + internal_height, x + width - len(self._min_limit_str), self._min_limit_str
        )
        stdscr.attroff(self._attr)


class TUIThreadInfo:
    def __init__(
        self,
        selected_attribute: int,
        active_attribute: int,
        inactive_attribute: int,
        bar_attributes: tuple[int, int, int],
    ):
        self._selected_attribute: int = selected_attribute
        self._active_attribute: int = active_attribute
        self._inactive_attribute: int = inactive_attribute

        # These are nice values to default to
        self._thread_name_width = 30
        self._cpu_usage_width = 8
        self._load_usage_width = 8
        self._stack_bytes_width = 18

        self.watermark_bar = TUIProgressBar(
            32,
            bar_attributes[0],
            (75, bar_attributes[1]),
            (90, bar_attributes[2]),
        )

    def set_field_widths(
        self, name: int, cpu_usage: int, load_usage: int, stack_bar: int, stack_bytes: int
    ):
        self._thread_name_width = name
        self._cpu_usage_width = cpu_usage
        self._load_usage_width = load_usage
        self._stack_bytes_width = stack_bytes

        self.watermark_bar.width = stack_bar

    def draw(
        self, stdscr: curses.window, y: int, x: int, thread_info: ThreadInfo, selected: bool = False
    ):
        col_pos = x
        _, screen_w = stdscr.getmaxyx()

        runtime = thread_info.runtime or ThreadRuntime(
            cpu=-1.0,
            cpu_normalized=-1.0,
            active=False,
            stack_watermark=0,
            stack_watermark_percent=0.0,
        )

        # Thread name
        thread_name_attr = (
            self._selected_attribute
            if selected
            else (self._active_attribute if runtime.active else self._inactive_attribute)
        )
        _addstr_clipped(
            stdscr,
            y,
            col_pos,
            _truncate_str(thread_info.name, self._thread_name_width),
            screen_w,
            thread_name_attr,
        )
        col_pos += self._thread_name_width + 1

        # Thread CPUs
        if runtime.cpu >= 0:
            cpu_display = _fit_str(f"{runtime.cpu_normalized:.2f}%", self._cpu_usage_width)
        else:
            cpu_display = _fit_str("-", self._cpu_usage_width)
        _addstr_clipped(stdscr, y, col_pos, cpu_display, screen_w)
        col_pos += self._cpu_usage_width + 1

        # Thread Loads
        if runtime.cpu >= 0:
            load_display = _fit_str(f"{runtime.cpu:.1f}%", self._load_usage_width)
        else:
            load_display = _fit_str("-", self._load_usage_width)
        _addstr_clipped(stdscr, y, col_pos, load_display, screen_w)
        col_pos += self._load_usage_width + 1

        # Thread Watermark Progress Bar
        self.watermark_bar.draw(stdscr, y, col_pos, runtime.stack_watermark_percent)
        col_pos += self.watermark_bar.width + 1

        # Thread Watermark Bytes
        watermark_bytes_display = _fit_str(
            f"{runtime.stack_watermark} / {thread_info.stack_size}", self._stack_bytes_width
        )
        _addstr_clipped(stdscr, y, col_pos, watermark_bytes_display, screen_w)


class TUIHeapInfo:
    def __init__(
        self,
        selected_attribute: int,
        default_attribute: int,
        bar_attributes: tuple[int, int, int],
    ):
        self._selected_attribute: int = selected_attribute
        self._default_attribute: int = default_attribute

        # These are nice values to default to
        self._heap_name_width = 30
        self._free_bytes_width = 8
        self._allocated_bytes_width = 8
        self._watermark_width = 18

        self.usage_bar = TUIProgressBar(
            32,
            bar_attributes[0],
            (75, bar_attributes[1]),
            (90, bar_attributes[2]),
        )

    def set_field_widths(
        self, name: int, free_bytes: int, allocated_bytes: int, usage_bar: int, watermark: int
    ):
        self._heap_name_width = name
        self._free_bytes_width = free_bytes
        self._allocated_bytes_width = allocated_bytes
        self._watermark_width = watermark

        self.usage_bar.width = usage_bar

    def draw(
        self, stdscr: curses.window, y: int, x: int, heap_info: HeapInfo, selected: bool = False
    ):
        col_pos = x
        _, screen_w = stdscr.getmaxyx()

        # Heap name
        heap_name_display = _truncate_str(heap_info.name, self._heap_name_width)
        heap_name_attr = self._selected_attribute if selected else self._default_attribute
        _addstr_clipped(stdscr, y, col_pos, heap_name_display, screen_w, heap_name_attr)
        col_pos += self._heap_name_width + 1

        # Free bytes
        free_bytes_display = _fit_str(str(heap_info.free_bytes), self._free_bytes_width)
        _addstr_clipped(stdscr, y, col_pos, free_bytes_display, screen_w)
        col_pos += self._free_bytes_width + 1

        # Allocated bytes
        allocated_bytes_display = _fit_str(
            str(heap_info.allocated_bytes), self._allocated_bytes_width
        )
        _addstr_clipped(stdscr, y, col_pos, allocated_bytes_display, screen_w)
        col_pos += self._allocated_bytes_width + 1

        # Heap Usage Progress Bar
        heap_size = heap_info.allocated_bytes + heap_info.free_bytes
        self.usage_bar.draw(stdscr, y, col_pos, heap_info.usage_percent)
        col_pos += self.usage_bar.width + 1

        # Heap Watermark Bytes
        watermark_bytes_display = _fit_str(
            f"{heap_info.max_allocated_bytes} / {heap_size}",
            self._watermark_width,
        )
        _addstr_clipped(stdscr, y, col_pos, watermark_bytes_display, screen_w)


SEMAPHORE = "SEM"
MUTEX = "MTX"
MSGQ = "MSG"
MEM_SLAB = "SLB"


class TUIKernelObjectInfo:
    """
    One kernel object as a table row: type, name, state bar, wait queue.

    Shared by the kernel objects list and the per-object detail views.
    """

    _MAX_LISTED_WAITERS = 3
    _UNKNOWN = "?"

    def __init__(
        self,
        selected_attribute: int,
        contended_attribute: int,
        busy_attribute: int,
        bar_attributes: tuple[int, int, int],
    ):
        self._selected_attr = selected_attribute
        self._contended_attr = contended_attribute
        self._busy_attr = busy_attribute

        self._type_width = 5
        self._name_width = 24
        self._waiters_width = 30

        # Thresholds out of reach: the bar carries the row's state color, not
        # a usage color.
        self.state_bar = TUIProgressBar(
            26,
            bar_attributes[0],
            (101.0, bar_attributes[1]),
            (102.0, bar_attributes[2]),
        )

    def set_field_widths(self, type_w: int, name_w: int, bar_w: int, waiters_w: int) -> None:
        self._type_width = type_w
        self._name_width = name_w
        self._waiters_width = waiters_w
        self.state_bar.width = bar_w

    def waiters_cell(self, waiters: tuple[str, ...] | None, max_width: int | None = None) -> str:
        """
        ``N (names)`` for a queue, ``-`` for an empty one, ``?`` when not walked.

        Drops names, then the list entirely, so the cell fits ``max_width``
        rather than being cut mid-name.
        """
        if waiters is None:
            return self._UNKNOWN
        if not waiters:
            return "-"

        count = len(waiters)
        listed = ", ".join(waiters[: self._MAX_LISTED_WAITERS])
        if count > self._MAX_LISTED_WAITERS:
            listed += ", ..."

        candidates = [f"{count} ({listed})"]
        if count > 1:
            candidates.append(f"{count} ({waiters[0]}, ...)")
        candidates.append(str(count))

        if max_width is None:
            return candidates[0]

        return next((c for c in candidates if len(c) <= max_width), candidates[-1])

    def row_values(self, kind: str, obj) -> tuple[float, str, str, int]:
        """``(bar percentage, bar label, waiters cell, attribute)`` for one object."""
        cell = self.waiters_cell(obj.waiters, self._waiters_width)

        if kind == SEMAPHORE:
            fill = (obj.count / obj.limit * 100.0) if obj.limit else 0.0
            attr = self._busy_attr if obj.waiters else 0
            return fill, f"{obj.count}/{obj.limit}", cell, attr

        if kind == MSGQ:
            label = f"{obj.used_msgs}/{obj.max_msgs} × {obj.msg_size}B"
            # A full queue blocks its senders.
            attr = self._contended_attr if obj.is_full else (self._busy_attr if obj.waiters else 0)
            return obj.fill_percent, label, cell, attr

        if kind == MEM_SLAB:
            label = f"{obj.num_used}/{obj.num_blocks} × {obj.block_size}B"
            # An exhausted slab blocks the next allocation.
            attr = (
                self._contended_attr
                if obj.is_exhausted
                else (self._busy_attr if obj.waiters else 0)
            )
            return obj.fill_percent, label, cell, attr

        if not obj.is_locked:
            return 0.0, "FREE", cell, 0

        owner = obj.owner_name or f"0x{obj.owner_address:X}"
        label = f"LOCKED ● {owner}"
        if obj.lock_count > 1:
            label += f" ×{obj.lock_count}"

        attr = self._contended_attr if obj.state is MutexState.CONTENDED else self._busy_attr

        return 100.0, label, cell, attr

    def draw(
        self, stdscr: curses.window, y: int, x: int, kind: str, obj, selected: bool = False
    ) -> None:
        fill, label, waiters, attr = self.row_values(kind, obj)
        # Only a slab carries a peak, and only where the build tracks it. A
        # zero peak sits on the track's first cell.
        mark = getattr(obj, "peak_percent", None)
        if mark == 0:
            mark = None
        self.draw_cells(stdscr, y, x, kind, obj.name, fill, label, waiters, attr, selected, mark)

    def draw_cells(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        kind: str,
        name: str,
        fill: float,
        label: str,
        waiters: str,
        attr: int,
        selected: bool = False,
        mark: float | None = None,
    ) -> None:
        """
        Draw the four cells of a row.

        Selection marks the type and name only, so the bar keeps its state
        color. ``mark`` is a high-water percentage drawn across the bar.
        """
        _, screen_w = stdscr.getmaxyx()
        selected_attr = self._selected_attr if selected else attr

        col_pos = x
        _addstr_clipped(
            stdscr, y, col_pos, _fit_str(kind, self._type_width), screen_w, selected_attr
        )
        col_pos += self._type_width + 1

        _addstr_clipped(
            stdscr,
            y,
            col_pos,
            _truncate_str(name, self._name_width).ljust(self._name_width),
            screen_w,
            selected_attr,
        )
        col_pos += self._name_width + 1

        if col_pos + self.state_bar.width <= screen_w:
            self.state_bar.draw(stdscr, y, col_pos, fill, label, attr, mark)
        col_pos += self.state_bar.width + 1

        _addstr_clipped(stdscr, y, col_pos, _fit_str(waiters, self._waiters_width), screen_w, attr)
