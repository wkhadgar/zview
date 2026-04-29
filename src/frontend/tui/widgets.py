# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses

from backend.base import HeapInfo, ThreadInfo, ThreadRuntime


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

    def draw(
        self,
        stdscr: curses.window,
        y: int,
        x: int,
        percentage: float,
    ):
        if percentage > self._high_threshold:
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

        percent_display = f"{percentage:.1f}%"
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


class TUITooltip:
    """Centered, filled, bordered popup listing keybinding sections."""

    _BORDER_THICKNESS = 2  # top + bottom
    _PADDING_ROWS = 2  # blank row above + below content
    _PADDING_COLS = 4  # left + right inner padding (2 each side)
    _KEY_DESC_GAP = 2  # spaces between key column and description column
    _MIN_BOX_WIDTH = 30

    def __init__(self, sections: list[tuple[str, list[tuple[str, str]]]], attr: int):
        """``sections`` is ``[(section_title, [(key, description), ...]), ...]``."""
        self._sections = sections
        self._attr = attr

    def _build_rows(self) -> list[tuple[str, str] | None]:
        """Layout rows; ``None`` is a blank separator between sections."""
        rows: list[tuple[str, str] | None] = []
        for idx, (title, bindings) in enumerate(self._sections):
            if idx > 0:
                rows.append(None)
            rows.append((title, ""))
            rows.extend((f"  {key}", desc) for key, desc in bindings)
        return rows

    def draw(self, stdscr: curses.window, height: int, width: int) -> None:
        rows = self._build_rows()
        visible = [r for r in rows if r is not None]
        key_w = max((len(k) for k, _ in visible), default=0)
        desc_w = max((len(d) for _, d in visible), default=0)

        inner_w = key_w + self._KEY_DESC_GAP + desc_w
        box_w = max(inner_w + self._PADDING_COLS, self._MIN_BOX_WIDTH)
        box_h = len(rows) + self._BORDER_THICKNESS + self._PADDING_ROWS

        if box_h > height or box_w > width:
            return

        y0 = (height - box_h) // 2
        x0 = (width - box_w) // 2

        blank = " " * box_w
        for row_offset in range(box_h):
            with contextlib.suppress(curses.error):
                stdscr.addstr(y0 + row_offset, x0, blank, self._attr)

        TUIBox(" Help ", " Press any key to dismiss ", self._attr).draw(
            stdscr, y0, x0, box_h, box_w
        )

        for i, row in enumerate(rows):
            line_y = y0 + 2 + i
            if row is None:
                continue
            key, desc = row
            with contextlib.suppress(curses.error):
                if not desc:
                    stdscr.addstr(line_y, x0 + 2, key, self._attr | curses.A_BOLD)
                else:
                    stdscr.addstr(line_y, x0 + 2, key.ljust(key_w), self._attr)
                    stdscr.addstr(line_y, x0 + 2 + key_w + self._KEY_DESC_GAP, desc, self._attr)


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
