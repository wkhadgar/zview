# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.z_scraper import HeapInfo, ThreadInfo, ThreadRuntime


def _truncate_str(text: str, max_size: int) -> str:
    return text if len(text) < max_size else text[: max_size - 3] + "..."


class TUIProgressBar:
    def __init__(
        self,
        width: int,
        std_attribute: int,
        medium_threshold: tuple[float, int],
        high_threshold: tuple[float, int],
    ):
        self.width = width
        self._bar_width = self.width - 2

        self._low_threshold_attr: int = std_attribute

        self._medium_threshold: float = medium_threshold[0]
        self._medium_threshold_attr: int = medium_threshold[1]

        self._high_threshold: float = high_threshold[0]
        self._high_threshold_attr: int = high_threshold[1]

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
    def __init__(self, h: int, w: int, title: str, description: str):
        self._h: int = h
        self._w: int = w
        self._title: str = _truncate_str(title, w - 2)
        self._description: str = _truncate_str(description, w - 2)

        horizontal_bar = "─" * (self._w - 2)
        self._top_str = "┌" + self._title + horizontal_bar[len(self._title) :] + "┐"
        self._bottom_str = "└" + self._description + horizontal_bar[len(self._description) :] + "┘"
        self._side_str = "│"

    def draw(self, stdscr: curses.window, y: int, x: int, attributes: int, **kwargs):
        stdscr.attron(attributes)

        stdscr.addstr(y, x, self._top_str)
        for row in range(1, self._h - 1):
            stdscr.addstr(y + row, x, self._side_str)
            stdscr.addstr(y + row, x + self._w - 1, self._side_str)
        stdscr.addstr(y + self._h - 1, x, self._bottom_str)

        stdscr.attroff(attributes)


class TUIGraph(TUIBox):
    def __init__(
        self,
        h: int,
        w: int,
        title: str,
        description: str,
        limits: tuple[int, int],
    ):
        super().__init__(h, w, title, description)

        self._max_limit: int = max(limits)
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
            res[i] = sum(bucket) // len(bucket)

        return res

    def draw(self, stdscr: curses.window, y, x, attributes: int, **kwargs):
        super().draw(stdscr, y, x, attributes)

        all_points: list[float | int] = kwargs.get("points", [])
        if not all_points:
            return

        norm_points = self._process_points(all_points, self._w - 2)

        internal_height = self._h - 2
        internal_width = self._w - 2
        stdscr.attron(attributes)
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

        stdscr.addstr(y + 1, x + self._w - len(self._max_limit_str), self._max_limit_str)
        stdscr.addstr(
            y + internal_height, x + self._w - len(self._min_limit_str), self._min_limit_str
        )
        stdscr.attroff(attributes)


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
        self._stack_bar_width = stack_bar
        self._stack_bytes_width = stack_bytes

        self.watermark_bar.width = stack_bar

    def draw(
        self, stdscr: curses.window, y: int, x: int, thread_info: ThreadInfo, selected: bool = False
    ):
        col_pos = x

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
        stdscr.addstr(
            y, col_pos, _truncate_str(thread_info.name, self._thread_name_width), thread_name_attr
        )
        col_pos += self._thread_name_width + 1

        # Thread CPUs
        if runtime.cpu >= 0:
            cpu_display = f"{runtime.cpu_normalized:.2f}%".center(self._cpu_usage_width)
        else:
            cpu_display = f"{'-':^{self._cpu_usage_width}}"
        stdscr.addstr(y, col_pos, cpu_display)
        col_pos += self._cpu_usage_width + 1

        # Thread Loads
        if runtime.cpu >= 0:
            load_display = f"{runtime.cpu:.1f}%".center(self._load_usage_width)
        else:
            load_display = f"{'-':^{self._load_usage_width}}"
        stdscr.addstr(y, col_pos, load_display)
        col_pos += self._load_usage_width + 1

        # Thread Watermark Progress Bar
        self.watermark_bar.draw(stdscr, y, col_pos, runtime.stack_watermark_percent)
        col_pos += self.watermark_bar.width + 1

        # Thread Watermark Bytes
        watermark_bytes_display = f"{runtime.stack_watermark} / {thread_info.stack_size}".center(
            self._stack_bytes_width
        )
        stdscr.addstr(y, col_pos, watermark_bytes_display)


class TUIHeapInfo:
    def __init__(
        self,
        selected_attribute: int,
        default_attribute: int,
        bar_attributes: tuple[int, int, int],
    ):
        self._selected_attribute: int = selected_attribute
        self._default_attribute: int = default_attribute

        self._heap_info_scheme = {
            "Heap": 30,
            "Free B": 8,
            "Used B": 8,
            "Heap Usage %": 32,
            "Watermark Bytes": 18,
        }
        self._columns = list(self._heap_info_scheme.keys())

        self.usage_bar = TUIProgressBar(
            self._heap_info_scheme[self._columns[3]],
            bar_attributes[0],
            (75, bar_attributes[1]),
            (90, bar_attributes[2]),
        )

    def draw(
        self, stdscr: curses.window, y: int, x: int, heap_info: HeapInfo, selected: bool = False
    ):
        col_pos = x

        # Widths
        heap_name_width = self._heap_info_scheme[self._columns[0]]
        free_bytes_width = self._heap_info_scheme[self._columns[1]]
        allocated_bytes_width = self._heap_info_scheme[self._columns[2]]
        heap_usage_width = self._heap_info_scheme[self._columns[3]]
        watermark_width = self._heap_info_scheme[self._columns[4]]

        # Heap name
        heap_name_display = _truncate_str(heap_info.name, heap_name_width)
        heap_name_attr = self._selected_attribute if selected else self._default_attribute
        stdscr.addstr(y, col_pos, heap_name_display, heap_name_attr)
        col_pos += heap_name_width + 1

        # Free bytes
        free_bytes_display = f"{heap_info.free_bytes:^{free_bytes_width}}"
        stdscr.addstr(y, col_pos, free_bytes_display)
        col_pos += free_bytes_width + 1

        # Allocated bytes
        allocated_bytes_display = f"{heap_info.allocated_bytes:^{allocated_bytes_width}}"
        stdscr.addstr(y, col_pos, allocated_bytes_display)
        col_pos += allocated_bytes_width + 1

        # Heap Usage Progress Bar
        heap_size = heap_info.allocated_bytes + heap_info.free_bytes
        self.usage_bar.draw(stdscr, y, col_pos, heap_info.usage_percent)
        col_pos += heap_usage_width + 1

        # Heap Watermark Bytes
        watermark_bytes_display = f"{heap_info.max_allocated_bytes} / {heap_size}".ljust(
            watermark_width
        )
        stdscr.addstr(y, col_pos, watermark_bytes_display)
