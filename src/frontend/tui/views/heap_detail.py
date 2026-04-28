# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.views.heap_list import HeapListView
from frontend.tui.widgets import TUIBox, TUIHeapInfo


class HeapDetailView(BaseStateView):
    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)
        self._scheme = HeapListView.SCHEMA
        self._graph_a_attr = theme.GRAPH_A
        self._frag_map_frame: TUIBox = TUIBox("Fragmentation Map", "", theme.GRAPH_B)

        bar_theme = (theme.PROGRESS_BAR_LOW, theme.PROGRESS_BAR_MEDIUM, theme.PROGRESS_BAR_HIGH)
        self._tui_heap_info: TUIHeapInfo = TUIHeapInfo(theme.CURSOR, theme.ACTIVE, bar_theme)
        self._tui_heap_info.set_field_widths(
            HeapListView.COLLUM_WIDTHS[0],
            HeapListView.COLLUM_WIDTHS[1],
            HeapListView.COLLUM_WIDTHS[2],
            HeapListView.COLLUM_WIDTHS[3],
            HeapListView.COLLUM_WIDTHS[4],
        )

    @staticmethod
    def get_sparsity_map(chunks: list[dict], width: int, height: int) -> list[str]:
        total_chars = width * height
        if not chunks or total_chars <= 0:
            return []

        total_bytes = sum(chunk["size"] for chunk in chunks)
        if total_bytes == 0:
            return []

        bytes_per_char = total_bytes / total_chars
        output = []
        chunk_idx = 0
        chunk_rem = float(chunks[0]["size"])
        chunk_is_used = chunks[0]["used"]

        for _ in range(total_chars):
            bucket_used = 0.0
            bucket_rem = bytes_per_char

            while bucket_rem > 0 and chunk_idx < len(chunks):
                take = min(chunk_rem, bucket_rem)
                if chunk_is_used:
                    bucket_used += take
                chunk_rem -= take
                bucket_rem -= take
                if chunk_rem <= 0:
                    chunk_idx += 1
                    if chunk_idx < len(chunks):
                        chunk_rem = float(chunks[chunk_idx]["size"])
                        chunk_is_used = chunks[chunk_idx]["used"]

            ratio = bucket_used / bytes_per_char
            if ratio == 0:
                output.append(" ")
            elif ratio <= 0.33:
                output.append("░")
            elif ratio <= 0.66:
                output.append("▒")
            elif ratio <= 0.99:
                output.append("▓")
            else:
                output.append("█")

        return ["".join(output[i : i + width]) for i in range(0, len(output), width)]

    @staticmethod
    def _get_fragmentation_metrics(chunks: list[dict]) -> dict:
        if not chunks:
            return {}
        total_chunks = len(chunks)
        allocated_chunks = sum(1 for c in chunks if c["used"])
        free_bytes = sum(c["size"] for c in chunks if not c["used"])
        largest_free = max((c["size"] for c in chunks if not c["used"]), default=0)
        ratio = (1 - largest_free / free_bytes) * 100 if free_bytes > 0 else 0.0
        return {
            "Largest free": (largest_free, "bytes"),
            "Frag ratio": (ratio, "percent"),
            "Chunks": (f"{allocated_chunks}/{total_chunks}", "raw"),
        }

    @staticmethod
    def _get_heap_details_footer(metrics: dict):
        if not metrics:
            return ""

        def fmt(value, hint):
            if hint == "bytes":
                return f"{value / 1024:.1f} KB" if value >= 1024 else f"{value} B"
            if hint == "percent":
                return f"{value:.1f}%"
            return str(value)

        return " · ".join([f"{k}: {fmt(v, h)}" for k, (v, h) in metrics.items()])

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()

        self._render_frame(stdscr, self._footer_hint(), height, width)

        curr_x = 0
        for col_header, h_width in self._scheme.items():
            if curr_x >= width:
                break

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        heap = next(
            (
                h
                for h in self.controller.heaps_data
                if h.address == self.controller.detailing_heap_address
            ),
            None,
        )

        if not heap or not heap.chunks:
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._tui_heap_info.draw(stdscr, 2, 0, heap, False)

        start_y = 5
        start_x = 1
        map_height = height - start_y - 4
        map_width = width - start_x - 1

        if map_height > 0 and map_width > 0:
            sparsity_matrix = self.get_sparsity_map(heap.chunks, map_width, map_height)
            metrics = self._get_fragmentation_metrics(heap.chunks)
            desc = self._get_heap_details_footer(metrics)

            self._frag_map_frame._description = desc
            self._frag_map_frame.draw(
                stdscr,
                start_y - 1,
                start_x - 1,
                map_height + 2,
                map_width + 2,
            )

            for i, row_str in enumerate(sparsity_matrix):
                stdscr.addstr(start_y + i, start_x, row_str, self._graph_a_attr)

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def keybindings(self) -> list[Keybind]:
        return [Keybind("<Enter>", "Back", "Return to the heap list")]

    def handle_input(self, key: int) -> ZViewState | None:
        if key in (curses.KEY_ENTER, SpecialCode.NEWLINE, SpecialCode.RETURN):
            return ZViewState.HEAP_LIST_VIEW
        elif key == SpecialCode.QUIT:
            self.controller.running = False
        return None
