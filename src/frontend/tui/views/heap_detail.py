# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import HeapInfo
from frontend.tui.views.base import Any, ZViewTUIAttributes
from frontend.tui.views.kernel_object_detail import KernelObjectDetailView, Panels
from frontend.tui.widgets import HEAP, TUIBox


class HeapDetailView(KernelObjectDetailView):
    """
    One heap: the bytes free and out, a map of its chunks, and the wait queue.

    The map lays the chunks out in address order and shades each cell by how
    much of the bytes it covers are in use.
    """

    _INFO_TITLES = (("Location", 2), ("Free", 1), ("Used", 1))
    _MAP_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT
    _UNCAPTURED = "chunk map not captured for this frame"

    def __init__(self, controller: Any, theme: ZViewTUIAttributes):
        super().__init__(controller, theme)
        self._map_title = "Fragmentation map"

    def _target(self) -> HeapInfo | None:
        address = self.controller.detailing_heap_address
        if address is None:
            return None

        return next((h for h in self.controller.heaps_data if h.address == address), None)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        heap = self._target()
        if heap is None:
            stdscr.addstr(1, 0, " Heap is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, HEAP, heap)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [self._site(heap.address), f"{heap.free_bytes} B", f"{heap.allocated_bytes} B"],
        )

        panels = self._panels(height, width, self._MAP_ROW)
        self._draw_map(stdscr, panels, heap)
        self._draw_wait_queue(stdscr, panels, heap.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _draw_map(self, stdscr: curses.window, panels: Panels, heap: HeapInfo) -> None:
        # Red with nothing free, yellow while a thread waits, plain otherwise.
        if heap.is_exhausted:
            attr = self._contended_attr
        else:
            attr = self._busy_attr if heap.waiters else 0

        inner_h = panels.graph_h - 2
        inner_w = panels.graph_w - 2

        metrics = self._get_fragmentation_metrics(heap.chunks or [])
        # A cell's worth depends on the box, so the footer names it.
        if heap.chunks and inner_h > 0 and inner_w > 0:
            chunks_cell = metrics.pop("Chunks", None)
            metrics["Cell"] = (sum(c["size"] for c in heap.chunks) / (inner_w * inner_h), "bytes")
            if chunks_cell is not None:
                metrics["Chunks"] = chunks_cell

        box = TUIBox(self._map_title, self._get_heap_details_footer(metrics), 0)
        box.draw(stdscr, self._MAP_ROW, 0, panels.graph_h, panels.graph_w)

        if inner_h <= 0 or inner_w <= 0:
            return

        if not heap.chunks:
            stdscr.addstr(self._MAP_ROW + 1, 1, self._UNCAPTURED[:inner_w], self._label_attr)
            return

        for idx, row in enumerate(self.get_sparsity_map(heap.chunks, inner_w, inner_h)):
            stdscr.addstr(self._MAP_ROW + 1 + idx, 1, row, attr)

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
            "Frag": (ratio, "percent"),
            "Chunks": (f"{allocated_chunks}/{total_chunks}", "raw"),
        }

    @staticmethod
    def _get_heap_details_footer(metrics: dict):
        if not metrics:
            return ""

        def fmt(value, hint):
            if hint == "bytes":
                if value >= 1024:
                    return f"{value / 1024:.1f} KB"

                return f"{value} B" if isinstance(value, int) else f"{value:.2f} B"
            if hint == "percent":
                return f"{value:.1f}%"
            return str(value)

        return " · ".join([f"{k}: {fmt(v, h)}" for k, (v, h) in metrics.items()])
