# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import curses

from backend.base import MemSlabInfo
from frontend.tui.views.kernel_object_detail import KernelObjectDetailView
from frontend.tui.widgets import MEM_SLAB, TUIGraph


class MemSlabDetailView(KernelObjectDetailView):
    """
    One memory slab: blocks in use, block size, the peak where the build
    tracks it, a usage graph and the wait queue.

    The graph plots one sample per polled frame.
    """

    # The title row already carries used over capacity and the waiters, so the
    # boxes add what it cannot show.
    _INFO_TITLES = (("Location", 2), ("Block size", 1), ("Peak", 1))
    _GRAPH_ROW = KernelObjectDetailView._INFO_ROW + KernelObjectDetailView._INFO_BOX_HEIGHT
    _UNTRACKED = "not tracked"

    def _target(self) -> MemSlabInfo | None:
        address = self.controller.detailing_mem_slab_address
        if address is None:
            return None

        return next((s for s in self.controller.mem_slabs_data if s.address == address), None)

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()
        self._render_frame(stdscr, self._footer_hint(), height, width)

        slab = self._target()
        if slab is None:
            stdscr.addstr(1, 0, " Slab is no longer being reported."[:width], self._error_attr)
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        self._draw_title(stdscr, width, MEM_SLAB, slab)
        self._draw_info_boxes(
            stdscr,
            self._INFO_ROW,
            width,
            [
                self._site(slab.address),
                f"{slab.block_size} B ({slab.total_bytes} B total)",
                self._peak_text(slab),
            ],
        )

        panels = self._panels(height, width, self._GRAPH_ROW)
        self._draw_usage_graph(stdscr, panels.graph_w, slab, panels.graph_h)
        self._draw_wait_queue(stdscr, panels, slab.waiters)

        self._render_status(stdscr, width, height - 2)
        stdscr.refresh()

    def _peak_text(self, slab: MemSlabInfo) -> str:
        if slab.max_used is None:
            return self._UNTRACKED

        return f"{slab.max_used}/{slab.num_blocks} blocks"

    def _draw_usage_graph(
        self, stdscr: curses.window, width: int, slab: MemSlabInfo, graph_height: int
    ) -> None:
        # Red with every block out, yellow while a thread waits for one, plain
        # otherwise.
        if slab.is_exhausted:
            attr = self._contended_attr
        else:
            attr = self._busy_attr if slab.waiters else 0

        caption = f"0 to {slab.num_blocks} blocks"
        if slab.max_used is not None:
            caption += f", peak {slab.max_used}"

        # Built per frame: the y scale is the slab's own block count.
        graph = TUIGraph("Blocks in use", caption, (0, slab.num_blocks or 1), attr)
        # One column per sample: extra points are averaged into moving buckets.
        columns = max(1, width - 2)
        history = list(self.controller.mem_slab_history.get(slab.address, ()))[-columns:]

        graph.draw(stdscr, self._GRAPH_ROW, 0, graph_height, width, points=history)
