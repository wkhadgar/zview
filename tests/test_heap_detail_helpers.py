# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``HeapDetailView`` pure-function helpers (no curses)."""

from frontend.tui.views.heap_detail import HeapDetailView


def test_sparsity_map_empty_chunks_returns_empty():
    assert HeapDetailView.get_sparsity_map([], 10, 4) == []


def test_sparsity_map_zero_geometry_returns_empty():
    chunks = [{"size": 100, "used": True}]
    assert HeapDetailView.get_sparsity_map(chunks, 0, 4) == []
    assert HeapDetailView.get_sparsity_map(chunks, 10, 0) == []


def test_sparsity_map_zero_total_bytes_returns_empty():
    chunks = [{"size": 0, "used": True}]
    assert HeapDetailView.get_sparsity_map(chunks, 10, 1) == []


def test_sparsity_map_all_used_renders_full_blocks():
    """Single fully-used chunk must produce only the full-block glyph."""
    chunks = [{"size": 100, "used": True}]
    rows = HeapDetailView.get_sparsity_map(chunks, 10, 1)
    assert rows == ["█" * 10]


def test_sparsity_map_all_free_renders_blanks():
    chunks = [{"size": 100, "used": False}]
    rows = HeapDetailView.get_sparsity_map(chunks, 10, 1)
    assert rows == [" " * 10]


def test_sparsity_map_split_used_free_yields_mixed_glyphs():
    """Half used, half free: left half full, right half blank."""
    chunks = [
        {"size": 50, "used": True},
        {"size": 50, "used": False},
    ]
    rows = HeapDetailView.get_sparsity_map(chunks, 10, 1)
    line = rows[0]
    # Left half should be full blocks, right half blanks.
    assert line[:5] == "█" * 5
    assert line[5:] == " " * 5


def test_sparsity_map_returns_correct_row_count():
    """``height`` rows of ``width`` chars each."""
    chunks = [{"size": 200, "used": True}]
    rows = HeapDetailView.get_sparsity_map(chunks, 8, 3)
    assert len(rows) == 3
    assert all(len(r) == 8 for r in rows)


def test_fragmentation_metrics_empty_returns_empty_dict():
    assert HeapDetailView._get_fragmentation_metrics([]) == {}


def test_fragmentation_metrics_no_free_chunks_zero_ratio():
    chunks = [{"size": 100, "used": True}]
    metrics = HeapDetailView._get_fragmentation_metrics(chunks)
    assert metrics["Largest free"] == (0, "bytes")
    assert metrics["Frag"] == (0.0, "percent")
    assert metrics["Chunks"] == ("1/1", "raw")


def test_fragmentation_metrics_single_free_chunk_ratio_is_zero():
    """One free chunk means largest_free == free_bytes -> ratio is 0%."""
    chunks = [
        {"size": 100, "used": True},
        {"size": 100, "used": False},
    ]
    metrics = HeapDetailView._get_fragmentation_metrics(chunks)
    assert metrics["Largest free"] == (100, "bytes")
    assert metrics["Frag"] == (0.0, "percent")
    assert metrics["Chunks"] == ("1/2", "raw")


def test_fragmentation_metrics_split_free_chunks_ratio_nonzero():
    """Two equal free chunks: largest is half of free total -> 50% fragmented."""
    chunks = [
        {"size": 100, "used": True},
        {"size": 50, "used": False},
        {"size": 50, "used": False},
    ]
    metrics = HeapDetailView._get_fragmentation_metrics(chunks)
    assert metrics["Largest free"] == (50, "bytes")
    assert metrics["Frag"][0] == 50.0
    assert metrics["Chunks"] == ("1/3", "raw")


def test_heap_details_footer_empty_returns_empty_string():
    assert HeapDetailView._get_heap_details_footer({}) == ""


def test_heap_details_footer_formats_units():
    metrics = {
        "Largest free": (2048, "bytes"),
        "Frag": (37.5, "percent"),
        "Chunks": ("1/3", "raw"),
    }
    out = HeapDetailView._get_heap_details_footer(metrics)
    assert "Largest free: 2.0 KB" in out
    assert "Frag: 37.5%" in out
    assert "Chunks: 1/3" in out
    assert " · " in out


def test_heap_details_footer_bytes_under_1k_uses_b_unit():
    out = HeapDetailView._get_heap_details_footer({"x": (512, "bytes")})
    assert out == "x: 512 B"


class _Theme:
    """Distinct attributes, so a color assertion means something."""

    ACTIVE = 1
    INACTIVE = 2
    PROGRESS_BAR_LOW = 3
    PROGRESS_BAR_MEDIUM = 4
    PROGRESS_BAR_HIGH = 5
    HEADER_FOOTER = 6
    ERROR = 7
    CURSOR = 8
    GRAPH_A = 9
    GRAPH_B = 10


class _StubWin:
    """curses window stand-in recording ``(row, col, text, attr)`` writes."""

    def __init__(self, height: int = 24, width: int = 209):
        self._h, self._w = height, width
        self.writes: list[tuple[int, int, str, int]] = []

    def getmaxyx(self):
        return self._h, self._w

    def addstr(self, y: int, x: int, text: str, attr: int = 0):
        assert 0 <= y < self._h, f"row {y} outside height {self._h}"
        assert x + len(text) <= self._w, f"row {y} overruns width: {x}+{len(text)} > {self._w}"
        self.writes.append((y, x, text, attr))

    def erase(self): ...
    def refresh(self): ...
    def attron(self, attr): ...
    def attroff(self, attr): ...
    def move(self, y, x): ...
    def clrtoeol(self): ...
    def hline(self, y, x, ch, n): ...

    def getbkgd(self):
        return 0


def _heap_controller(waiters=("worker",), used_chunks=8, free=256):
    """Controller holding one heap under the cursor, with a walkable chunk map."""
    from unittest.mock import MagicMock

    from backend.base import HeapInfo

    chunks = [{"used": i % 2 == 0, "size": 64} for i in range(used_chunks)]
    heap = HeapInfo(
        name="bench_heap",
        address=0x5000,
        free_bytes=free,
        allocated_bytes=256,
        max_allocated_bytes=320,
        usage_percent=50.0,
        chunks=chunks,
        waiters=waiters,
    )
    controller = MagicMock()
    controller.scraper.decl_site.return_value = None
    controller.heaps_data = [heap]
    controller.detailing_heap_address = 0x5000
    controller.status_message = "Running"
    return controller


def _render(controller, height: int = 24, width: int = 209) -> _StubWin:
    view = HeapDetailView(controller, _Theme())
    win = _StubWin(height, width)
    view.render(win, height, width)
    return win, view


def test_a_heap_draws_its_wait_queue_beside_the_map():
    win, view = _render(_heap_controller())

    queue_col = 209 - view._QUEUE_WIDTH
    titles = [(y, x) for y, x, text, _ in win.writes if text.startswith("┌Wait queue")]
    assert titles and all(x == queue_col for _, x in titles)
    assert any(x == queue_col + 2 and "worker" in text for _, x, text, _ in win.writes)


def test_a_narrow_terminal_puts_the_heap_queue_under_the_map():
    win, _ = _render(_heap_controller(), width=70)

    titles = [(y, x) for y, x, text, _ in win.writes if text.startswith("┌Wait queue")]
    assert titles and all(x == 0 for _, x in titles)


def _map_attrs(win: _StubWin) -> set[int]:
    return {attr for _, x, text, attr in win.writes if x == 1 and set(text) <= set(" ░▒▓█")}


def test_a_quiet_heap_draws_its_map_plain():
    win, _ = _render(_heap_controller(waiters=()))

    assert _map_attrs(win) == {0}


def test_a_thread_waiting_for_memory_colors_the_map_busy():
    win, _ = _render(_heap_controller(waiters=("worker",)))

    assert _map_attrs(win) == {_Theme.PROGRESS_BAR_MEDIUM}


def test_a_heap_with_nothing_free_colors_the_map_contended():
    win, _ = _render(_heap_controller(waiters=(), free=0))

    assert _map_attrs(win) == {_Theme.ERROR}


def _footer(win: _StubWin) -> str:
    return next(text for _, _, text, _ in win.writes if text.startswith("└Largest free"))


def test_the_map_names_what_a_cell_is_worth():
    """The map scales to the box, so the scale cannot be left implied."""
    win, _ = _render(_heap_controller())

    assert "Cell:" in _footer(win)


def test_a_cell_worth_less_than_a_byte_still_reads_as_a_scale():
    """128 bytes over a 177x14 map: a cell covers a twentieth of a byte."""
    win, _ = _render(_heap_controller(used_chunks=2))

    assert "Cell: 0.05 B" in _footer(win)
