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
    """Single fully-used chunk must produce only ``█``."""
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
    assert metrics["Frag ratio"] == (0.0, "percent")
    assert metrics["Chunks"] == ("1/1", "raw")


def test_fragmentation_metrics_single_free_chunk_ratio_is_zero():
    """One free chunk means largest_free == free_bytes -> ratio is 0%."""
    chunks = [
        {"size": 100, "used": True},
        {"size": 100, "used": False},
    ]
    metrics = HeapDetailView._get_fragmentation_metrics(chunks)
    assert metrics["Largest free"] == (100, "bytes")
    assert metrics["Frag ratio"] == (0.0, "percent")
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
    assert metrics["Frag ratio"][0] == 50.0
    assert metrics["Chunks"] == ("1/3", "raw")


def test_heap_details_footer_empty_returns_empty_string():
    assert HeapDetailView._get_heap_details_footer({}) == ""


def test_heap_details_footer_formats_units():
    metrics = {
        "Largest free": (2048, "bytes"),
        "Frag ratio": (37.5, "percent"),
        "Chunks": ("1/3", "raw"),
    }
    out = HeapDetailView._get_heap_details_footer(metrics)
    assert "Largest free: 2.0 KB" in out
    assert "Frag ratio: 37.5%" in out
    assert "Chunks: 1/3" in out
    assert " · " in out


def test_heap_details_footer_bytes_under_1k_uses_b_unit():
    out = HeapDetailView._get_heap_details_footer({"x": (512, "bytes")})
    assert out == "x: 512 B"
