# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for pure-logic helpers and bucketing inside ``frontend.tui.widgets``."""

import curses

import pytest

from frontend.tui.widgets import SiteValue, TUIGraph, TUIProgressBar, _addstr_clipped


@pytest.fixture
def graph() -> TUIGraph:
    """``TUIGraph`` doesn't touch curses at construction; safe to instantiate."""
    return TUIGraph(title="t", description="d", limits=(0, 100), attribute=0)


def test_process_points_pads_when_fewer_than_target(graph):
    """Short series gets zero-padded on the LEFT to reach ``target_len``."""
    out = graph._process_points([5, 10, 15], target_len=6)
    assert out == [0.0, 0.0, 0.0, 5, 10, 15]


def test_process_points_passes_through_when_already_at_target(graph):
    """``n == target_len`` triggers bucketing with bucket size 1: identity."""
    out = graph._process_points([10, 20, 30, 40], target_len=4)
    assert out == [10, 20, 30, 40]


def test_process_points_buckets_when_longer_than_target(graph):
    """8 points down to 4 buckets: each bucket averages 2 points."""
    out = graph._process_points([2, 4, 6, 8, 10, 12, 14, 16], target_len=4)
    # Buckets: (2,4)=3, (6,8)=7, (10,12)=11, (14,16)=15
    assert out == [3, 7, 11, 15]


def test_process_points_empty_returns_zero_padded(graph):
    out = graph._process_points([], target_len=3)
    assert out == [0.0, 0.0, 0.0]


class _FakeStdscr:
    """Minimal stdscr stand-in that records ``addstr`` calls."""

    def __init__(self):
        self.calls: list[tuple[int, int, str, int | None]] = []

    def addstr(self, y: int, x: int, text: str, attr: int | None = None):
        self.calls.append((y, x, text, attr))


def test_addstr_clipped_skips_when_no_room():
    stdscr = _FakeStdscr()
    _addstr_clipped(stdscr, 0, 80, "hello", screen_w=80)
    assert stdscr.calls == []


def test_addstr_clipped_truncates_to_remaining_width():
    """Text longer than ``screen_w - x`` is sliced before being drawn."""
    stdscr = _FakeStdscr()
    _addstr_clipped(stdscr, 0, 75, "this_is_a_long_label", screen_w=80)
    assert len(stdscr.calls) == 1
    y, x, text, attr = stdscr.calls[0]
    assert (y, x) == (0, 75)
    assert text == "this_"  # 80 - 75 = 5 chars allowed
    assert attr is None


def test_addstr_clipped_passes_through_attr_when_provided():
    stdscr = _FakeStdscr()
    _addstr_clipped(stdscr, 1, 0, "hi", screen_w=80, attr=42)
    assert stdscr.calls == [(1, 0, "hi", 42)]


def test_addstr_clipped_swallows_curses_error():
    """``curses.error`` from the terminal cell-edge quirk must not propagate."""

    class _ExplodingStdscr:
        def addstr(self, *_a, **_kw):
            raise curses.error("end of line")

    # No assertion needed; test passes if no exception bubbles up.
    _addstr_clipped(_ExplodingStdscr(), 0, 0, "x", screen_w=80)


class _FakeBarStdscr(_FakeStdscr):
    """``_FakeStdscr`` that also accepts the attribute calls a bar makes."""

    def attron(self, _attr: int) -> None:
        pass

    def attroff(self, _attr: int) -> None:
        pass


def _bar_blocks(percentage: float) -> tuple[int, int]:
    """``(filled cells, rightmost column touched)`` for a 12-wide bar."""
    stdscr = _FakeBarStdscr()
    bar = TUIProgressBar(12, 0, (101.0, 0), (102.0, 0))
    bar.draw(stdscr, 0, 0, percentage, label="x", attr=0)

    blocks = max((len(text) for _, _, text, _ in stdscr.calls if set(text) == {"█"}), default=0)
    rightmost = max(x + len(text) for _, x, text, _ in stdscr.calls)
    return blocks, rightmost


def test_a_bar_fills_no_further_than_its_track():
    """A corrupted reading (a huge count over a small limit) stays inside the bar."""
    blocks, rightmost = _bar_blocks(3_000_000_000.0)
    assert blocks == 10  # width 12, minus the two edge characters
    assert rightmost <= 12


_SITE = ("/home/dev/ws/app/src/main.c", 264)


def test_a_site_cell_shows_three_path_parts_when_they_fit():
    assert SiteValue(_SITE, 0x20000190).render(40) == "app/src/main.c:264 (0x20000190)"


def test_a_site_cell_drops_path_parts_to_fit():
    assert SiteValue(_SITE, 0x20000190).render(29) == "src/main.c:264 (0x20000190)"
    assert SiteValue(_SITE, 0x20000190).render(25) == "main.c:264 (0x20000190)"


def test_a_site_cell_falls_back_to_the_address_alone():
    assert SiteValue(_SITE, 0x20000190).render(12) == "0x20000190"


def test_a_cell_without_a_site_is_the_address():
    assert SiteValue(None, 0x20000190).render(40) == "0x20000190"


def test_a_symbol_prefixes_the_cell():
    site = ("/home/dev/ws/zephyr/lib/os/thread_entry.c", 36)
    cell = SiteValue(site, 0x10001A49, "z_thread_entry")

    assert cell.render(60) == "z_thread_entry @ lib/os/thread_entry.c:36 (0x10001a49)"
    assert cell.render(30) == "z_thread_entry (0x10001a49)"
