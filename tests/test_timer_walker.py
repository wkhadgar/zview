# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.kernel.layout import KernelLayout
from zview.kernel.session import Session
from zview.kernel.timers import Timers


class FakeScraper:
    """Minimal AbstractScraper stand-in over a word-addressed memory map."""

    def __init__(self, words: dict[int, int]):
        self._words = words

    def read32(self, at: int, amount: int = 1):
        return tuple(self._words.get(at + 4 * i, 0) for i in range(amount))


class FakeElf:
    """Minimal ElfInspector stand-in naming the functions in ``symbols``."""

    def __init__(self, symbols: dict[int, str] | None = None):
        self._symbols = symbols or {}

    def get_struct_size(self, name: str) -> int:
        if name != "k_timer":
            raise LookupError(name)
        return 56

    def get_symbol_name_at(self, address: int) -> str | None:
        return self._symbols.get(address)

    def function_site(self, address: int) -> None:
        return None

    def decl_site(self, address: int) -> None:
        return None


# The bench's 56-byte k_timer: the timeout node at 0, wait_q at 24, expiry_fn at
# 32, stop_fn at 36, the 8-byte period at 40 and status at 48.
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    timer_timeout_node=0,
    timer_wait_q=24,
    timer_expiry_fn=32,
    timer_stop_fn=36,
    timer_period=40,
    timer_status=48,
    thread_qnode=0,
)


def _timer_words(address: int, fields: dict[int, int]) -> dict[int, int]:
    """One k_timer at ``address``, with ``{offset: word}`` set and every other word 0."""
    # An empty wait queue points at itself.
    words = {address + 24: address + 24, address + 28: address + 24}
    words.update({address + offset: value for offset, value in fields.items()})
    return words


def _walk(words: dict[int, int], elf: FakeElf | None = None) -> dict:
    session = Session(FakeScraper(words), elf or FakeElf(), LAYOUT)
    return Timers().walk(session, {"bench_tick": [0x2000]})


def test_a_queued_timeout_node_is_a_running_timer():
    """The node's ``next`` is set only while the timeout is queued."""
    stopped = _walk(_timer_words(0x2000, {}))["bench_tick"]
    running = _walk(_timer_words(0x2000, {0: 0x2000_0400}))["bench_tick"]

    assert (stopped.is_running, running.is_running) == (False, True)


def test_the_period_takes_both_words_of_its_ticks():
    """A 64-bit tick count past 32 bits would otherwise lose its high word."""
    info = _walk(_timer_words(0x2000, {40: 25, 44: 1}))["bench_tick"]

    assert info.period_ticks == (1 << 32) + 25
    assert not info.is_oneshot


def test_an_expiry_fn_is_named_without_its_thumb_bit():
    info = _walk(_timer_words(0x2000, {32: 0x6C9}), FakeElf({0x6C8: "tick_expiry"}))["bench_tick"]

    assert (info.expiry_fn, info.expiry_fn_symbol) == (0x6C9, "tick_expiry")


def test_a_null_stop_fn_has_no_symbol():
    info = _walk(_timer_words(0x2000, {}))["bench_tick"]

    assert (info.stop_fn, info.stop_fn_symbol) == (None, None)
