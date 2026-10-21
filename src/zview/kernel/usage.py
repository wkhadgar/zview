# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr thread usage counters, read as of one instant."""

from collections.abc import Sequence

from zview.backend.base import AbstractScraper

# Passes over the counters before keeping one that a stamp moved across.
_ATTEMPTS = 2


def read_usage_snapshot(
    scraper: AbstractScraper,
    threads: Sequence[int],
    usage_offset: int,
    stamps: Sequence[int],
) -> dict[int, int]:
    """
    Every thread's usage counter, ``{thread address: cycles}``.

    The counters are read back to back between two reads of each CPU's switch
    stamp, at ``stamps``, all in one ``read_many``. The kernel rewrites a stamp
    whenever it adds to a counter, so stamps that did not move mean the
    counters hold as of one instant. A pass a stamp moved across is tried once
    more, and the second pass is kept either way.
    """
    stamp_reads = [("read32", stamp, 1) for stamp in stamps]
    counter_reads = [("read64", thread + usage_offset, 1) for thread in threads]
    after_counters = len(stamps) + len(threads)

    for _ in range(_ATTEMPTS):
        values = [r[0] for r in scraper.read_many(stamp_reads + counter_reads + stamp_reads)]
        counters = dict(zip(threads, values[len(stamps) : after_counters], strict=True))
        if values[after_counters:] == values[: len(stamps)]:
            break

    return counters
