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
    stamp, at ``stamps``. The kernel rewrites a stamp whenever it adds to a
    counter, so stamps that did not move mean the counters hold as of one
    instant. A pass a stamp moved across is tried once more, and the second
    pass is kept either way.
    """
    for _ in range(_ATTEMPTS):
        before = [scraper.read32(stamp)[0] for stamp in stamps]
        counters = {thread: scraper.read64(thread + usage_offset)[0] for thread in threads}
        if [scraper.read32(stamp)[0] for stamp in stamps] == before:
            break

    return counters
