# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr thread usage counters, read as of one instant."""

from collections.abc import Sequence

from zview.backend.base import AbstractScraper


def read_usage_snapshot(
    scraper: AbstractScraper,
    threads: Sequence[int],
    usage_offset: int,
    stamps: Sequence[int],
    *,
    attempts: int = 1,
) -> tuple[dict[int, int], bool | None]:
    """
    Every thread's usage counter, ``{thread address: cycles}``, and whether
    they hold as of one instant, None without stamps to tell.

    The counters are read back to back between two reads of each CPU's switch
    stamp, at ``stamps``, all in one ``read_many``. The kernel rewrites a stamp
    on every switch, so stamps that did not move mean no switch fell between
    the counters. A pass a stamp moved across is made again, up to
    ``attempts`` passes, and the last one is kept.
    """
    stamp_reads = [("read32", stamp, 1) for stamp in stamps]
    counter_reads = [("read64", thread + usage_offset, 1) for thread in threads]
    after_counters = len(stamps) + len(threads)

    for _ in range(attempts):
        values = [r[0] for r in scraper.read_many(stamp_reads + counter_reads + stamp_reads)]
        counters = dict(zip(threads, values[len(stamps) : after_counters], strict=True))
        coherent = values[after_counters:] == values[: len(stamps)]
        if coherent:
            break

    return counters, coherent if stamps else None
