# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from backend.base import AbstractScraper

FILL = 0xAAAAAAAA
USED = 0x12345678


class WatermarkScraper(AbstractScraper):
    """Minimal concrete AbstractScraper that scripts ``read32`` responses."""

    def __init__(self, responses: list[list[int]]):
        super().__init__(target_mcu=None)
        self._responses = list(responses)
        self.calls: list[tuple[int, int]] = []

    def connect(self):
        self._is_connected = True

    def disconnect(self):
        self._is_connected = False

    def read_bytes(self, at, amount):
        return b""

    def read8(self, at, amount=1):
        return []

    def read32(self, at, amount=1):
        self.calls.append((at, amount))
        return self._responses.pop(0)[:amount]

    def read64(self, at, amount=1):
        return []


def test_zero_size_short_circuits():
    scraper = WatermarkScraper([])
    assert scraper.calculate_dynamic_watermark(0x1000, 0, thread_id=1) == 0
    assert scraper.calls == []
    assert 1 not in scraper.watermark_cache


def test_pristine_stack_reports_zero():
    scraper = WatermarkScraper([[FILL] * 8])
    wm = scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1)
    assert wm == 0
    assert scraper.calls == [(0x1000, 8)]
    assert scraper.watermark_cache[1] == 0


def test_fully_used_stack_reports_full_size():
    scraper = WatermarkScraper([[USED] * 8])
    wm = scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1)
    assert wm == 32
    assert scraper.watermark_cache[1] == 32


def test_partial_usage_counts_fill_prefix():
    # 3 fill words at bottom, then used words -> 12 bytes unused, watermark 20.
    scraper = WatermarkScraper([[FILL, FILL, FILL, USED, USED, USED, USED, USED]])
    wm = scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1)
    assert wm == 20


def test_cache_hit_rereads_only_unused_portion():
    scraper = WatermarkScraper(
        [
            [FILL, FILL, FILL, USED, USED, USED, USED, USED],
            [FILL, FILL, FILL],
        ]
    )
    assert scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1) == 20
    assert scraper.calls[-1] == (0x1000, 8)

    # Second call re-reads only the low 3 words (still all fill -> watermark unchanged).
    assert scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1) == 20
    assert scraper.calls[-1] == (0x1000, 3)


def test_cache_hit_detects_growth():
    scraper = WatermarkScraper(
        [
            [FILL, FILL, FILL, USED, USED, USED, USED, USED],
            [USED, USED, USED],
        ]
    )
    assert scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1) == 20
    # Low 3 words now non-fill -> stack fully used.
    assert scraper.calculate_dynamic_watermark(0x1000, 32, thread_id=1) == 32


def test_independent_threads_track_separate_caches():
    scraper = WatermarkScraper(
        [
            [FILL, USED, USED, USED],
            [USED, USED, USED, USED],
        ]
    )
    assert scraper.calculate_dynamic_watermark(0x1000, 16, thread_id=1) == 12
    assert scraper.calculate_dynamic_watermark(0x2000, 16, thread_id=2) == 16
    assert scraper.watermark_cache == {1: 12, 2: 16}
