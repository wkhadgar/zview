# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the ELF-versus-target image check."""

from pathlib import Path

import pytest

from backend.base import ProbeReadError
from backend.image import ImageMismatch, elf_span, verify_target_image


@pytest.fixture
def elf_path():
    return Path(__file__).parent / "fixtures" / "zephyr.elf"


class _FakeScraper:
    """Scraper stand-in serving fixed bytes, recording its batch bracketing."""

    def __init__(self, data: bytes | None = None, error: Exception | None = None):
        self._data = data
        self._error = error
        self.batches = 0
        self.ends = 0

    def begin_batch(self) -> None:
        self.batches += 1

    def end_batch(self) -> None:
        self.ends += 1

    def read_bytes(self, at: int, amount: int) -> bytes:
        del at
        if self._error is not None:
            raise self._error
        assert self._data is not None
        return self._data[:amount]


def test_the_span_sits_at_the_entry_point(elf_path):
    """The Thumb bit the entry point carries is not part of the address."""
    span = elf_span(elf_path)

    assert span is not None
    address, expected = span
    assert address == 0xA04  # the fixture's entry point is 0xA05
    assert len(expected) == 64
    assert any(expected)


def test_a_target_running_the_elf_passes(elf_path):
    _, expected = elf_span(elf_path)
    scraper = _FakeScraper(expected)

    verify_target_image(scraper, elf_path)

    assert (scraper.batches, scraper.ends) == (1, 1)


def test_a_target_running_another_image_is_refused(elf_path):
    _, expected = elf_span(elf_path)
    scraper = _FakeScraper(bytes(len(expected)))

    with pytest.raises(ImageMismatch):
        verify_target_image(scraper, elf_path)

    assert scraper.ends == 1


def test_a_failed_read_is_not_a_mismatch(elf_path):
    """A probe that cannot read the span says nothing about the image."""
    scraper = _FakeScraper(error=ProbeReadError("no answer"))

    verify_target_image(scraper, elf_path)

    assert scraper.ends == 1
