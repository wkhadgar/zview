# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Check that a target runs the image of the ELF it is inspected with."""

import sys

from elftools.elf.elffile import ELFFile

from backend.base import AbstractScraper, ProbeError

_SPAN_BYTES = 64


class ImageMismatch(Exception):
    """Raised when the target's memory does not hold the ELF's image."""


def elf_span(elf_path) -> tuple[int, bytes] | None:
    """``(address, bytes)`` of the span the ELF holds at its entry point, if any."""
    with open(elf_path, "rb") as file:
        elf = ELFFile(file)
        # Cortex-M entry points carry the Thumb bit.
        entry = elf.header.e_entry & ~1

        for segment in elf.iter_segments():
            header = segment.header
            if header.p_type != "PT_LOAD":
                continue
            offset = entry - header.p_vaddr
            if offset < 0 or offset + _SPAN_BYTES > header.p_filesz:
                continue
            return entry, segment.data()[offset : offset + _SPAN_BYTES]

    return None


def verify_target_image(scraper: AbstractScraper, elf_path) -> None:
    """
    Raise ``ImageMismatch`` when the target's memory differs from the ELF's span.

    An unreadable span is reported on stderr and left at that.
    """
    span = elf_span(elf_path)
    if span is None:
        return

    address, expected = span
    scraper.begin_batch()
    try:
        found = bytes(scraper.read_bytes(address, len(expected)))
    except ProbeError as error:
        print(
            f"Could not read this ELF's entry point at 0x{address:08X}, so the image on "
            f"the target is unchecked ({error}).",
            file=sys.stderr,
        )
        return
    finally:
        scraper.end_batch()

    if found != expected:
        differing = sum(a != b for a, b in zip(found, expected, strict=True))
        raise ImageMismatch(
            f"the target is not running this ELF ({differing} of the {len(expected)} bytes "
            f"at its entry point 0x{address:08X} differ)\n"
            "Flash the build you are inspecting, or point -e at the image the target runs."
        )
