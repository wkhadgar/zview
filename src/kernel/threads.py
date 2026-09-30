# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr kernel thread list walker."""

from typing import Literal

from backend.base import AbstractScraper, ThreadInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout


def _word_at(words, offset: int | None) -> int | None:
    """The struct word at byte ``offset``, or None when the build has no such member."""
    return None if offset is None else words[offset // 4]


def walk_thread_list(
    scraper: AbstractScraper,
    elf: ElfInspector,
    threads_head_address: int,
    layout: KernelLayout,
    endianess: Literal["little", "big"],
    has_names: bool,
    max_threads: int = 64,
) -> dict[str, ThreadInfo]:
    """
    Walk the kernel thread linked list and return ``{name: ThreadInfo}``.
    Wrapped in a single ``begin_batch``/``end_batch``. Raises ``RuntimeError``
    when the head pointer cannot be read.
    """
    try:
        if not scraper.is_connected:
            scraper.connect()

        scraper.begin_batch()
        thread_ptr = scraper.read32(threads_head_address)[0] if threads_head_address else 0
    except Exception as e:
        scraper.end_batch()
        raise RuntimeError("Unable to read kernel thread list.") from e

    stack_struct_size = elf.get_struct_size("k_thread")
    words_to_read = stack_struct_size // 4
    next_ptr_word_idx = layout.thread_next // 4
    name_word_idx = (
        (layout.thread_name // 4) if (has_names and layout.thread_name is not None) else 0
    )

    threads: dict[str, ThreadInfo] = {}
    for _ in range(max_threads):
        if thread_ptr == 0:
            break

        try:
            thread_struct_words = scraper.read32(thread_ptr, words_to_read)
        except Exception as e:
            raise Exception(f"Error reading thread struct at 0x{thread_ptr:X}") from e

        if has_names:
            words = thread_struct_words[name_word_idx:]
            full_bytes = b''.join(w.to_bytes(4, endianess) for w in words)
            thread_name = full_bytes.split(b'\0', 1)[0].decode(errors="ignore")
        else:
            thread_name = f"thread @ 0x{thread_ptr:X}"

        threads[thread_name] = ThreadInfo(
            thread_ptr,
            _word_at(thread_struct_words, layout.stack_start),
            _word_at(thread_struct_words, layout.stack_size),
            thread_name,
            None,
        )

        thread_ptr = thread_struct_words[next_ptr_word_idx]

    scraper.end_batch()
    return threads
