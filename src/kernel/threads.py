# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr kernel thread list walker."""

from typing import Literal

from backend.base import AbstractScraper, ThreadInfo
from backend.elf_inspector import ElfInspector


def walk_thread_list(
    scraper: AbstractScraper,
    elf: ElfInspector,
    threads_head_address: int,
    offsets: dict,
    endianess: Literal["little", "big"],
    has_names: bool,
    max_threads: int = 64,
) -> dict[str, ThreadInfo]:
    """
    Walk the kernel thread linked list starting at ``threads_head_address``
    and return a ``{name: ThreadInfo}`` map. Batches the entire walk via the
    scraper's ``begin_batch``/``end_batch`` hooks. Raises ``RuntimeError``
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
    next_ptr_word_idx = offsets["k_thread"]["next_thread"] // 4
    stack_start_word_idx = offsets["thread_info"]["stack_start"] // 4
    name_word_idx = offsets["k_thread"]["name"] // 4 if has_names else 0
    stack_size_word_idx = offsets["thread_info"]["stack_size"] // 4

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
            thread_struct_words[stack_start_word_idx],
            thread_struct_words[stack_size_word_idx],
            thread_name,
            None,
        )

        thread_ptr = thread_struct_words[next_ptr_word_idx]

    scraper.end_batch()
    return threads
