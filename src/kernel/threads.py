# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr kernel thread walkers."""

from typing import Literal

from backend.base import AbstractScraper, ThreadInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout

# thread_state bits of an object that is not a live thread: the placeholder
# the kernel switches away from at boot, and a thread that has exited.
_THREAD_DUMMY = 1 << 0
_THREAD_DEAD = 1 << 3

# K_THREAD_DEFINE names its object after the thread's id.
_DEFINED_THREAD_PREFIX = "_k_thread_obj_"


def _word_at(words, offset: int | None) -> int | None:
    """The struct word at byte ``offset``, or None when the build has no such member."""
    return None if offset is None else words[offset // 4]


def _byte_at(words, offset: int, endianess: Literal["little", "big"]) -> int:
    """The struct byte at ``offset``."""
    return words[offset // 4].to_bytes(4, endianess)[offset % 4]


def _name_at(words, offset: int, endianess: Literal["little", "big"]) -> str:
    """The NUL-terminated string starting at byte ``offset`` of the struct."""
    raw = b"".join(w.to_bytes(4, endianess) for w in words[offset // 4 :])
    return raw.split(b"\0", 1)[0].decode(errors="ignore")


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

    threads: dict[str, ThreadInfo] = {}
    for _ in range(max_threads):
        if thread_ptr == 0:
            break

        try:
            thread_struct_words = scraper.read32(thread_ptr, words_to_read)
        except Exception as e:
            raise Exception(f"Error reading thread struct at 0x{thread_ptr:X}") from e

        if has_names:
            thread_name = _name_at(thread_struct_words, layout.thread_name, endianess)
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


def walk_static_threads(
    scraper: AbstractScraper,
    elf: ElfInspector,
    objects: dict[str, list[int]],
    layout: KernelLayout,
    endianess: Literal["little", "big"],
    has_names: bool,
    max_threads: int = 64,
) -> dict[str, ThreadInfo]:
    """
    Read the statically allocated ``k_thread`` objects, ``{path: addresses}``, and
    return the live ones as ``{name: ThreadInfo}``, for a build without the
    kernel's thread list. Wrapped in a single ``begin_batch``/``end_batch``.

    An object still all zero was never created. A thread the build does not name
    is named after its object.
    """
    if not scraper.is_connected:
        scraper.connect()
    scraper.begin_batch()

    words_to_read = elf.get_struct_size("k_thread") // 4
    threads: dict[str, ThreadInfo] = {}
    for path, address in ((p, a) for p, addresses in objects.items() for a in addresses):
        if len(threads) == max_threads:
            break

        try:
            words = scraper.read32(address, words_to_read)
        except Exception as e:
            raise Exception(f"Error reading thread struct at 0x{address:X}") from e

        state = (
            0 if layout.thread_state is None else _byte_at(words, layout.thread_state, endianess)
        )
        if not any(words) or state & (_THREAD_DUMMY | _THREAD_DEAD):
            continue

        name = _name_at(words, layout.thread_name, endianess) if has_names else ""
        name = name or path.removeprefix(_DEFINED_THREAD_PREFIX)
        if name in threads:
            name = f"{name} @ 0x{address:X}"

        threads[name] = ThreadInfo(
            address,
            _word_at(words, layout.stack_start),
            _word_at(words, layout.stack_size),
            name,
            None,
        )

    scraper.end_batch()
    return threads
