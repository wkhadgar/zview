# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_timer`` group."""

from zview.backend.base import TimerInfo
from zview.backend.elf_inspector import ElfInspector
from zview.kernel.compat import FieldTable
from zview.kernel.group import Group
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.structs import struct_words
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue

TIMER_FIELDS: FieldTable = {
    # The node's first word is its ``next``, set while the timeout is queued.
    "timer_timeout_node": ((("k_timer", "timeout"), ("_timeout", "node")),),
    "timer_wait_q": ((("k_timer", "wait_q"),),),
    "timer_expiry_fn": ((("k_timer", "expiry_fn"),),),
    "timer_stop_fn": ((("k_timer", "stop_fn"),),),
    # ``k_timeout_t`` holds only ``ticks``, at its offset 0.
    "timer_period": ((("k_timer", "period"),),),
    "timer_status": ((("k_timer", "status"),),),
}


def _function(
    elf: ElfInspector, pointer: int
) -> tuple[int | None, str | None, tuple[str, int] | None]:
    """A function pointer, the symbol it names and its site, all ``None`` for a NULL one."""
    if pointer == 0:
        return None, None, None
    # A Thumb code pointer has bit 0 set.
    return pointer, elf.get_symbol_name_at(pointer & ~1), elf.function_site(pointer)


class Timers(Group):
    key = "timers"
    struct = "k_timer"
    required = (TIMER_FIELDS,)

    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict[str, TimerInfo]:
        """
        Read every statically declared ``k_timer`` and return ``{label: TimerInfo}``.

        Each timer costs one bulk struct read plus one read per waiting thread.
        On a wait queue layout that cannot be walked (scalable), ``waiters`` is
        ``None``.
        """
        scraper, elf, layout = session.scraper, session.elf, session.layout
        walk_waiters, thread_names = session.is_waitq_walkable, session.thread_names
        words_to_read = struct_words(elf, "k_timer")

        timers: dict[str, TimerInfo] = {}
        for label, address in label_instances(addresses):
            words = scraper.read32(address, words_to_read)

            waiters: tuple[str, ...] | None = None
            if walk_waiters and layout.thread_qnode is not None:
                waiters = resolve_waiter_names(
                    walk_wait_queue(
                        scraper,
                        address + layout.timer_wait_q,
                        layout.thread_qnode,
                        head=words[layout.timer_wait_q // 4],
                    ),
                    thread_names,
                )

            expiry_fn, expiry_fn_symbol, expiry_fn_site = _function(
                elf, words[layout.timer_expiry_fn // 4]
            )
            stop_fn, stop_fn_symbol, stop_fn_site = _function(elf, words[layout.timer_stop_fn // 4])
            period = layout.timer_period // 4
            timers[label] = TimerInfo(
                name=label,
                address=address,
                is_running=words[layout.timer_timeout_node // 4] != 0,
                period_ticks=words[period] | (words[period + 1] << 32),
                status=words[layout.timer_status // 4],
                expiry_fn=expiry_fn,
                expiry_fn_symbol=expiry_fn_symbol,
                stop_fn=stop_fn,
                stop_fn_symbol=stop_fn_symbol,
                waiters=waiters,
                decl_site=elf.decl_site(address),
                expiry_fn_site=expiry_fn_site,
                stop_fn_site=stop_fn_site,
            )

        return timers
