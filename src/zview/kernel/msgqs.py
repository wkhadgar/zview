# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_msgq`` group."""

from zview.backend.base import MsgqInfo
from zview.kernel.compat import FieldTable
from zview.kernel.group import Group
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.structs import struct_words
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue

MSGQ_FIELDS: FieldTable = {
    "msgq_wait_q": ((("k_msgq", "wait_q"),),),
    "msgq_msg_size": ((("k_msgq", "msg_size"),),),
    "msgq_max_msgs": ((("k_msgq", "max_msgs"),),),
    "msgq_used_msgs": ((("k_msgq", "used_msgs"),),),
}


class Msgqs(Group):
    key = "msgqs"
    struct = "k_msgq"
    required = (MSGQ_FIELDS,)

    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict[str, MsgqInfo]:
        """
        Read every statically declared ``k_msgq`` and return ``{label: MsgqInfo}``.

        ``addresses`` maps a symbol name to its instance addresses. Each queue
        costs one bulk struct read plus one read per waiting thread. On a wait
        queue layout that cannot be walked (scalable), ``waiters`` is ``None``.
        """
        scraper, elf, layout = session.scraper, session.elf, session.layout
        walk_waiters, thread_names = session.walks_waiters, session.thread_names
        words_to_read = struct_words(elf, "k_msgq")
        used_idx = layout.msgq_used_msgs // 4
        max_idx = layout.msgq_max_msgs // 4
        size_idx = layout.msgq_msg_size // 4
        wait_q_idx = layout.msgq_wait_q // 4

        msgqs: dict[str, MsgqInfo] = {}
        for label, address in label_instances(addresses):
            words = scraper.read32(address, words_to_read)

            waiters: tuple[str, ...] | None = None
            if walk_waiters and layout.thread_qnode is not None:
                waiters = resolve_waiter_names(
                    walk_wait_queue(
                        scraper,
                        address + layout.msgq_wait_q,
                        layout.thread_qnode,
                        head=words[wait_q_idx],
                    ),
                    thread_names,
                )

            msgqs[label] = MsgqInfo(
                name=label,
                address=address,
                used_msgs=words[used_idx],
                max_msgs=words[max_idx],
                msg_size=words[size_idx],
                waiters=waiters,
            )

        return msgqs
