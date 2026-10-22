# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Every group's kind, in the order a frame reads the groups."""

from zview.kinds.heaps import Heaps
from zview.kinds.kind import Kind
from zview.kinds.mem_slabs import MemSlabs
from zview.kinds.msgqs import Msgqs
from zview.kinds.mutexes import Mutexes
from zview.kinds.semaphores import Semaphores
from zview.kinds.threads import Threads
from zview.kinds.workqs import Workqs

KINDS: tuple[Kind, ...] = (
    Threads(),
    Heaps(),
    Semaphores(),
    Mutexes(),
    Msgqs(),
    MemSlabs(),
    Workqs(),
)
BY_KEY: dict[str, Kind] = {kind.key: kind for kind in KINDS}
