# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""The groups a frame reads, in the order a frame reads them."""

from zview.kernel.group import Group
from zview.kernel.heaps import Heaps
from zview.kernel.mem_slabs import MemSlabs
from zview.kernel.msgqs import Msgqs
from zview.kernel.mutexes import Mutexes
from zview.kernel.semaphores import Semaphores
from zview.kernel.threads import Threads
from zview.kernel.workqs import Workqs

# A recording holds the reads in this order, so a new group goes last.
GROUPS: tuple[type[Group], ...] = (
    Threads,
    Heaps,
    Semaphores,
    Mutexes,
    Msgqs,
    MemSlabs,
    Workqs,
)
