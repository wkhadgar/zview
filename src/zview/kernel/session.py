# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What a group's discover and walk read with."""

from dataclasses import dataclass, field

from zview.backend.base import AbstractScraper
from zview.backend.elf_inspector import ElfInspector
from zview.kernel.compat import WaitQFlavor
from zview.kernel.layout import KernelLayout


@dataclass
class Session:
    """
    The probe, the ELF and its layout, and what one frame shares across
    groups: the wait queue layout and the names of the threads a wait queue
    can hold. ``parts`` are the groups' parts that are on.
    ``state`` keeps each group's own values across frames, and ``errors``
    collects the failures a walk reports without giving up.
    """

    scraper: AbstractScraper
    elf: ElfInspector
    layout: KernelLayout
    waitq_flavor: WaitQFlavor = "simple"
    thread_names: dict[int, str] = field(default_factory=dict)
    parts: frozenset[str] = frozenset()
    state: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def walks_waiters(self) -> bool:
        """Wait queues are walkable only in the dlist (simple) flavor."""
        return self.waitq_flavor == "simple"

    def has_part(self, part: str) -> bool:
        return part in self.parts
