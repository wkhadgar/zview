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
    groups: the wait queue layout and every group's discovered ``rows``, as
    ``{key: {name: addresses}}``. ``parts`` are the groups' parts that are on.
    """

    scraper: AbstractScraper
    elf: ElfInspector
    layout: KernelLayout
    waitq_flavor: WaitQFlavor = "simple"
    rows: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    parts: frozenset[str] = frozenset()

    @property
    def is_waitq_walkable(self) -> bool:
        """Wait queues are walkable only in the dlist (simple) flavor."""
        return self.waitq_flavor == "simple"

    @property
    def thread_names(self) -> dict[int, str]:
        """``{address: name}`` of the threads a wait queue can hold."""
        return {
            address: name
            for name, addresses in self.rows.get("threads", {}).items()
            for address in addresses
        }

    def has_part(self, part: str) -> bool:
        return part in self.parts
