# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""The base each group a frame reads is declared from."""

from abc import ABC, abstractmethod
from typing import ClassVar

from zview.kernel.compat import FieldTable
from zview.kernel.session import Session


class BrokenFrame(Exception):
    """A walk read values that cannot be of one instant, such as a counter that went back."""


class Group(ABC):
    """
    A kind of kernel object a frame reads, declared in its own module.

    ``key`` names it in a frame, a recording's features and a selection. A
    table resolves all of its fields or none: the group is read only when every
    ``required`` table resolves, and each ``optional`` one resolves on its own,
    so independent fields go in tables of their own. Each of its ``parts``
    turns on some of its reads: those ``live_parts`` names on live, and in a
    replay only those the recording names too. ``discover`` finds its rows
    after connect, and ``walk`` reads them every frame as ``{label: row}``. One
    instance lives per ELF load, so what a group keeps across frames is its own.
    """

    key: ClassVar[str]
    struct: ClassVar[str]
    required: ClassVar[tuple[FieldTable, ...]] = ()
    optional: ClassVar[tuple[FieldTable, ...]] = ()
    parts: ClassVar[tuple[str, ...]] = ()

    def capture_all(self) -> None:  # noqa: B027
        """Read every detail a view may ask for, as a recording holds them all."""

    def discover(self, session: Session) -> dict[str, list[int]]:
        """``{symbol: addresses}`` of the struct's statically declared instances."""
        return session.elf.find_struct_instances(self.struct)

    def live_parts(self, session: Session) -> frozenset[str]:
        """The parts a live session turns on: all of them."""
        return frozenset(self.parts)

    def notices(self, session: Session) -> tuple[str, ...]:
        """One notice per capability the build's Kconfig leaves out."""
        return ()

    def reset(self) -> None:  # noqa: B027
        """Forget what the walk keeps across frames, so the next frame seeds it again."""

    @abstractmethod
    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict:
        """Read the rows at ``addresses`` and return ``{label: info}``."""
