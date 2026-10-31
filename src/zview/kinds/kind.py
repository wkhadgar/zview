# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What each group's infos mean to whoever shows them, declared once per group."""

import dataclasses
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


class Level(enum.Enum):
    """How much an info's state asks for attention."""

    QUIET = enum.auto()
    # In use, or with threads queued on it.
    BUSY = enum.auto()
    # Turning callers away: contended, full, exhausted or refusing.
    BLOCKED = enum.auto()


@dataclass(frozen=True)
class State:
    """An info at a glance: how full it is, a caption, and its level."""

    fill: float
    text: str
    level: Level


def level(*, is_blocked: bool = False, is_busy: bool = False) -> Level:
    """The first level that applies."""
    return next(
        lvl
        for applies, lvl in (
            (is_blocked, Level.BLOCKED),
            (is_busy, Level.BUSY),
            (True, Level.QUIET),
        )
        if applies
    )


def waiters_text(waiters: tuple[str, ...] | None) -> str:
    """A wait queue in a dump line: ``unknown`` when it was not walked, ``-`` when empty."""
    if waiters is None:
        return "unknown"

    return ",".join(waiters) if waiters else "-"


class Kind(ABC):
    """
    One group's infos as a reader sees them, keyed like its kernel ``Group``.

    ``label`` and ``title`` name the kind in a table and on its aggregate row,
    ``pressure`` is the word for an info that is ``BLOCKED``, ``state`` reads
    one info at a glance, and ``listed`` says whether a table lists it. A view
    lists the kinds its selection reads, so a kind no table lists leaves them
    unset.
    """

    key: ClassVar[str]
    label: ClassVar[str | None] = None
    title: ClassVar[str | None] = None
    pressure: ClassVar[str] = "contended"

    @abstractmethod
    def dump(self, info) -> str:
        """The info as one line of ``zview dump``."""

    def state(self, info) -> State | None:
        """The info at a glance, or ``None`` for a kind or an info without one."""
        return None

    def listed(self, info) -> bool:
        return True

    def to_json(self, info) -> dict:
        """
        The info's fields, then its dump line as ``line`` and its state, where
        it has one, as ``glance``: a thread already has a ``state`` field.
        """
        row = dataclasses.asdict(info)
        row["line"] = self.dump(info)
        if (state := self.state(info)) is not None:
            row["glance"] = {
                "fill": state.fill,
                "text": state.text,
                "level": state.level.name.lower(),
            }
        return row

    def in_use(self, info) -> bool:
        """Held or working, with or without threads queued on it."""
        return False
