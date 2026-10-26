# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What the polling thread queues: each frame it reads, and what stops or troubles it."""

from collections import deque
from dataclasses import dataclass

from zview.selection import Selection

# The frames an object's history keeps.
HISTORY_LENGTH = 256


@dataclass(frozen=True)
class Frame:
    """One poll: the rows of each group read, by group key."""

    groups: dict[str, list]


@dataclass(frozen=True)
class Notice:
    """A fault the poll went on after, such as a read that failed once."""

    text: str


@dataclass(frozen=True)
class Fatal:
    """The poll stopped: the target never connected, or it was lost."""

    text: str


@dataclass(frozen=True)
class ReplayEnd:
    """The recording being replayed ended cleanly."""

    text: str


class FrameHistory:
    """Each group's latest infos, and each object's info in every frame that read it."""

    def __init__(self) -> None:
        self._latest: dict[str, list] = {}
        self._history: dict[tuple[str, int], deque] = {}
        self._named: dict[tuple[str, str], object] = {}

    def add(self, frame: Frame, muted: frozenset[tuple[str, str]] = frozenset()) -> None:
        """
        Take a frame's infos. A group the frame leaves out keeps its last ones,
        and a ``muted`` row keeps its last info and history.
        """
        self._latest.update(frame.groups)
        for group, infos in frame.groups.items():
            for info in (i for i in infos if (group, i.name) not in muted):
                key = (group, info.address)
                self._history.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(info)
                self._named[(group, info.name)] = info

    def latest(self, group: str, selection: Selection | None = None) -> list:
        """
        The group's infos from the last frame that read it, those ``selection``
        reads, then each row it mutes and its patterns match, as last read.
        """
        selection = selection or Selection()
        infos = [info for info in self._latest.get(group, []) if selection.reads(group, info.name)]
        muted = (
            self._named.get(row)
            for row in sorted(selection.muted)
            if row[0] == group and selection.matches(*row)
        )
        return infos + [info for info in muted if info is not None]

    def of(self, group: str, address: int) -> list:
        """The info of the object at ``address`` in each frame that read it, oldest first."""
        return list(self._history.get((group, address), ()))
