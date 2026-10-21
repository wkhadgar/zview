# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What the polling thread queues: each frame it reads, and what stops or troubles it."""

from collections import deque
from dataclasses import dataclass

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

    def add(self, frame: Frame) -> None:
        """Take a frame's infos. A group the frame leaves out keeps its last ones."""
        self._latest.update(frame.groups)
        for group, infos in frame.groups.items():
            for info in infos:
                key = (group, info.address)
                self._history.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(info)

    def latest(self, group: str) -> list:
        return self._latest.get(group, [])

    def of(self, group: str, address: int) -> list:
        """The info of the object at ``address`` in each frame that read it, oldest first."""
        return list(self._history.get((group, address), ()))
