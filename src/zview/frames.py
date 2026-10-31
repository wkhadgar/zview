# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What the polling thread queues: each frame it reads, and what stops or troubles it."""

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from zview.selection import Selection

# The frames an object's history keeps.
HISTORY_LENGTH = 256


@dataclass(frozen=True)
class Frame:
    """One poll: the rows of each group read, by group key, and when, in epoch seconds."""

    groups: dict[str, list]
    at: float = field(default_factory=time.time)


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
    """
    Each group's latest infos, and each object's info in every frame that read
    it. The poll's thread adds while frontends read.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, list] = {}
        self._latest_at: dict[str, float] = {}
        self._history: dict[tuple[str, int], deque] = {}

    def add(self, frame: Frame) -> None:
        """Take a frame's infos. A group the frame leaves out keeps its last ones."""
        with self._lock:
            self._latest.update(frame.groups)
            self._latest_at.update(dict.fromkeys(frame.groups, frame.at))
            for group, infos in frame.groups.items():
                for info in infos:
                    key = (group, info.address)
                    self._history.setdefault(key, deque(maxlen=HISTORY_LENGTH)).append(info)

    def latest(self, group: str, selection: Selection | None = None) -> list:
        """The group's infos from the last frame that read it, those ``selection`` reads."""
        selection = selection or Selection()
        with self._lock:
            return [i for i in self._latest.get(group, []) if selection.reads(group, i.name)]

    def latest_at(self, group: str) -> float | None:
        """When the last frame that read the group was read, or ``None`` if none has."""
        with self._lock:
            return self._latest_at.get(group)

    def of(self, group: str, address: int) -> list:
        """The info of the object at ``address`` in each frame that read it, oldest first."""
        with self._lock:
            return list(self._history.get((group, address), ()))
