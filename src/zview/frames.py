# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""What the polling thread queues: each frame it reads, and what stops or troubles it."""

from dataclasses import dataclass


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
