# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Which groups of a frame, and which of their rows, are read."""

from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True)
class Selection:
    """
    The groups a frame reads, narrowed by name.

    A row is read when its group is in ``groups`` and not in ``skipped``, its
    name matches an ``include`` pattern or none are given, it matches no
    ``exclude`` pattern, and it is not ``muted``. Patterns are shell globs.
    ``muted`` holds exact ``(group, name)`` pairs, since a name such as
    ``z_idle_threads[0]`` reads as a glob.
    """

    groups: frozenset[str] | None = None
    skipped: frozenset[str] = frozenset()
    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    muted: frozenset[tuple[str, str]] = frozenset()

    def reads_group(self, group: str) -> bool:
        return (self.groups is None or group in self.groups) and group not in self.skipped

    def reads(self, group: str, name: str) -> bool:
        if not self.reads_group(group) or (group, name) in self.muted:
            return False
        if self.include and not any(fnmatchcase(name, pattern) for pattern in self.include):
            return False
        return not any(fnmatchcase(name, pattern) for pattern in self.exclude)

    def keep(self, frame: dict) -> dict:
        """``frame`` with only the groups and rows this selection reads."""
        return {
            group: [row for row in rows if self.reads(group, row.name)]
            for group, rows in frame.items()
            if self.reads_group(group)
        }
