# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Which groups of a frame, and which of their rows, are read."""

from dataclasses import dataclass
from fnmatch import fnmatchcase


def _shown(scope: str | None, glob: str) -> str:
    """A pattern as the flags write it."""
    return glob if scope is None else f"{scope}:{glob}"


@dataclass(frozen=True)
class Selection:
    """
    The groups a frame reads, narrowed by name.

    A row is read when its group is in ``groups`` and not in ``skipped``, its
    name matches an ``include`` pattern that applies to the group or none
    applies, it matches no ``exclude`` pattern that applies, and it is not
    ``muted``. Patterns are ``(group, glob)`` pairs, and a pattern with no
    group applies to every group. ``muted`` holds exact ``(group, name)``
    pairs, since a name such as ``z_idle_threads[0]`` reads as a glob.
    """

    groups: frozenset[str] | None = None
    skipped: frozenset[str] = frozenset()
    include: tuple[tuple[str | None, str], ...] = ()
    exclude: tuple[tuple[str | None, str], ...] = ()
    muted: frozenset[tuple[str, str]] = frozenset()

    def reads_group(self, group: str) -> bool:
        return (self.groups is None or group in self.groups) and group not in self.skipped

    def reads(self, group: str, name: str) -> bool:
        if not self.reads_group(group) or (group, name) in self.muted:
            return False
        includes = [pattern for scope, pattern in self.include if scope in (None, group)]
        if includes and not any(fnmatchcase(name, pattern) for pattern in includes):
            return False
        return not any(
            fnmatchcase(name, pattern) for scope, pattern in self.exclude if scope in (None, group)
        )

    def unmatched(self, names: dict[str, list[str]]) -> tuple[str, ...]:
        """
        One notice per pattern or muted row that names no row in ``names``,
        each group's row names, among the groups this selection reads.
        """
        read = {group: rows for group, rows in names.items() if self.reads_group(group)}

        def matches(scope: str | None, glob: str) -> bool:
            return any(
                fnmatchcase(name, glob)
                for group, rows in read.items()
                if scope in (None, group)
                for name in rows
            )

        notices = [
            f"Warning: {word} pattern {_shown(scope, glob)} matches no row"
            for word, patterns in (("include", self.include), ("exclude", self.exclude))
            for scope, glob in patterns
            if not matches(scope, glob)
        ]
        notices += [
            f"Warning: muted row {group}:{name} matches no row"
            for group, name in sorted(self.muted)
            if name not in read.get(group, ())
        ]
        return tuple(notices)

    def to_header(self) -> dict:
        """This selection as a recording header holds it."""
        return {
            "groups": None if self.groups is None else sorted(self.groups),
            "skipped": sorted(self.skipped),
            "include": [[scope, pattern] for scope, pattern in self.include],
            "exclude": [[scope, pattern] for scope, pattern in self.exclude],
            "muted": sorted([group, name] for group, name in self.muted),
        }

    @classmethod
    def from_header(cls, header: dict | None) -> "Selection":
        """The selection a recording header holds. A header without one read everything."""
        if not header:
            return cls()
        return cls(
            groups=None if header.get("groups") is None else frozenset(header["groups"]),
            skipped=frozenset(header.get("skipped", ())),
            include=tuple((scope, pattern) for scope, pattern in header.get("include", ())),
            exclude=tuple((scope, pattern) for scope, pattern in header.get("exclude", ())),
            muted=frozenset((group, name) for group, name in header.get("muted", ())),
        )

    def keep(self, frame: dict) -> dict:
        """``frame`` with only the groups and rows this selection reads."""
        return {
            group: [row for row in rows if self.reads(group, row.name)]
            for group, rows in frame.items()
            if self.reads_group(group)
        }
