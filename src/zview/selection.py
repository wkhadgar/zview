# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Which groups of a frame, and which of their rows, are read."""

from collections.abc import Iterable
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from glob import escape

Pattern = tuple[str | None, str]

# The patterns of every row of every group, read when no line is given.
EVERY_ROW: tuple[Pattern, ...] = ((None, "*"),)


def _parse_pattern(value: str) -> tuple[str | None, str, bool]:
    """
    ``(group, glob, excluded)`` of a bare ``GLOB`` for every group or of
    ``GROUP:GLOB`` for one, a ``!`` before the glob excluding it. Raises
    ``ValueError`` on ``!GROUP:GLOB``.
    """
    group, sep, glob = value.partition(":")
    scope, glob = (group, glob) if sep else (None, value)
    if scope is not None and scope.startswith("!"):
        raise ValueError(f"Write {scope[1:]}:!{glob} to exclude within one group, not {value}.")
    return scope, glob.removeprefix("!"), glob.startswith("!")


def _shown(scope: str | None, glob: str) -> str:
    """A pattern as the flags write it."""
    return glob if scope is None else f"{scope}:{glob}"


@dataclass(frozen=True)
class Selection:
    """
    The groups a frame reads, narrowed by name.

    A row is read when its group is in ``groups`` and not in ``skipped``, its
    name matches an ``include`` pattern that applies to the group, it matches
    no ``exclude`` pattern that applies. Patterns are ``(group, glob)``
    pairs, and a pattern with no group is for every group: a group's own
    includes replace those, while every exclude for it applies. A group no
    include applies to, or that excludes ``*``, is not read at all.
    """

    groups: frozenset[str] | None = None
    skipped: frozenset[str] = frozenset()
    include: tuple[Pattern, ...] = EVERY_ROW
    exclude: tuple[Pattern, ...] = ()

    def _listed(self, group: str) -> bool:
        """The group is in ``groups`` and not in ``skipped``."""
        return (self.groups is None or group in self.groups) and group not in self.skipped

    def _globs(self, group: str) -> tuple[list[str], list[str]]:
        """The include globs and the exclude globs that apply to ``group``."""
        own = [glob for scope, glob in self.include if scope == group]
        includes = own or [glob for scope, glob in self.include if scope is None]
        return includes, [glob for scope, glob in self.exclude if scope in (None, group)]

    def reads_group(self, group: str) -> bool:
        includes, excludes = self._globs(group)
        return self._listed(group) and bool(includes) and "*" not in excludes

    def reads(self, group: str, name: str) -> bool:
        """The row's group is read and its name passes the patterns."""
        includes, excludes = self._globs(group)
        return (
            self._listed(group)
            and any(fnmatchcase(name, glob) for glob in includes)
            and not any(fnmatchcase(name, glob) for glob in excludes)
        )

    def unmatched(self, names: dict[str, list[str]]) -> tuple[str, ...]:
        """
        One notice per pattern that names no row in ``names``, each group's
        row names, among the groups in ``groups`` and not in
        ``skipped``, after one when the patterns read none of those groups.
        """
        read = {group: rows for group, rows in names.items() if self._listed(group)}
        none_read = not any(self.reads_group(group) for group in read)

        def matches(scope: str | None, glob: str) -> bool:
            return any(
                fnmatchcase(name, glob)
                for group, rows in read.items()
                if scope in (None, group)
                for name in rows
            )

        notices = ["Warning: the patterns read no group"] if none_read else []
        notices += [
            f"Warning: {word} pattern {_shown(scope, glob)} matches no row"
            for word, patterns in (("include", self.include), ("exclude", self.exclude))
            for scope, glob in patterns
            if not matches(scope, glob)
        ]
        return tuple(notices)

    def pattern_line(self) -> str:
        """The include patterns, then the excludes marked ``!``, comma separated."""
        return ", ".join(
            [
                *(_shown(scope, glob) for scope, glob in self.include),
                *(_shown(scope, f"!{glob}") for scope, glob in self.exclude),
            ]
        )

    def with_pattern_line(self, line: str) -> "Selection":
        """
        This selection with the patterns of ``line``, as ``pattern_line``
        writes them. Raises ``ValueError`` on ``!GROUP:GLOB``.
        """
        patterns = [_parse_pattern(p.strip()) for p in line.split(",") if p.strip()]
        return replace(
            self,
            include=tuple((scope, glob) for scope, glob, excluded in patterns if not excluded),
            exclude=tuple((scope, glob) for scope, glob, excluded in patterns if excluded),
        )

    def excluding(self, rows: Iterable[tuple[str, str]]) -> "Selection":
        """
        This selection without the ``(group, name)`` rows, each name matched
        exactly, since one such as ``z_idle_threads[0]`` reads as a glob.
        """
        exact = tuple((group, escape(name)) for group, name in rows)
        return replace(self, exclude=(*self.exclude, *exact))

    def to_header(self) -> dict:
        """This selection as a recording header holds it."""
        return {
            "groups": None if self.groups is None else sorted(self.groups),
            "skipped": sorted(self.skipped),
            "include": [[scope, pattern] for scope, pattern in self.include],
            "exclude": [[scope, pattern] for scope, pattern in self.exclude],
        }

    @classmethod
    def from_header(cls, header: dict | None) -> "Selection":
        """
        The selection a recording header holds. A header without one, or
        without includes, read every row, which ``*`` names.
        """
        if not header:
            return cls()
        return cls(
            groups=None if header.get("groups") is None else frozenset(header["groups"]),
            skipped=frozenset(header.get("skipped", ())),
            include=tuple((scope, pattern) for scope, pattern in header.get("include", EVERY_ROW)),
            exclude=tuple((scope, pattern) for scope, pattern in header.get("exclude", ())),
        )

    def keep(self, frame: dict) -> dict:
        """``frame`` with only the groups and rows this selection reads."""
        return {
            group: [row for row in rows if self.reads(group, row.name)]
            for group, rows in frame.items()
            if self.reads_group(group)
        }
