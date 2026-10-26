# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""ZScraper: backend + ELF + kernel groups, read one frame at a time."""

import contextlib
from typing import Literal

from zview.backend.base import AbstractScraper
from zview.backend.elf_inspector import ElfInspector
from zview.backend.image import verify_target_image
from zview.kernel import compat
from zview.kernel.group import Group
from zview.kernel.groups import GROUPS
from zview.kernel.layout import KernelLayout
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.selection import Selection


class ZScraper:
    # What the session reads, and what the view on screen needs: a frame reads
    # what both allow. The TUI sets ``view`` on a live target only, dump and
    # record leave it reading everything.
    selection: Selection = Selection()
    view: Selection = Selection()

    def __init__(self, meta_scraper: AbstractScraper, elf_path):
        self._m_scraper: AbstractScraper = meta_scraper

        self._load_elf(elf_path)

    def reload_elf(self) -> None:
        """Re-parse the ELF at the same path and rebuild everything derived from it."""
        self._load_elf(self.elf_path)

    @property
    def is_live(self) -> bool:
        """Whether the session reads a target rather than a recording."""
        return self._m_scraper.is_live

    def connect(self) -> None:
        """Connect to the target, unless connected."""
        if not self._m_scraper.is_connected:
            self._m_scraper.connect()

    def reattach(self) -> None:
        """Disconnect, re-parse the ELF, reconnect, and check the target runs it."""
        self._m_scraper.disconnect()
        self.reload_elf()
        self._m_scraper.connect()
        verify_target_image(self._m_scraper, self.elf_path)

    def _load_elf(self, elf_path) -> None:
        """Parse ``elf_path`` and resolve the layout and the groups it provides."""
        self.elf_path = elf_path
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._endianess: Literal["little", "big"] = (
            "little" if self._elf_inspector._little_endian else "big"
        )
        self._m_scraper.endianess = "<" if self._endianess == "little" else ">"

        # One of each group per ELF, in the order a frame reads them, and per
        # group key whether it is read and the rows its discover found.
        self.groups: dict[str, Group] = {cls.key: cls() for cls in GROUPS}
        self.available: dict[str, bool] = {}
        self.rows: dict[str, dict] = {key: {} for key in self.groups}
        self.parts: set[str] = set()

        # Built Zephyr version (when the ELF is in its build tree) and the wait
        # queue layout.
        self.zephyr_version: str | None = compat.detect_zephyr_version(elf_path)
        self.waitq_flavor: compat.WaitQFlavor = compat.waitq_flavor(self._elf_inspector)

        self._layout = self._resolve_layout()
        # Each group's live parts start on, and a replay keeps those its recording names.
        self.parts = {
            part for group in self.groups.values() for part in group.live_parts(self._session())
        }
        self._restrict_features_to_recording()

        if not self._m_scraper.is_live:
            self.capture_all()

        self.reset_runtime_state()

    def _resolve_layout(self) -> KernelLayout:
        """Resolve each group's tables; a group whose required tables miss is not read."""
        elf = self._elf_inspector

        fields: dict = {}
        for group in self.groups.values():
            required = [compat.resolve_fields(elf, table) for table in group.required]
            self.available[group.key] = None not in required
            for found in (t for t in required if t is not None):
                fields.update(found)
            for table in group.optional:
                fields.update(compat.resolve_fields(elf, table) or {})

        return KernelLayout(**fields)

    def discover(self) -> None:
        """
        Find each available group's rows on the target; a group with none is
        not read. Runs after connect, and again on a refresh.
        """
        session = self._session()
        for group in self.groups.values():
            if self.available[group.key]:
                self.rows[group.key] = group.discover(session)
                self.available[group.key] = bool(self.rows[group.key])

    def _session(self) -> Session:
        """What the groups read with, as of now."""
        return Session(
            scraper=self._m_scraper,
            elf=self._elf_inspector,
            layout=self._layout,
            waitq_flavor=self.waitq_flavor,
            rows=self.rows,
            parts=frozenset(self.parts),
        )

    def capture_all(self) -> None:
        """Read every detail a view may ask for, as a recording holds them all."""
        for group in self.groups.values():
            group.capture_all()

    def _restrict_features_to_recording(self) -> None:
        """
        Limit a replayed session to the features and the selection its recording
        captured.

        Replay matches a strict read sequence. A recording with no feature list
        predates the primitives and is treated as threads and heaps only.
        """
        if self._m_scraper.is_live:
            return

        self.selection = getattr(self._m_scraper, "selection", None) or Selection()
        features = getattr(self._m_scraper, "features", None)
        if features is None:
            return

        for group in self.groups.values():
            self.available[group.key] = self.available[group.key] and group.key in features
        self.parts &= set(features)

    def active_features(self) -> tuple[str, ...]:
        """The features this session polls, as recorded in a recording header."""
        features: list[str] = []
        # A group read, then each of its parts that is on.
        for group in self.groups.values():
            if self.available[group.key]:
                features += [group.key, *(part for part in group.parts if part in self.parts)]
        return tuple(features)

    def absent_features(self) -> tuple[str, ...]:
        """One notice per capability the build's Kconfig leaves out."""
        session = self._session()
        return tuple(notice for group in self.groups.values() for notice in group.notices(session))

    def decl_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the object at ``address`` is declared, if known."""
        return self._elf_inspector.decl_site(address)

    def function_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the function at ``address`` is defined, if known."""
        return self._elf_inspector.function_site(address)

    def check_selection(self, selection: Selection) -> None:
        """Raises ``ValueError`` when ``selection`` names a group no frame holds."""
        mentioned = {
            *(selection.groups or ()),
            *(scope for scope, _ in (*selection.include, *selection.exclude) if scope is not None),
        }
        if unknown := mentioned - set(self.groups):
            named, known = ", ".join(sorted(unknown)), ", ".join(self.groups)
            raise ValueError(f"Unknown group {named}; choose from {known}.")

    def unmatched(self, selection: Selection) -> tuple[str, ...]:
        """One notice per pattern of ``selection`` that names no discovered row."""
        return selection.unmatched(
            {
                key: [label for label, _ in label_instances(self.rows[key])]
                for key in self.groups
                if self.available[key]
            }
        )

    def _reads_group(self, group: str) -> bool:
        """Both the session's selection and the view's read ``group``."""
        return self.selection.reads_group(group) and self.view.reads_group(group)

    def _reads(self, group: str, name: str) -> bool:
        """Both the session's selection and the view's read the row ``name``."""
        return self.selection.reads(group, name) and self.view.reads(group, name)

    def _poll_groups(self) -> dict:
        """
        Read every group the selection and the view read, in order. A failed
        read raises, and the frame is dropped.

        Wait queues are walked only on the ``simple`` layout; otherwise
        ``waiters`` is ``None``.
        """
        frame: dict = {}
        session = self._session()
        for group in self.groups.values():
            if not self.available[group.key]:
                continue
            if not self._reads_group(group.key):
                # Read again, the group seeds its baselines afresh, not from before the gap.
                group.reset()
                continue

            # Matched against the labels the rows will carry.
            selected = {
                label: [address]
                for label, address in label_instances(self.rows[group.key])
                if self._reads(group.key, label)
            }
            frame[group.key] = list(group.walk(session, selected).values())

        return frame

    def read_frame(self) -> dict:
        """The groups of one frame, read in one batch of the backend."""
        self._m_scraper.begin_batch()
        try:
            return self._poll_groups()
        finally:
            with contextlib.suppress(Exception):
                self._m_scraper.end_batch()

    def __enter__(self):
        self._m_scraper.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._m_scraper.__exit__(exc_type, exc_val, exc_tb)

    def read_variable(self, var_name: str) -> list:
        self.connect()

        var_addresses = self._elf_inspector.get_symbol_info(var_name, "address")
        var_sizes = self._elf_inspector.get_symbol_info(var_name, "size")

        return [
            self._m_scraper.read8(var_address, var_size)
            for var_address, var_size in zip(var_addresses, var_sizes, strict=True)
        ]

    def reset_runtime_state(self) -> None:
        """Clear the watermark cache and each group's baselines; the next poll seeds them."""
        self._m_scraper.watermark_cache.clear()
        for group in self.groups.values():
            group.reset()
