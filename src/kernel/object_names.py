# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Labelling for discovered kernel object instances."""

from collections.abc import Iterator


def label_instances(addresses: dict[str, list[int]]) -> Iterator[tuple[str, int]]:
    """
    Yield ``(label, address)`` for every discovered instance.

    A symbol mapping to one address keeps its name. A symbol mapping to
    several, such as an array or same-named statics in different translation
    units, gets an ``@0x...`` suffix per instance.
    """
    for name, instance_addresses in addresses.items():
        unique = list(dict.fromkeys(instance_addresses))
        for address in unique:
            yield (name if len(unique) == 1 else f"{name}@0x{address:X}"), address
