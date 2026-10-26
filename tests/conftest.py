# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the test modules."""

from unittest.mock import MagicMock

import pytest

from zview import logging_setup
from zview.core import Core
from zview.frontend.tui.views.base import ZViewTUIAttributes
from zview.frontend.zview_tui import ZView
from zview.selection import Selection


@pytest.fixture(autouse=True, scope="session")
def _a_log_of_their_own(tmp_path_factory):
    """The tests log to a file of their own, not to the zview.log a user reads."""
    log_dir = tmp_path_factory.mktemp("log")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(logging_setup, "_resolve_log_dir", lambda: log_dir)
        yield


@pytest.fixture
def app() -> ZView:
    """A ``ZView`` on a core over a mocked scraper, built without a terminal."""
    scraper = MagicMock()
    scraper.selection = Selection()
    scraper.groups = {"threads": MagicMock(idle_addresses=(0xDEAD,))}
    return ZView(Core(scraper), MagicMock(), theme=ZViewTUIAttributes.create_mono())
