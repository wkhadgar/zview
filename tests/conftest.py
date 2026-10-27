# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from zview import logging_setup


@pytest.fixture(autouse=True, scope="session")
def _a_log_of_their_own(tmp_path_factory):
    """The tests log to a file of their own, not to the zview.log a user reads."""
    log_dir = tmp_path_factory.mktemp("log")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(logging_setup, "_resolve_log_dir", lambda: log_dir)
        yield
