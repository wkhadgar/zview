# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""ZView logging configuration. Installs a rotating file handler on the root `zview` logger."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_ROOT_LOGGER_NAME = "zview"
_MAX_BYTES = 2 * 1024 * 1024
_BACKUP_COUNT = 3
_configured = False


def _resolve_log_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "zview"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "zview"
    else:
        return Path.home() / ".cache" / "zview"


def configure(level: int = logging.INFO) -> Path:
    """
    Install a rotating file handler on the ZView root logger.
    Idempotent. Returns the absolute path of the log file.
    """
    global _configured

    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = log_dir / "zview.log"

    if _configured:
        return log_path

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    handler = RotatingFileHandler(
        log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(handler)

    _configured = True
    return log_path
