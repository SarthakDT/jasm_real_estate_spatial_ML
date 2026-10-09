"""
Shared logging setup for every pipeline stage.

Previously each stage module (`train.py`, `features.py`, ...) called
`logging.basicConfig()` independently — since `basicConfig` is a no-op once
the root logger already has a handler, whichever module `main.py` happened
to import first silently decided the format for every other stage. This
module centralizes that: one console handler (matching the original
`[LEVEL] message` format) plus a rotating file handler at
logs/pipeline.log, configured exactly once regardless of import order.

Also defines the custom SUCCESS/STATUS levels used throughout the pipeline's
health-report style logging (`_success(...)`, `_status(...)` in each stage).
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src import config

SUCCESS = 25
STATUS = 26
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(STATUS, "STATUS")

LOG_DIR = os.path.join(config.BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "pipeline.log")

_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = RotatingFileHandler(LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)


def success(logger: logging.Logger, msg: str, *args) -> None:
    logger.log(SUCCESS, msg, *args)


def status(logger: logging.Logger, msg: str, *args) -> None:
    logger.log(STATUS, msg, *args)
