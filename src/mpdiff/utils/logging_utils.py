"""Logging configuration helpers."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure package logging with a compact formatter.

    Parameters
    ----------
    level:
        Logging level name (e.g. ``"DEBUG"``, ``"INFO"``).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
