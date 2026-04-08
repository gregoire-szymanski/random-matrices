"""Timing utilities for benchmarking numerical tasks."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Generator


@dataclass(slots=True)
class TimerResult:
    """Runtime result for a timed block."""

    label: str
    elapsed_seconds: float


@contextmanager
def timed_block(label: str, logger: logging.Logger | None = None) -> Generator[TimerResult, None, None]:
    """Context manager that measures execution time.

    Parameters
    ----------
    label:
        Name of the timed operation.
    logger:
        Optional logger used to emit runtime diagnostics.

    Yields
    ------
    TimerResult
        Mutable timer result; elapsed time is populated on exit.
    """
    start = perf_counter()
    result = TimerResult(label=label, elapsed_seconds=0.0)
    try:
        yield result
    finally:
        result.elapsed_seconds = perf_counter() - start
        if logger is not None:
            logger.info("%s completed in %.3fs", label, result.elapsed_seconds)
