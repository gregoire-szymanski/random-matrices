"""Top-level end-to-end experiment runner.

This module is an explicit alias over :mod:`run_full_pipeline`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .run_full_pipeline import run_full_pipeline


def run_end_to_end(config_path: str | Path) -> dict[str, Any]:
    """Run simulation -> realized spectrum -> MP inverse -> recovery report."""
    return run_full_pipeline(config_path)
