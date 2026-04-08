"""Checks that notebook config references point to existing files."""

from __future__ import annotations

import json
import re
from pathlib import Path


def test_notebook_config_paths_exist() -> None:
    pattern = re.compile(r"Path\((['\"])(.*?)\1\)")
    notebook_root = Path(__file__).resolve().parents[1] / "notebooks"

    missing: list[tuple[str, str]] = []
    for nb_path in notebook_root.rglob("*.ipynb"):
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for match in pattern.finditer(source):
                value = match.group(2)
                if value.startswith("configs/"):
                    if not (Path(__file__).resolve().parents[1] / value).exists():
                        missing.append((str(nb_path), value))

    assert not missing, f"Missing notebook config references: {missing}"
