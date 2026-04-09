"""Checks that notebook config references point to existing files."""

from __future__ import annotations

import json
import re
from pathlib import Path


def test_notebook_config_paths_exist() -> None:
    path_pattern = re.compile(r"Path\((['\"])(.*?)\1\)")
    root_join_pattern = re.compile(r"PROJECT_ROOT\s*/\s*(['\"])(configs/[^'\"]+)\1")
    notebook_root = Path(__file__).resolve().parents[1] / "notebooks"

    missing: list[tuple[str, str]] = []
    for nb_path in notebook_root.rglob("*.ipynb"):
        nb = json.loads(nb_path.read_text())
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            values = [m.group(2) for m in path_pattern.finditer(source)] + [
                m.group(2) for m in root_join_pattern.finditer(source)
            ]
            for value in values:
                if value.startswith("configs/") and not (Path(__file__).resolve().parents[1] / value).exists():
                    missing.append((str(nb_path), value))

    assert not missing, f"Missing notebook config references: {missing}"
