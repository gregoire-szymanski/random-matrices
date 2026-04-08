#!/usr/bin/env python3
"""Print a deterministic project tree for documentation maintenance.

Example
-------
python tools/print_project_tree.py --max-depth 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "outputs",
    "htmlcov",
}


def should_skip(path: Path, excludes: set[str]) -> bool:
    """Return True when path should be skipped by tree printer."""
    return any(part in excludes for part in path.parts)


def iter_entries(root: Path) -> list[Path]:
    """Return sorted children (dirs first, then files)."""
    children = [p for p in root.iterdir() if not p.name.startswith(".")]
    children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
    return children


def print_tree(
    root: Path,
    max_depth: int,
    include_files: bool,
    excludes: set[str],
) -> None:
    """Print tree to stdout."""

    def rec(node: Path, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return

        entries = []
        for child in iter_entries(node):
            if should_skip(child.relative_to(root), excludes):
                continue
            if not include_files and child.is_file():
                continue
            entries.append(child)

        for idx, child in enumerate(entries):
            is_last = idx == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            print(f"{prefix}{connector}{child.name}")
            if child.is_dir():
                extension = "    " if is_last else "|   "
                rec(child, prefix + extension, depth + 1)

    print(root.name)
    rec(root, prefix="", depth=0)


def parse_args() -> argparse.Namespace:
    """CLI arguments for tree printer."""
    parser = argparse.ArgumentParser(description="Print project tree")
    parser.add_argument("--root", type=Path, default=Path("."), help="Root directory to inspect")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum traversal depth")
    parser.add_argument(
        "--dirs-only",
        action="store_true",
        help="Show directories only",
    )
    parser.add_argument(
        "--exclude",
        default=",".join(sorted(DEFAULT_EXCLUDES)),
        help="Comma-separated folder names to exclude",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = parse_args()
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Invalid root directory: {root}")

    excludes = {token.strip() for token in args.exclude.split(",") if token.strip()}
    print_tree(
        root=root,
        max_depth=max(args.max_depth, 1),
        include_files=not args.dirs_only,
        excludes=excludes,
    )


if __name__ == "__main__":
    main()
