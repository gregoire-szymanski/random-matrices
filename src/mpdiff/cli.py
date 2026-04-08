"""Command line interface for mpdiff experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from mpdiff.experiments.run_full_pipeline import run_full_pipeline
from mpdiff.experiments.run_mp_forward import run_mp_forward
from mpdiff.experiments.run_mp_inverse import run_mp_inverse
from mpdiff.experiments.run_simulation import run_simulation


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="mpdiff experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ["simulation", "mp-forward", "mp-inverse", "full-pipeline"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, required=True, help="Path to YAML/TOML config")

    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()

    if args.command == "simulation":
        run_simulation(args.config)
    elif args.command == "mp-forward":
        run_mp_forward(args.config)
    elif args.command == "mp-inverse":
        run_mp_inverse(args.config)
    elif args.command == "full-pipeline":
        run_full_pipeline(args.config)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
