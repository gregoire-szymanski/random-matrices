"""Command line interface for mpdiff experiments."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser."""
    parser = argparse.ArgumentParser(description="mpdiff experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "simulation",
        "mp-forward",
        "mp-inverse",
        "full-pipeline",
        "end-to-end",
        "inversion-benchmark",
    ]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, required=True, help="Path to YAML/TOML config")

    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()

    if args.command == "simulation":
        from mpdiff.experiments.run_simulation import run_simulation

        run_simulation(args.config)
    elif args.command == "mp-forward":
        from mpdiff.experiments.run_mp_forward import run_mp_forward

        run_mp_forward(args.config)
    elif args.command == "mp-inverse":
        from mpdiff.experiments.run_mp_inverse import run_mp_inverse

        run_mp_inverse(args.config)
    elif args.command in {"full-pipeline", "end-to-end"}:
        from mpdiff.experiments.run_full_pipeline import run_full_pipeline

        run_full_pipeline(args.config)
    elif args.command == "inversion-benchmark":
        from mpdiff.experiments.run_inversion_benchmark import run_inversion_benchmark

        run_inversion_benchmark(args.config)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
