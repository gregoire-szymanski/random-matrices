"""Tests for CLI parser command registration."""

from __future__ import annotations

from mpdiff.cli import build_parser


def test_cli_includes_expected_commands() -> None:
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "command")
    commands = set(subparsers_action.choices.keys())

    expected = {
        "simulation",
        "mp-forward",
        "mp-inverse",
        "full-pipeline",
        "end-to-end",
        "inversion-benchmark",
    }
    assert expected.issubset(commands)
