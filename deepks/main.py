"""Compatibility facade for legacy `deepks.main` imports.

The real CLI implementation lives in `deepks.cli.main`.
This facade keeps monkeypatch-friendly dispatch behavior for existing tests/tools.
"""

import argparse

from deepks.cli import main as _cli


def train_cli(args=None):
    return _cli.train_cli(args)


def test_cli(args=None):
    return _cli.test_cli(args)


def scf_cli(args=None):
    return _cli.scf_cli(args)


def stats_cli(args=None):
    return _cli.stats_cli(args)


def iter_cli(args=None):
    return _cli.iter_cli(args)


def main_cli(args=None):
    parser = argparse.ArgumentParser(
        prog="deepks",
        description="A program to generate accurate energy functionals.",
    )
    parser.add_argument(
        "command",
        help="specify the sub-command to run, possible choices: train, test, scf, stats, iterate",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="arguments to be passed to the sub-command")

    parsed = parser.parse_args(args)

    cmd = parsed.command.upper()
    if cmd == "TRAIN":
        sub_cli = train_cli
    elif cmd == "TEST":
        sub_cli = test_cli
    elif cmd == "SCF":
        sub_cli = scf_cli
    elif cmd == "STATS":
        sub_cli = stats_cli
    elif cmd.startswith("ITER"):
        sub_cli = iter_cli
    else:
        return ValueError(f"unsupported sub-command: {parsed.command}")

    sub_cli(parsed.args)


if __name__ == "__main__":
    main_cli()
