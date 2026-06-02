#!/usr/bin/env python
"""Regenerate docs/input-parameter.md from the input docs source."""

import argparse
from pathlib import Path

from deepks.io.input import render_input_parameter_doc


DOC_SYNC_TARGET = Path(__file__).resolve().parents[2] / "docs" / "input-parameter.md"


def sync_input_parameter_docs(output_path=DOC_SYNC_TARGET):
    """Render input parameter docs and write them to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_input_parameter_doc(), encoding="utf-8")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        prog="sync_input_parameter_docs",
        description="Regenerate docs/input-parameter.md from deepks/io/input/docs.py",
    )
    parser.add_argument(
        "--output",
        default=str(DOC_SYNC_TARGET),
        help="Output path for rendered input-parameter docs",
    )
    args = parser.parse_args()

    output_path = sync_input_parameter_docs(args.output)
    print(f"Wrote input parameter docs to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
