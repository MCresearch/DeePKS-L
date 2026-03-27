#!/usr/bin/env python
"""Unified DeePKS command-line interface."""

import os
import sys


def _configure_line_buffered_stdio():
    """Best-effort line buffering for CLI text streams.

    This keeps long-running CLI output visible under schedulers and redirected
    pipes without replacing ``sys.stdout`` / ``sys.stderr`` wrappers.
    """
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True)
        except (OSError, ValueError, TypeError):
            continue


def postprocess_result(result):
    """Convert successful workflow results into a zero process exit code.

    Console-script entrypoints execute this function under ``sys.exit(main())``.
    Returning workflow payloads such as dicts or strings therefore turns a
    successful run into exit code 1 with the payload echoed to stderr.
    """
    _ = result
    return 0


def main():
    """Main entry point for DeePKS CLI."""
    import argparse

    _configure_line_buffered_stdio()

    parser = argparse.ArgumentParser(
        prog="deepks",
        description="DeePKS: Deep Kohn-Sham DFT with machine learning"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="input.yaml",
        help="Configuration file (default: input.yaml)"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="DeePKS 1.0"
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found", file=sys.stderr)
        sys.exit(1)

    from deepks.io.input import build_runtime_config
    from deepks.io.input.dispatcher import dispatch_command

    try:
        runtime_config = build_runtime_config(args.config)
        result = dispatch_command(runtime_config)
        return postprocess_result(result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
