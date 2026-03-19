#!/usr/bin/env python
"""Unified DeePKS command-line interface."""

import os
import sys


def _get_model_backend():
    """Get model backend instance."""
    from deepks.io.adapters import CorrNetModelBackend
    return CorrNetModelBackend()


def _get_physics_backend(scf_soft='pyscf'):
    """Get physics backend based on scf_soft parameter.

    Args:
        scf_soft: SCF software name ('pyscf' or 'abacus')

    Returns:
        PhysicsBackend instance
    """
    from deepks.core.physics import get_scf_backend
    return get_scf_backend(scf_soft)


def main():
    """Main entry point for DeePKS CLI."""
    import argparse

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

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found", file=sys.stderr)
        sys.exit(1)

    # Load and process configuration
    from deepks.io.input import load_config, get_default_config
    from deepks.io.input.merger import merge_configs, apply_parameter_inheritance
    from deepks.io.input.dispatcher import dispatch_command

    try:
        # Load configuration file
        config = load_config(args.config)

        # Determine type from config
        if 'type' not in config:
            print("Error: 'type' field is required in configuration file", file=sys.stderr)
            print("Valid types: train, test, scf, stats, iterate", file=sys.stderr)
            sys.exit(1)

        task_type = config['type']

        # Get defaults based on type and backend
        scf_soft = config.get('scf_soft', 'pyscf')
        defaults = get_default_config(task_type, scf_soft)

        # Merge defaults with config
        config = merge_configs(defaults, config)

        # Apply parameter inheritance for iterate type
        if task_type == 'iterate':
            config = apply_parameter_inheritance(config)

        # Dispatch to appropriate handler
        dispatch_command(config)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
