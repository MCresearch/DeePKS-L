#!/usr/bin/env python
"""DeePKS main entry point.

This is the unified entry point for all DeePKS tasks.
It loads configuration, validates it, and dispatches to the appropriate workflow.

Usage:
    deepks [config.yaml]    # Uses input.yaml by default
    deepks --help           # Show help message
"""

import sys
import os


def load_and_validate_config():
    """Load and validate configuration file.

    Returns:
        dict: Validated configuration dictionary
    """
    from deepks.io.input import load_config, get_default_config
    from deepks.io.input.merger import merge_configs, apply_parameter_inheritance

    # Get config file path
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'input.yaml'

    # Check file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found", file=sys.stderr)
        print(f"Usage: deepks [config.yaml]", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_config(config_file)

    # Check type field
    if 'type' not in config:
        print("Error: 'type' field is required in configuration file", file=sys.stderr)
        print("Valid types: train, test, scf, stats, iterate", file=sys.stderr)
        sys.exit(1)

    task_type = config['type']

    # Get defaults and merge
    scf_soft = config.get('scf_soft', 'pyscf')
    defaults = get_default_config(task_type, scf_soft)
    config = merge_configs(defaults, config)

    # Apply parameter inheritance for iterate type
    if task_type == 'iterate':
        config = apply_parameter_inheritance(config)

    return config


def dispatch_to_workflow(config):
    """Dispatch to appropriate workflow based on type.

    Args:
        config: Configuration dictionary
    """
    from deepks.io.input.dispatcher import dispatch_command
    dispatch_command(config)


def print_help():
    """Print help message."""
    print("""DeePKS: Deep Kohn-Sham DFT with machine learning

Usage:
    deepks [config.yaml]

Arguments:
    config.yaml    Configuration file (default: input.yaml)

Configuration file must contain:
    type: <task_type>      # Required: train, test, scf, stats, iterate
    scf_soft: <backend>    # Optional: pyscf (default) or abacus

Examples:
    deepks                 # Uses input.yaml
    deepks my_config.yaml  # Uses specified config
    deepks --help          # Show this help

For more information, see docs/input-parameter.md
""")


def main():
    """Main entry point for DeePKS."""
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        sys.exit(0)

    try:
        # 1. Load and validate configuration
        config = load_and_validate_config()

        # 2. Dispatch to appropriate workflow
        dispatch_to_workflow(config)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
