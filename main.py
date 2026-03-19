#!/usr/bin/env python
"""DeePKS main entry point.

This is the unified entry point for all DeePKS commands.
It loads configuration, validates it, and dispatches to the appropriate workflow.
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
    from deepks.io.input.validator import validate_config

    # Get config file path
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'input.yaml'

    # Check file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found", file=sys.stderr)
        print(f"Usage: python main.py [config.yaml]", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    config = load_config(config_file)

    # Check command field
    if 'command' not in config:
        print("Error: 'command' field is required in configuration file", file=sys.stderr)
        print("Valid commands: train, test, scf, stats, iterate", file=sys.stderr)
        sys.exit(1)

    command = config['command']

    # Get defaults and merge
    scf_soft = config.get('scf_soft', 'pyscf')
    defaults = get_default_config(command, scf_soft)
    config = merge_configs(defaults, config)

    # Apply parameter inheritance for iterate command
    if command == 'iterate':
        config = apply_parameter_inheritance(config)

    # Validate configuration
    try:
        validate_config(config, command)
    except Exception as e:
        print(f"Error: Configuration validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    return config


def dispatch_to_workflow(config):
    """Dispatch to appropriate workflow based on command.

    Args:
        config: Configuration dictionary
    """
    command = config['command']

    if command == 'scf':
        # TODO: Will be implemented in step 2
        # from deepks.workflows.scf import run_scf_workflow
        # run_scf_workflow(config)

        # Temporary: use old dispatcher
        from deepks.io.input.dispatcher import dispatch_command
        dispatch_command(config)

    elif command == 'train':
        # TODO: Will be implemented in step 4
        # from deepks.workflows.train import run_train_workflow
        # run_train_workflow(config)

        # Temporary: use old dispatcher
        from deepks.io.input.dispatcher import dispatch_command
        dispatch_command(config)

    elif command == 'test':
        # Temporary: use old dispatcher
        from deepks.io.input.dispatcher import dispatch_command
        dispatch_command(config)

    elif command == 'iterate':
        # TODO: Will be implemented in step 5
        # from deepks.workflows.iterate import run_iterate_workflow
        # run_iterate_workflow(config)

        # Temporary: use old dispatcher
        from deepks.io.input.dispatcher import dispatch_command
        dispatch_command(config)

    elif command == 'stats':
        # TODO: Will be implemented later
        # from deepks.workflows.stats import run_stats_workflow
        # run_stats_workflow(config)

        # Temporary: use old dispatcher
        from deepks.io.input.dispatcher import dispatch_command
        dispatch_command(config)

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        print("Valid commands: train, test, scf, stats, iterate", file=sys.stderr)
        sys.exit(1)


def print_help():
    """Print help message."""
    print("""DeePKS: Deep Kohn-Sham DFT with machine learning

Usage:
    python main.py [config.yaml]

Arguments:
    config.yaml    Configuration file (default: input.yaml)

Configuration file must contain:
    command: <command_name>    # Required: train, test, scf, stats, iterate
    scf_soft: <backend>        # Optional: pyscf (default) or abacus

Examples:
    python main.py input.yaml
    python main.py my_config.yaml

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
