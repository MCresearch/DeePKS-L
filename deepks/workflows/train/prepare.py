"""Train workflow - Prepare stage.

This module handles the preparation stage of training workflow:
- Load training and test data
- Set up data readers
- Configure model parameters
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional

from deepks.io.readers import GroupReader
from deepks.utils import load_dirs, load_elem_table
from deepks.core.ml.utils import fit_elem_const


def prepare_train_data(config: Dict[str, Any]) -> Tuple[GroupReader, Optional[GroupReader], Dict[str, Any]]:
    """Prepare training data (Stage 1).

    This function loads training and test data, sets up readers,
    and prepares model configuration.

    Args:
        config: Configuration dictionary

    Returns:
        tuple: (train_reader, test_reader, model_config)
    """
    # Set random seed
    seed = config.get('seed')
    if seed is None:
        seed = np.random.randint(0, 2**32)
    print(f'# using seed: {seed}')
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Extract configuration
    train_paths = config.get('systems_train', [])
    test_paths = config.get('systems_test')
    data_args = config.get('data_args', {})
    model_args = config.get('model_args', {})
    proj_basis = config.get('proj_basis')
    fit_elem = config.get('fit_elem', False)
    restart = config.get('restart')

    # Add proj_basis to model_args if provided
    if proj_basis is not None:
        model_args = {**model_args, 'proj_basis': proj_basis}

    # Load training data
    train_paths = load_dirs(train_paths)
    print(f'# training with {len(train_paths)} system(s)')
    train_reader = GroupReader(train_paths, **data_args)

    # Load test data
    if test_paths is not None:
        test_paths = load_dirs(test_paths)
        print(f'# testing with {len(test_paths)} system(s)')
        test_reader = GroupReader(test_paths, **data_args)
    else:
        print('# testing with training set')
        test_reader = None

    # Prepare model configuration
    input_dim = train_reader.ndesc
    if model_args.get("input_dim", input_dim) != input_dim:
        print(f"# `input_dim` in `model_args` does not match data "
              f"({input_dim}). Use the one in data.")
    model_args["input_dim"] = input_dim

    # Fit element constants if requested
    if fit_elem and restart is None:
        elem_table = model_args.get("elem_table", None)
        if isinstance(elem_table, str):
            elem_table = load_elem_table(elem_table)
        elem_table = fit_elem_const(train_reader, test_reader, elem_table)
        model_args["elem_table"] = elem_table

    # Prepare model config
    model_config = {
        'model_args': model_args,
        'restart': restart,
        'ckpt_file': config.get('ckpt_file', 'model.pth'),
        'graph_file': config.get('graph_file'),
        'device': config.get('device'),
        'preprocess_args': config.get('preprocess_args', {}),
        'train_args': config.get('train_args', {}),
        'fit_elem': fit_elem
    }

    return train_reader, test_reader, model_config
