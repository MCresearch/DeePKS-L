"""Train workflow - Train stage.

This module handles the training stage of training workflow:
- Create or load model
- Preprocess data
- Train the model
"""

from typing import Dict, Any, Tuple, Optional

from deepks.ml.models.corrnet import CorrNet
from deepks.ml.utils import preprocess, fit_elem_const
from deepks.ml.train.train import train as train_function
from deepks.io.readers import GroupReader
from deepks.workflows.defaults import DEVICE


def train_model(train_reader: GroupReader,
                test_reader: Optional[GroupReader],
                model_config: Dict[str, Any]) -> Tuple[CorrNet, Dict[str, Any]]:
    """Train the model (Stage 2).

    This function creates or loads a model, preprocesses data,
    and trains the model.

    Args:
        train_reader: Training data reader
        test_reader: Test data reader (optional)
        model_config: Model configuration

    Returns:
        tuple: (trained_model, training_statistics)
    """
    # Extract configuration
    model_args = model_config['model_args']
    restart = model_config.get('restart')
    ckpt_file = model_config.get('ckpt_file', 'model.pth')
    graph_file = model_config.get('graph_file')
    device = model_config.get('device', DEVICE)
    preprocess_args = model_config.get('preprocess_args', {})
    train_args = model_config.get('train_args', {})
    fit_elem = model_config.get('fit_elem', False)

    # Add ckpt_file and device to train_args
    train_args = {
        **train_args,
        'ckpt_file': ckpt_file,
        'device': device
    }

    # Add graph_file if provided
    if graph_file is not None:
        train_args['graph_file'] = graph_file

    # Create or load model
    if restart is not None:
        print(f'# loading model from {restart}')
        model = CorrNet.load(restart)
        # Fit element constants if model has elem_table
        if model.elem_table is not None:
            fit_elem_const(train_reader, test_reader, model.elem_table)
    else:
        print('# creating new model')
        model = CorrNet(**model_args).double()

    # Preprocess data
    print('# preprocessing data')
    preprocess(model, train_reader, **preprocess_args)

    # Train model
    print('# starting training')
    import inspect
    valid_train_keys = set(inspect.signature(train_function).parameters.keys())
    filtered_train_args = {k: v for k, v in train_args.items() if k in valid_train_keys}
    train_function(model, train_reader, test_reader=test_reader, **filtered_train_args)

    # Collect training statistics
    train_stats = {
        'n_epochs': train_args.get('n_epoch', 1000),
        'final_lr': train_args.get('start_lr', 0.001),
        'model_saved': ckpt_file
    }

    return model, train_stats
