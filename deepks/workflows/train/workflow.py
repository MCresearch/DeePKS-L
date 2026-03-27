"""Train workflow - main orchestration.

This module implements the training workflow for DeePKS models.
It follows a three-stage pattern but delegates to existing training code.
"""

from .prepare import prepare_train_data
from .train import train_model
from .evaluate import evaluate_model
from .types import TrainResult
from dataclasses import asdict
from contextlib import redirect_stdout


def run_train_workflow(config):
    """Run training workflow.

    This is the main entry point for model training. It orchestrates
    the three stages: prepare, train, and evaluate.

    Physical Process:
    1. Prepare: Load training and test data
    2. Train: Train the DeePKS model
    3. Evaluate: Evaluate model performance

    Args:
        config: Configuration dictionary containing:
            - type: 'train'
            - systems_train: Training system paths
            - systems_test: Test system paths (optional)
            - model_args: Model architecture parameters
            - data_args: Data loading parameters
            - train_args: Training parameters
            - preprocess_args: Preprocessing parameters
            - restart: Path to restart from checkpoint
            - ckpt_file: Path to save checkpoint
            - proj_basis: Projection basis file
            - fit_elem: Whether to fit element constants
            - seed: Random seed
            - device: Device to use (cpu/cuda)

    Returns:
        dict: Training results with model path and statistics
    """
    # Stage 1: Prepare - Load and prepare data
    train_data, test_data, model_config = prepare_train_data(config)

    # Stage 2: Train - Train the model
    model, train_stats = train_model(train_data, test_data, model_config)

    # Stage 3: Evaluate - Evaluate model performance and write log.test
    metrics = evaluate_model(model, test_data, config)

    # Reproduce legacy log.test: run test() with stdout redirected
    ckpt_file = config.get('ckpt_file', 'model.pth')
    test_log = config.get('test_log', 'log.test')
    if test_data is not None:
        from deepks.ml.eval.test import test as run_test
        run_device = config.get('device', 'cpu')
        model = model.to(run_device)
        # Reconstruct the header line that GroupReader prints at construction time
        data_keys = list(dict.fromkeys(test_data.readers[0].sample_all().keys()))
        header_line = f'# load {test_data.nsystems} systems with fields {data_keys}'
        with open(test_log, 'w', 1) as f_test, redirect_stdout(f_test):
            print(header_line)
            print(ckpt_file)
            run_test(model, test_data, dump_prefix=None, device=run_device)

    result = TrainResult(
        model_path=config.get('ckpt_file', 'model.pth'),
        metrics=metrics,
        train_stats=train_stats,
    )
    return asdict(result)
