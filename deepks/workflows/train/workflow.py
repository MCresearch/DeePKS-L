"""Train workflow - main orchestration.

This module implements the training workflow for DeePKS models.
It follows a three-stage pattern but delegates to existing training code.
"""

from dataclasses import asdict

from deepks.interface.registry import get_recipe
from deepks.workflows.train.runtime import prepare_train_runtime, run_training_stage
from .types import TrainResult


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
    train_data, test_data, model_config = prepare_train_runtime(config)

    # Stage 2: Train - Train the model
    model, train_stats = run_training_stage(train_data, test_data, model_config)

    # Stage 3: Evaluate - Evaluate model performance and write log.test
    recipe = get_recipe(config=config)
    metrics = recipe.evaluate_model(model, test_data, config)

    # Write the workflow log.test file by running evaluation with stdout redirected
    runtime = config.get("runtime") if isinstance(config.get("runtime"), dict) else {}
    runtime_io = runtime.get("io") if isinstance(runtime.get("io"), dict) else {}
    ckpt_file = runtime_io.get("ckpt_file", "model.pth")
    test_log = runtime.get("test_log", "log.test")
    if test_data is not None:
        run_device = runtime.get("device", "cpu")
        recipe.write_test_log(
            model,
            test_data,
            ckpt_file=ckpt_file,
            test_log=test_log,
            device=run_device,
        )

    result = TrainResult(
        model_path=ckpt_file,
        metrics=metrics,
        train_stats=train_stats,
    )
    return asdict(result)
