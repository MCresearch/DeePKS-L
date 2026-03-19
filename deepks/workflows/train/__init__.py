"""Train workflow package.

This package implements the training workflow for DeePKS models.
It follows a three-stage pattern:
1. Prepare: Load and prepare training data
2. Train: Train the model
3. Evaluate: Evaluate model performance
"""

from .workflow import run_train_workflow

__all__ = ['run_train_workflow']
