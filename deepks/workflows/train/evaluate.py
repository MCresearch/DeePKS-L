"""Train workflow - Evaluate stage.

This module handles the evaluation stage of training workflow:
- Evaluate model on test set
- Compute metrics
- Generate evaluation report
"""

import numpy as np
from typing import Dict, Any, Optional

from deepks.core.ml.models.corrnet import CorrNet
from deepks.core.ml.eval.evaluator import Evaluator
from deepks.core.ml.utils import make_loss
from deepks.io.readers import GroupReader


def evaluate_model(model: CorrNet,
                   test_reader: Optional[GroupReader],
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the model (Stage 3).

    This function evaluates the trained model on the test set
    and computes various metrics.

    Args:
        model: Trained model
        test_reader: Test data reader (if None, skip evaluation)
        config: Configuration dictionary

    Returns:
        dict: Evaluation results with metrics
    """
    if test_reader is None:
        print('# no test set provided, skipping evaluation')
        return {
            'test_loss': None,
            'test_metrics': {}
        }

    print('# evaluating model on test set')

    # Extract evaluation parameters
    train_args = config.get('train_args', {})
    energy_per_atom = train_args.get('energy_per_atom', 0)

    # Create evaluator for testing (only energy with L2 loss)
    test_eval = Evaluator(
        energy_factor=1.,
        energy_lossfn=make_loss(),  # Default L2 loss
        force_factor=0.,
        density_factor=0.,
        grad_penalty=0.,
        energy_per_atom=energy_per_atom
    )

    # Evaluate on all test batches
    model.eval()
    test_losses = []

    for batch in test_reader.sample_all_batch():
        loss = test_eval(model, batch)
        test_losses.append([loss_term.item() for loss_term in loss])

    # Compute average test loss
    test_loss = np.mean(test_losses, axis=0)
    test_rmse = np.sqrt(np.abs(test_loss[-1]))

    print(f'# test RMSE: {test_rmse:.4e}')

    # Prepare results
    results = {
        'test_loss': test_loss[-1],
        'test_rmse': test_rmse,
        'test_metrics': {
            'energy_loss': test_loss[0] if len(test_loss) > 1 else test_loss[-1],
            'total_loss': test_loss[-1]
        }
    }

    return results
