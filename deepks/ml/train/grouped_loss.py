"""Generic grouped-loss aggregation helpers for the ML train/eval loop."""

import numpy as np


class GroupedLossTracker:
    """Aggregate ordered batch losses, optionally grouped by a metadata key."""

    def __init__(self):
        self.grouped_losses = {}
        self.all_losses = []
        self.n_loss_term = 0

    def add_loss(self, group_key, loss):
        assert len(loss) > 0, "loss should not be empty"
        if not self.n_loss_term:
            self.n_loss_term = len(loss)
        assert len(loss) == self.n_loss_term, (
            f"loss length differs for group {group_key}, expected {self.n_loss_term}, got {len(loss)}"
        )
        values = [loss_term.item() for loss_term in loss]
        self.all_losses.append(values)
        if group_key is None:
            return
        self.grouped_losses.setdefault(group_key, []).append(values)

    def group_keys(self):
        return sorted(self.grouped_losses.keys())

    def avg_group_loss(self):
        return {group_key: np.mean(losses, axis=0) for (group_key, losses) in self.grouped_losses.items()}

    def print_avg_group_loss(self, align_len=20):
        avg_group_loss = sorted(self.avg_group_loss().items(), key=lambda x: x[0])
        for (_, group_loss) in avg_group_loss:
            for avg_loss_term in group_loss[:-1]:
                print(f"{avg_loss_term:>{align_len}.4e}", end="")

    def avg_loss(self):
        return np.mean(self.all_losses, axis=0)
