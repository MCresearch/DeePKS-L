"""Reusable ML-side trainer."""

import os
from time import time

import numpy as np
import torch
import torch.optim as optim

from deepks.ml.train.grouped_loss import GroupedLossTracker


class Trainer:
    """Generic trainer for model/objective pairs."""

    def __init__(self, *, batch_adapter=None):
        self.batch_adapter = batch_adapter

    def _preview_batch(self, reader):
        if self.batch_adapter is not None:
            return self.batch_adapter(reader.sample_all())
        if hasattr(reader, "sample_all_task_batch"):
            return reader.sample_all_task_batch()
        return reader.sample_all()

    @staticmethod
    def _batch_group_key(batch):
        metadata = getattr(batch, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        return metadata.get("group_key")

    def _iter_eval_batches(self, reader):
        if self.batch_adapter is not None:
            for batch in reader.sample_all_batch():
                yield self.batch_adapter(batch)
            return
        if hasattr(reader, "sample_all_task_batches"):
            yield from reader.sample_all_task_batches()
            return
        yield from reader.sample_all_batch()

    def _iter_train_batches(self, reader):
        if self.batch_adapter is not None:
            for batch in reader:
                yield self.batch_adapter(batch)
            return
        if hasattr(reader, "iter_task_batches"):
            yield from reader.iter_task_batches()
            return
        yield from reader

    def train(
        self,
        model,
        train_reader,
        objective,
        n_epoch=1000,
        test_reader=None,
        *,
        test_objective=None,
        start_lr=0.001,
        decay_steps=100,
        decay_rate=0.96,
        stop_lr=None,
        decay_rate_iter=None,
        weight_decay=0.0,
        fix_embedding=False,
        display_epoch=100,
        display_detail_test=0,
        display_grouped_loss=False,
        ckpt_file="model.pth",
        graph_file=None,
        device="cpu",
    ):
        """Train a model with caller-provided objective objects."""

        if objective is None:
            raise ValueError("Trainer.train requires an objective")

        model = model.to(device)
        model.eval()
        print("# working on device:", device)
        if test_reader is None:
            test_reader = train_reader
        if test_objective is None:
            test_objective = objective
        if fix_embedding and model.embedder is not None:
            model.embedder.requires_grad_(False)
        if decay_rate_iter is not None:
            current_dir = os.getcwd()
            current_iter = current_dir.split("/")[-2].split(".")[-1]
            if current_iter != "init":
                current_iter = int(current_iter)
                start_lr = start_lr * (decay_rate_iter ** current_iter)
                print(f"# resetting start_lr to {start_lr:.2e} because of decay_rate_iter")
        optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
        if stop_lr is not None:
            decay_rate = (stop_lr / start_lr) ** (1 / (n_epoch // decay_steps))
            print(f"# resetting decay_rate: {decay_rate:.4f} to satisfy stop_lr: {stop_lr:.2e}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)

        print("# epoch      trn_err   tst_err        lr  trn_time  tst_time", end="")
        preview_batch = self._preview_batch(train_reader.readers[0])
        data_keys = preview_batch.display_keys() if hasattr(preview_batch, "display_keys") else preview_batch.keys()
        align_len = 20
        objective.print_head("trn_loss", data_keys, align_len)
        if display_detail_test:
            test_objective.print_head("tst_loss", data_keys, align_len)

        tic = time()
        train_loss_tracker = GroupedLossTracker()
        test_loss_tracker = GroupedLossTracker()
        for adapted_batch in self._iter_eval_batches(train_reader):
            loss = objective.compute_losses(model, adapted_batch)
            train_loss_tracker.add_loss(self._batch_group_key(adapted_batch), loss)
        trn_loss = train_loss_tracker.avg_loss()
        for adapted_batch in self._iter_eval_batches(test_reader):
            loss = test_objective.compute_losses(model, adapted_batch)
            test_loss_tracker.add_loss(self._batch_group_key(adapted_batch), loss)
        tst_loss = test_loss_tracker.avg_loss()
        tst_time = time() - tic
        if display_grouped_loss:
            for group_key in train_loss_tracker.group_keys():
                objective.print_head(str(group_key) + "_trn", data_keys, align_len)
            for group_key in test_loss_tracker.group_keys():
                if display_detail_test:
                    test_objective.print_head(str(group_key) + "_tst", data_keys, align_len)
                else:
                    test_objective.print_head(str(group_key) + "_tst", [], align_len)
        print("")

        print(
            f"  {0:<8d}  {np.sqrt(np.abs(trn_loss[-1])):>.2e}  {np.sqrt(np.abs(tst_loss[-1])):>.2e}"
            f"  {start_lr:>.2e}  {0:>8.2f}  {tst_time:>8.2f}",
            end="",
        )
        for loss_term in trn_loss[:-1]:
            print(f"{loss_term:>{align_len}.4e}", end="")
        if display_detail_test:
            for loss_term in tst_loss[:-1]:
                print(f"{loss_term:>{align_len}.4e}", end="")
        if display_grouped_loss:
            train_loss_tracker.print_avg_group_loss(align_len)
            test_loss_tracker.print_avg_group_loss(align_len)
        print("")

        for epoch in range(1, n_epoch + 1):
            tic = time()
            train_loss_tracker = GroupedLossTracker()
            test_loss_tracker = GroupedLossTracker()
            for adapted_sample in self._iter_train_batches(train_reader):
                model.train()
                optimizer.zero_grad()
                loss = objective.compute_losses(model, adapted_sample)
                loss[-1].backward()
                optimizer.step()
                train_loss_tracker.add_loss(self._batch_group_key(adapted_sample), loss)
            scheduler.step()

            if epoch % display_epoch == 0:
                model.eval()
                trn_loss = train_loss_tracker.avg_loss()
                trn_time = time() - tic
                tic = time()
                for adapted_batch in self._iter_eval_batches(test_reader):
                    loss = test_objective.compute_losses(model, adapted_batch)
                    test_loss_tracker.add_loss(self._batch_group_key(adapted_batch), loss)
                tst_loss = test_loss_tracker.avg_loss()
                tst_time = time() - tic
                print(
                    f"  {epoch:<8d}  {np.sqrt(np.abs(trn_loss[-1])):>.2e}  {np.sqrt(np.abs(tst_loss[-1])):>.2e}"
                    f"  {scheduler.get_last_lr()[0]:>.2e}  {trn_time:>8.2f}  {tst_time:8.2f}",
                    end="",
                )
                for loss_term in trn_loss[:-1]:
                    print(f"{loss_term:>{align_len}.4e}", end="")
                if display_detail_test and epoch % (display_detail_test * display_epoch) == 0:
                    for loss_term in tst_loss[:-1]:
                        print(f"{loss_term:>{align_len}.4e}", end="")
                if display_grouped_loss:
                    train_loss_tracker.print_avg_group_loss(align_len)
                    test_loss_tracker.print_avg_group_loss(align_len)
                print("")
                if ckpt_file:
                    model.save(ckpt_file)

        if ckpt_file:
            model.save(ckpt_file)
        if graph_file:
            model.compile_save(graph_file)
