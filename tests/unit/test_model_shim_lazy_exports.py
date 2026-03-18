"""Canonical ML module export coverage."""

import importlib


def test_train_pipeline_exports():
    train_pipeline = importlib.import_module("deepks.pipelines.train.train")
    train_core = importlib.import_module("deepks.core.ml.train.train")
    eval_pipeline = importlib.import_module("deepks.pipelines.train.test")
    eval_core = importlib.import_module("deepks.core.ml.eval.test")

    assert train_pipeline.main is train_core.main
    assert train_pipeline.train is train_core.train
    assert eval_pipeline.main is eval_core.main
    assert eval_pipeline.test is eval_core.test


def test_core_ml_exports():
    evaluator_mod = importlib.import_module("deepks.core.ml.eval.evaluator")
    model_mod = importlib.import_module("deepks.core.ml.models.corrnet")
    utils_mod = importlib.import_module("deepks.core.ml.utils")

    assert evaluator_mod.Evaluator.__module__ == "deepks.core.ml.eval.evaluator"
    assert model_mod.CorrNet.__module__ == "deepks.core.ml.models.corrnet"
    assert utils_mod.fit_elem_const.__module__ == "deepks.core.ml.utils"
    assert utils_mod.preprocess.__module__ == "deepks.core.ml.utils"
