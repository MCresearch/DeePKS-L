"""
Legacy model shim export coverage.

Tests:
- `test_model_train_test_shim_exports`
- `test_model_eval_utils_shim_exports`
"""

import importlib

import deepks.model.evaluator as evaluator_shim
import deepks.model.model as model_shim
import deepks.model.test as test_shim
import deepks.model.train as train_shim
import deepks.model.utils as utils_shim

evaluator_impl = importlib.import_module("deepks.core.ml.eval.evaluator")
eval_test_impl = importlib.import_module("deepks.core.ml.eval.test")
model_impl = importlib.import_module("deepks.core.ml.models.corrnet")
utils_impl = importlib.import_module("deepks.core.ml.utils")
train_impl = importlib.import_module("deepks.core.ml.train.train")


def test_model_train_test_shim_exports():
    assert train_shim.main is train_impl.main
    assert train_shim.train is train_impl.train
    assert test_shim.main is eval_test_impl.main
    assert test_shim.test is eval_test_impl.test


def test_model_eval_utils_shim_exports():
    assert evaluator_shim.Evaluator is evaluator_impl.Evaluator
    assert model_shim.CorrNet is model_impl.CorrNet
    assert utils_shim.fit_elem_const is utils_impl.fit_elem_const
    assert utils_shim.preprocess is utils_impl.preprocess
