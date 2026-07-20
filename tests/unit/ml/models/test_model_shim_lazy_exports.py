"""Canonical ML module export coverage."""

import importlib
import pytest


def test_train_pipeline_exports():
    # Train pipelines have been removed, skip this test
    pytest.skip("Train pipelines removed in refactoring")


def test_core_ml_exports():
    objective_mod = importlib.import_module("deepks.interface.objectives.descriptor_properties")
    model_mod = importlib.import_module("deepks.ml.models.corrnet")
    utils_mod = importlib.import_module("deepks.ml.utils")

    assert objective_mod.DescriptorPropertyObjectiveAdapter.__module__ == "deepks.interface.objectives.descriptor_properties"
    assert model_mod.CorrNet.__module__ == "deepks.ml.models.corrnet"
    assert utils_mod.make_loss.__module__ == "deepks.ml.utils"
