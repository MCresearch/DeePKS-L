"""
Sampling helper coverage for reader refactor.

Tests:
- `test_build_system_probabilities_normalized`
- `test_build_group_sampling_cache_structure`
"""

import numpy as np

from deepks.io.readers.sampling import build_group_sampling_cache, build_system_probabilities


class _DummyReader:
    def __init__(self, natm, nframes, batch_size, neg=None):
        self.natm = natm
        self.nframes = nframes
        self.batch_size = batch_size
        self.neg = neg


def test_build_system_probabilities_normalized():
    probs = build_system_probabilities([4, 2, 2])
    assert np.isclose(probs.sum(), 1.0)
    assert np.allclose(probs, np.array([0.5, 0.25, 0.25]))


def test_build_group_sampling_cache_structure():
    readers = [
        _DummyReader(natm=2, nframes=4, batch_size=2, neg=None),
        _DummyReader(natm=3, nframes=2, batch_size=2, neg=None),
        _DummyReader(natm=3, nframes=2, batch_size=2, neg=None),
    ]
    cache = build_group_sampling_cache(readers)

    assert (2, None) in cache["group_dict"]
    assert (3, None) in cache["group_dict"]
    assert np.isclose(cache["group_probs"].sum(), 1.0)

    for shape, probs in cache["batch_prob"].items():
        assert np.isclose(probs.sum(), 1.0)
        assert probs.shape[0] == len(cache["group_dict"][shape])
