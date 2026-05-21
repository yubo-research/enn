from __future__ import annotations

import numpy as np
import pytest

from enn.turbo.python_fallback.components.incumbent_tracker import (
    IncrementalIncumbentTracker,
    _NoiselessMax,
    _ScalarTopM,
    build_incumbent_tracker,
    incumbent_tracker_m_from_surrogate_config,
)
from enn.turbo.config import ENNSurrogateConfig, NoSurrogateConfig

_KISS_RUST_INCUMBENT_TRACKER_SYMBOLS = (
    "push_top_m",
    "sorted_indices",
    "tracker_m_from_enn_k",
    "tracker_m_no_surrogate",
)


def test_kiss_rust_incumbent_tracker_symbol_registry():
    assert len(_KISS_RUST_INCUMBENT_TRACKER_SYMBOLS) >= 4


def test_noiseless_scalar_tracks_all_max_ties_incrementally():
    tracker = IncrementalIncumbentTracker(m=3, noise_aware=False, num_metrics=1)
    tracker.tell(0, 1.0)
    assert np.array_equal(tracker.ask(), np.array([0], dtype=int))
    tracker.tell(1, 3.0)
    assert np.array_equal(tracker.ask(), np.array([1], dtype=int))
    tracker.tell(2, 2.0)
    assert np.array_equal(tracker.ask(), np.array([1], dtype=int))
    tracker.tell(3, 3.0)
    assert np.array_equal(tracker.ask(), np.array([1, 3], dtype=int))


def test_noisy_scalar_tracks_top_m_incrementally():
    tracker = IncrementalIncumbentTracker(m=3, noise_aware=True, num_metrics=1)
    for idx, val in enumerate([0.1, 0.9, 0.3, 0.8, 0.2]):
        tracker.tell(idx, val)
    assert np.array_equal(tracker.ask(), np.array([1, 2, 3], dtype=int))


def test_multi_metric_tracks_union_of_top_m_per_metric():
    tracker = IncrementalIncumbentTracker(m=2, noise_aware=False, num_metrics=2)
    rows = [[10, 0], [9, 1], [0, 10], [1, 9], [5, 5]]
    for idx, row in enumerate(rows):
        tracker.tell(idx, np.array(row, dtype=float))
    assert np.array_equal(tracker.ask(), np.array([0, 1, 2, 3], dtype=int))


def test_top_m_ties_prefer_lower_indices_at_boundary():
    tracker = IncrementalIncumbentTracker(m=2, noise_aware=True, num_metrics=1)
    for idx, val in enumerate([5.0, 5.0, 5.0]):
        tracker.tell(idx, val)
    assert np.array_equal(tracker.ask(), np.array([0, 1], dtype=int))


def test_ask_empty_returns_empty_int_array():
    tracker = IncrementalIncumbentTracker(m=1, noise_aware=False, num_metrics=1)
    assert np.array_equal(tracker.ask(), np.array([], dtype=int))
    assert tracker.ask().dtype == int


def test_tracker_validates_single_observation_shape():
    scalar = IncrementalIncumbentTracker(m=1, noise_aware=False, num_metrics=1)
    with pytest.raises(ValueError):
        scalar.tell(0, np.array([1.0, 2.0]))
    multi = IncrementalIncumbentTracker(m=1, noise_aware=False, num_metrics=2)
    with pytest.raises(ValueError):
        multi.tell(0, np.array([1.0]))


def test_tracker_reset_and_rebuild():
    tracker = IncrementalIncumbentTracker(m=2, noise_aware=True, num_metrics=1)
    for idx, val in enumerate([1.0, 3.0, 2.0, 4.0, 0.5]):
        tracker.tell(idx, val)
    tracker.reset()
    assert np.array_equal(tracker.ask(), np.array([], dtype=int))
    trimmed = np.array([2.0, 4.0, 0.5], dtype=float)
    tracker.rebuild(trimmed)
    assert np.array_equal(tracker.ask(), np.array([0, 1], dtype=int))


def test_ask_does_not_store_full_y_history():
    tracker = IncrementalIncumbentTracker(m=2, noise_aware=True, num_metrics=1)
    assert not hasattr(tracker, "_y_rows")
    tracker.tell(0, 1.0)
    assert not hasattr(tracker, "_y_rows")


def test_noise_aware_scalar_top_k_matches_enn_k_semantics():
    tracker = IncrementalIncumbentTracker(m=2, noise_aware=True, num_metrics=1)
    for idx, val in enumerate([0.1, 0.9, 0.3, 0.8]):
        tracker.tell(idx, val)
    assert np.array_equal(tracker.ask(), np.array([1, 3], dtype=int))


def test_non_enn_surrogate_config_uses_all_candidates_m():
    assert incumbent_tracker_m_from_surrogate_config(NoSurrogateConfig()) >= 10**8


def test_no_surrogate_tracker_returns_all_indices():
    m = incumbent_tracker_m_from_surrogate_config(NoSurrogateConfig())
    tracker = IncrementalIncumbentTracker(m=m, noise_aware=False, num_metrics=1)
    for idx, val in enumerate([0.1, 0.3, -0.2]):
        tracker.tell(idx, val)
    assert np.array_equal(tracker.ask(), np.array([0, 1, 2], dtype=int))


def test_enn_k_from_config_sets_tracker_m():
    cfg = ENNSurrogateConfig(k=5)
    assert incumbent_tracker_m_from_surrogate_config(cfg) == 5


def test_scalar_top_m_helpers_directly():
    topm = _ScalarTopM(2)
    topm.tell(0, 1.0)
    topm.tell(1, 3.0)
    topm.tell(2, 2.0)
    assert np.array_equal(topm.ask(), np.array([1, 2], dtype=int))
    topm.reset()
    assert topm.ask().size == 0


def test_noiseless_max_helpers_directly():
    state = _NoiselessMax()
    state.tell(0, 1.0)
    state.tell(1, 3.0)
    assert np.array_equal(state.ask(), np.array([1], dtype=int))
    state.reset()
    assert state.ask().size == 0


def test_build_incumbent_tracker_from_config():
    from enn.turbo.config import TurboTRConfig
    from enn.turbo.python_fallback.turbo_trust_region import TurboTrustRegion

    surrogate = ENNSurrogateConfig(k=4)
    tr_state = TurboTrustRegion(
        num_dim=2,
        config=TurboTRConfig(noise_aware=True),
    )
    tracker = build_incumbent_tracker(surrogate, tr_state)
    assert tracker._m == 4
    assert tracker._noise_aware is True
