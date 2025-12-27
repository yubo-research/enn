from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from enn.turbo.morbo_trust_region import MorboTrustRegion
from enn.turbo.no_trust_region import NoTrustRegion
from enn.turbo.tr_helpers import validate_trust_region_request
from enn.turbo.turbo_trust_region import TurboTrustRegion


def test_no_trust_region_init():
    tr = NoTrustRegion(num_dim=3, num_arms=4)
    assert tr.num_dim == 3 and tr.num_arms == 4 and tr.length == 1.0


def test_no_trust_region_update_does_nothing():
    tr = NoTrustRegion(num_dim=3, num_arms=4)
    tr.update(np.array([1.0, 2.0, 3.0]))
    assert tr.length == 1.0


def test_no_trust_region_needs_restart():
    tr = NoTrustRegion(num_dim=3, num_arms=4)
    assert not tr.needs_restart()


def test_no_trust_region_validate_request():
    tr = NoTrustRegion(num_dim=3, num_arms=4)
    tr.validate_request(4)
    with pytest.raises(ValueError):
        tr.validate_request(5)


def test_no_trust_region_compute_bounds_1d():
    tr = NoTrustRegion(num_dim=3, num_arms=4)
    x_center = np.array([0.5, 0.5, 0.5])
    lb, ub = tr.compute_bounds_1d(x_center)
    assert np.allclose(lb, 0.0) and np.allclose(ub, 1.0)


def test_no_trust_region_generate_candidates():
    from enn.turbo.tr_helpers import generate_tr_candidates

    tr = NoTrustRegion(num_dim=3, num_arms=4)
    rng = np.random.default_rng(42)
    sobol = qmc.Sobol(d=3, scramble=True, seed=42)
    x_center = np.array([0.5, 0.5, 0.5])
    candidates = generate_tr_candidates(
        tr.compute_bounds_1d, x_center, None, 100, rng=rng, sobol_engine=sobol
    )
    assert candidates.shape == (100, 3)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)


def test_turbo_trust_region_validate_request():
    tr = TurboTrustRegion(num_dim=3, num_arms=4)
    tr.validate_request(4)
    with pytest.raises(ValueError):
        tr.validate_request(5)


def test_turbo_trust_region_get_incumbent_indices():
    tr = TurboTrustRegion(num_dim=3, num_arms=4)
    rng = np.random.default_rng(42)
    y = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    indices = tr.get_incumbent_indices(y, rng)
    assert 1 in indices


def test_morbo_trust_region_validate_request():
    rng = np.random.default_rng(42)
    tr = MorboTrustRegion(num_dim=3, num_arms=4, num_metrics=2, rng=rng)
    tr.validate_request(4)
    with pytest.raises(ValueError):
        tr.validate_request(5)


def test_morbo_trust_region_get_incumbent_indices():
    rng = np.random.default_rng(42)
    tr = MorboTrustRegion(num_dim=3, num_arms=4, num_metrics=2, rng=rng)
    y = np.array([[1.0, 5.0], [5.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    indices = tr.get_incumbent_indices(y, rng)
    assert len(indices) >= 1


def test_validate_trust_region_request_exact():
    validate_trust_region_request(4, 4)
    with pytest.raises(ValueError):
        validate_trust_region_request(3, 4)


def test_validate_trust_region_request_fallback():
    validate_trust_region_request(3, 4, is_fallback=True)
    with pytest.raises(ValueError):
        validate_trust_region_request(5, 4, is_fallback=True)
