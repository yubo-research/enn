from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from enn.turbo.config.morbo_tr_config import MorboTRConfig
from enn.turbo.config.no_tr_config import NoTRConfig
from enn.turbo.config.rescalarize import Rescalarize
from enn.turbo.config.turbo_tr_config import TRLengthConfig, TurboTRConfig
from enn.turbo.morbo_trust_region import MorboTrustRegion
from enn.turbo.no_trust_region import NoTrustRegion
from enn.turbo.tr_helpers import (
    compute_full_box_bounds_1d,
)
from enn.turbo.turbo_trust_region import TurboTrustRegion


def test_no_trust_region_init():
    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    assert tr.num_dim == 3 and tr.length == 1.0


def test_no_trust_region_update_does_nothing():
    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    tr.update(np.array([1.0, 2.0, 3.0]))
    assert tr.length == 1.0


def test_no_trust_region_needs_restart():
    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    assert not tr.needs_restart()


def test_no_trust_region_validate_request():
    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    tr.validate_request(4)
    tr.validate_request(5)  # NoTrustRegion has no constraints


def test_no_trust_region_compute_bounds_1d():
    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    x_center = np.array([0.5, 0.5, 0.5])
    lb, ub = tr.compute_bounds_1d(x_center)
    assert np.allclose(lb, 0.0) and np.allclose(ub, 1.0)


def test_no_trust_region_generate_candidates():
    from enn.turbo.tr_helpers import generate_tr_candidates

    config = NoTRConfig()
    tr = NoTrustRegion(config=config, num_dim=3)
    rng = np.random.default_rng(42)
    sobol = qmc.Sobol(d=3, scramble=True, seed=42)
    x_center = np.array([0.5, 0.5, 0.5])
    candidates = generate_tr_candidates(
        tr.compute_bounds_1d, x_center, None, 100, rng=rng, sobol_engine=sobol
    )
    assert candidates.shape == (100, 3)
    assert np.all(candidates >= 0.0) and np.all(candidates <= 1.0)


def test_turbo_trust_region_validate_request():
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    tr = TurboTrustRegion(config=config, num_dim=3)
    tr.validate_request(4)
    # First call sets num_arms=4; changing it raises
    with pytest.raises(ValueError):
        tr.validate_request(5)


def test_turbo_trust_region_get_incumbent_indices():
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    tr = TurboTrustRegion(config=config, num_dim=3)
    rng = np.random.default_rng(42)
    y = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    indices = tr.get_incumbent_indices(y, rng)
    assert 1 in indices


def test_morbo_trust_region_validate_request():
    rng = np.random.default_rng(42)
    config = MorboTRConfig(num_metrics=2, rescalarize=Rescalarize.ON_PROPOSE)
    tr = MorboTrustRegion(config=config, num_dim=3, rng=rng)
    tr.validate_request(4)
    # First call sets num_arms=4; changing it raises
    with pytest.raises(ValueError):
        tr.validate_request(5)


def test_morbo_trust_region_get_incumbent_indices():
    rng = np.random.default_rng(42)
    config = MorboTRConfig(num_metrics=2, rescalarize=Rescalarize.ON_PROPOSE)
    tr = MorboTrustRegion(config=config, num_dim=3, rng=rng)
    y = np.array([[1.0, 5.0], [5.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    indices = tr.get_incumbent_indices(y, rng)
    assert len(indices) >= 1


def test_compute_full_box_bounds_1d():
    lb, ub = compute_full_box_bounds_1d(np.array([0.25, 0.5, 0.75]))
    assert np.allclose(lb, 0.0) and np.allclose(ub, 1.0)


def test_turbo_trust_region_compute_bounds_1d_with_lengthscales():
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.8, length_min=0.5**7, length_max=1.6)
    )
    tr = TurboTrustRegion(config=config, num_dim=3)
    x_center = np.array([0.5, 0.5, 0.5])
    lengthscales = np.array([0.5, 1.0, 2.0])
    lb, ub = tr.compute_bounds_1d(x_center, lengthscales=lengthscales)
    # Expected half-widths: lengthscales * length / 2 = [0.5, 1.0, 2.0] * 0.8 / 2
    #                     = [0.2, 0.4, 0.8]
    # lb = clip(0.5 - [0.2, 0.4, 0.8], 0, 1) = [0.3, 0.1, 0.0]
    # ub = clip(0.5 + [0.2, 0.4, 0.8], 0, 1) = [0.7, 0.9, 1.0]
    assert np.allclose(lb, [0.3, 0.1, 0.0])
    assert np.allclose(ub, [0.7, 0.9, 1.0])


def test_turbo_trust_region_expansion_and_contraction():
    config = TurboTRConfig(
        length=TRLengthConfig(length_init=0.4, length_min=0.1, length_max=1.6)
    )
    tr = TurboTrustRegion(config=config, num_dim=4)
    tr.validate_request(num_arms=4)
    assert tr.length == 0.4
    # 3 consecutive successes should double the length (success_tolerance=3)
    # Feed strictly increasing values to trigger success
    values = []
    for v in [1.0, 2.0, 3.0, 4.0]:
        values.append(v)
        tr.update(np.array(values, dtype=float))
    # After 3 successes: length = min(2.0 * 0.4, 1.6) = 0.8
    assert np.isclose(tr.length, 0.8), f"Expected 0.8 after expansion, got {tr.length}"
    # Feed failures (values that don't improve) to trigger contraction
    # failure_tolerance = ceil(max(4/4, 4/4)) = 1
    for _ in range(tr.failure_tolerance):
        values.append(values[-1])  # same value = no improvement
        tr.update(np.array(values, dtype=float))
    # After failure_tolerance failures: length = 0.5 * 0.8 = 0.4
    assert np.isclose(
        tr.length, 0.4
    ), f"Expected 0.4 after contraction, got {tr.length}"
