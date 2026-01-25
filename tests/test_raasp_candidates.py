from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import qmc

from enn.turbo.config.enums import CandidateRV, RAASPDriver
from enn.turbo.tr_helpers import (
    generate_tr_candidates,
    generate_tr_candidates_fast,
    generate_tr_candidates_orig,
)


def _compute_bounds(x_center, lengthscales=None):
    return np.zeros_like(x_center), np.ones_like(x_center)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_generate_tr_candidates_orig_sobol(rng):
    num_dim = 10
    num_candidates = 8
    x_center = np.full(num_dim, 0.5)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=42)

    cand = generate_tr_candidates_orig(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol_engine,
    )

    assert cand.shape == (num_candidates, num_dim)
    assert np.all(cand >= 0.0) and np.all(cand <= 1.0)
    # Check that at least some values are perturbed
    assert not np.allclose(cand, 0.5)


def test_generate_tr_candidates_orig_uniform(rng):
    num_dim = 10
    num_candidates = 8
    x_center = np.full(num_dim, 0.5)

    cand = generate_tr_candidates_orig(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
    )

    assert cand.shape == (num_candidates, num_dim)
    assert np.all(cand >= 0.0) and np.all(cand <= 1.0)


def test_generate_tr_candidates_orig_sobol_engine_required(rng):
    num_dim = 10
    num_candidates = 8
    x_center = np.full(num_dim, 0.5)

    with pytest.raises(ValueError, match="sobol_engine is required"):
        generate_tr_candidates_orig(
            _compute_bounds,
            x_center,
            None,
            num_candidates,
            rng=rng,
            candidate_rv=CandidateRV.SOBOL,
            sobol_engine=None,
        )


def test_generate_tr_candidates_fast_sobol(rng):
    num_dim = 10
    num_candidates = 8
    x_center = np.full(num_dim, 0.5)

    cand = generate_tr_candidates_fast(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        num_pert=2,
    )

    assert cand.shape == (num_candidates, num_dim)
    assert np.all(cand >= 0.0) and np.all(cand <= 1.0)
    # Verify that perturbations happened
    perturbed_mask = ~np.isclose(cand, 0.5)
    assert np.all(perturbed_mask.sum(axis=1) >= 1)


def test_generate_tr_candidates_fast_uniform(rng):
    num_dim = 10
    num_candidates = 8
    x_center = np.full(num_dim, 0.5)

    cand = generate_tr_candidates_fast(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=2,
    )

    assert cand.shape == (num_candidates, num_dim)
    assert np.all(cand >= 0.0) and np.all(cand <= 1.0)


def test_generate_tr_candidates_dispatcher(rng):
    num_dim = 10
    num_candidates = 4
    x_center = np.full(num_dim, 0.5)
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=42)

    # Test FAST dispatch
    cand_fast = generate_tr_candidates(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=None,
        raasp_driver=RAASPDriver.FAST,
        num_pert=2,
    )
    assert cand_fast.shape == (num_candidates, num_dim)

    # Test ORIG dispatch
    cand_orig = generate_tr_candidates(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol_engine,
        raasp_driver=RAASPDriver.ORIG,
        num_pert=2,
    )
    assert cand_orig.shape == (num_candidates, num_dim)


def test_generate_tr_candidates_fast_edge_cases(rng):
    num_dim = 5
    num_candidates = 10
    x_center = np.zeros(num_dim)

    # num_pert = 1
    cand = generate_tr_candidates_fast(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=1,
    )
    assert cand.shape == (num_candidates, num_dim)
    # Because of binomial sampling and np.maximum(ks, 1),
    # we expect at least 1 perturbation per row.
    assert np.all((~np.isclose(cand, 0.0)).sum(axis=1) >= 1)

    # num_pert = num_dim
    cand = generate_tr_candidates_fast(
        _compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=num_dim,
    )
    assert np.all((~np.isclose(cand, 0.0)).sum(axis=1) == num_dim)
