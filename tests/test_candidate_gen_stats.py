import numpy as np
import pytest
from scipy.stats import qmc
from enn.turbo.turbo_utils import generate_tr_candidates, generate_tr_candidates_fast
from enn.turbo.config import CandidateRV, RAASPDriver


def test_candidate_generation_statistical_properties():
    """
    Compares the statistical properties of generate_tr_candidates and generate_tr_candidates_fast.
    We check:
    1. Mean and variance of perturbed dimensions.
    2. Number of perturbed dimensions per candidate.
    3. Range of generated values.
    """
    num_dim = 100
    num_candidates = 1000
    num_pert = 20
    rng = np.random.default_rng(42)
    x_center = np.full(num_dim, 0.5)

    def compute_bounds(center, lengthscales=None):
        return np.zeros_like(center), np.ones_like(center)

    # --- 1. Test Uniform Perturbations ---
    orig_uniform = generate_tr_candidates(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        sobol_engine=None,
        raasp_driver=RAASPDriver.ORIG,
        num_pert=num_pert,
    )

    fast_uniform = generate_tr_candidates(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        sobol_engine=None,
        raasp_driver=RAASPDriver.FAST,
        num_pert=num_pert,
    )

    for name, cand in [
        ("Original Uniform", orig_uniform),
        ("Fast Uniform", fast_uniform),
    ]:
        # Check shape
        assert cand.shape == (num_candidates, num_dim)

        # Check that center is preserved in non-perturbed dimensions
        # Count how many elements are NOT equal to the center value
        perturbed_mask = ~np.isclose(cand, 0.5)
        per_row_pert = perturbed_mask.sum(axis=1)

        # Both should now follow the same distribution (Binomial)
        # Mean should be approx num_pert
        assert np.mean(per_row_pert) == pytest.approx(num_pert, abs=2.0)
        # Variance should be approx num_pert * (1 - num_pert/num_dim)
        expected_var = num_pert * (1 - num_pert / num_dim)
        assert np.var(per_row_pert) == pytest.approx(expected_var, abs=5.0)
        # Ensure at least one dimension is perturbed in every row
        assert np.all(per_row_pert >= 1)

        # Check that values are within [0, 1]
        assert np.all(cand >= 0.0)
        assert np.all(cand <= 1.0)

        # Check mean of perturbed values (should be ~0.5)
        perturbed_values = cand[perturbed_mask]
        assert np.mean(perturbed_values) == pytest.approx(0.5, abs=0.05)
        # Check variance of perturbed values (uniform [0,1] variance is 1/12 approx 0.0833)
        assert np.var(perturbed_values) == pytest.approx(0.0833, abs=0.01)

    # --- 2. Test Sobol Perturbations ---
    sobol_engine = qmc.Sobol(d=num_dim, scramble=True, seed=42)
    orig_sobol = generate_tr_candidates(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=sobol_engine,
        raasp_driver=RAASPDriver.ORIG,
        num_pert=num_pert,
    )

    fast_sobol = generate_tr_candidates(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.SOBOL,
        sobol_engine=None,
        raasp_driver=RAASPDriver.FAST,
        num_pert=num_pert,
    )

    for name, cand in [("Original Sobol", orig_sobol), ("Fast Sobol", fast_sobol)]:
        perturbed_mask = ~np.isclose(cand, 0.5)
        perturbed_values = cand[perturbed_mask]

        # Sobol should also be roughly uniform in [0, 1]
        assert np.mean(perturbed_values) == pytest.approx(0.5, abs=0.05)
        assert np.var(perturbed_values) == pytest.approx(0.0833, abs=0.01)


def test_fast_candidates_various_inputs():
    """Test generate_tr_candidates_fast with different num_pert and dimensions."""
    num_dim = 50
    num_candidates = 100
    rng = np.random.default_rng(0)
    x_center = np.zeros(num_dim)

    def compute_bounds(center, lengthscales=None):
        return np.full_like(center, -1.0), np.full_like(center, 1.0)

    # num_pert = 1
    cand = generate_tr_candidates_fast(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=1,
    )
    perturbed_mask = ~np.isclose(cand, 0.0)
    # Mean should be 1, but individual rows vary (Binomial)
    assert np.mean(perturbed_mask.sum(axis=1)) == pytest.approx(1.0, abs=0.5)
    assert np.all(perturbed_mask.sum(axis=1) >= 1)

    # Test num_pert = num_dim
    cand = generate_tr_candidates_fast(
        compute_bounds,
        x_center,
        None,
        num_candidates,
        rng=rng,
        candidate_rv=CandidateRV.UNIFORM,
        num_pert=num_dim,
    )
    perturbed_mask = ~np.isclose(cand, 0.0)
    assert np.all(perturbed_mask.sum(axis=1) == num_dim)
