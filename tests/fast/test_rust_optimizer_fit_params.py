"""Regression test: num_fit_samples and num_fit_candidates passed to Rust backend.

Bug: _config_to_rust_overrides() in rust_optimizer.py extracts index_driver but
does not extract num_fit_samples or num_fit_candidates from ENNSurrogateConfig.
This causes all experiments with different nfs= values to produce identical results.

See: enn_bug_report.md
"""

from __future__ import annotations

from enn.turbo.config import (
    ENNFitConfig,
    ENNSurrogateConfig,
    turbo_enn_config,
)
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.rust_optimizer import _config_to_rust_overrides


def _make_enn_config(num_fit_samples: int | None, num_fit_candidates: int | None):
    """Create a TuRBO-ENN config with specified fit parameters."""
    fit = ENNFitConfig(
        num_fit_samples=num_fit_samples,
        num_fit_candidates=num_fit_candidates,
    )
    enn = ENNSurrogateConfig(k=4, fit=fit)
    return turbo_enn_config(enn=enn, num_init=6)


def test_num_fit_samples_passed_to_rust_overrides():
    """num_fit_samples should be included in Rust config overrides."""
    config = _make_enn_config(num_fit_samples=100, num_fit_candidates=None)
    overrides = _config_to_rust_overrides(config)

    assert overrides is not None
    assert "num_fit_samples" in overrides, (
        "num_fit_samples not passed to Rust backend; "
        "different nfs= values will produce identical results"
    )
    assert overrides["num_fit_samples"] == 100


def test_num_fit_candidates_passed_to_rust_overrides():
    """num_fit_candidates should be included in Rust config overrides."""
    config = _make_enn_config(num_fit_samples=None, num_fit_candidates=500)
    overrides = _config_to_rust_overrides(config)

    assert overrides is not None
    assert "num_fit_candidates" in overrides, (
        "num_fit_candidates not passed to Rust backend; "
        "different values will produce identical results"
    )
    assert overrides["num_fit_candidates"] == 500


def test_both_fit_params_passed_to_rust_overrides():
    """Both num_fit_samples and num_fit_candidates should be in overrides."""
    config = _make_enn_config(num_fit_samples=50, num_fit_candidates=200)
    overrides = _config_to_rust_overrides(config)

    assert overrides is not None
    assert "num_fit_samples" in overrides
    assert "num_fit_candidates" in overrides
    assert overrides["num_fit_samples"] == 50
    assert overrides["num_fit_candidates"] == 200


def test_rust_optimizer_passes_bpann_disk_index_driver():
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(index_driver=ENNIndexDriver.BPANN_DISK)
    )
    overrides = _config_to_rust_overrides(config)

    assert overrides is not None
    assert overrides["index_driver"] == "BPANN_DISK"


def test_none_fit_tell_returns_posterior_mean_and_frozen_query_is_seed_stable():
    """tell returns the posterior mean. Unset fit samples freeze scales, so a new query does not depend on the seed."""
    import numpy as np

    from enn import create_optimizer
    from enn.turbo.config import turbo_zero_config

    bounds = np.array([[0.0, 1.0]] * 4, dtype=float)
    x = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.4, 0.3],
            [0.8, 0.7, 0.2, 0.1],
            [0.4, 0.4, 0.4, 0.4],
            [0.9, 0.1, 0.8, 0.2],
        ],
        dtype=float,
    )
    y = np.array([0.2, 1.5, -0.4, 0.7, 0.1], dtype=float)
    query = np.array([[0.55, 0.45, 0.35, 0.25]], dtype=float)

    def query_mu(seed: int, num_fit_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
        if num_fit_samples is None:
            fit = ENNFitConfig()
        else:
            fit = ENNFitConfig(num_fit_samples=num_fit_samples, num_fit_candidates=30)
        cfg = turbo_enn_config(enn=ENNSurrogateConfig(k=3, fit=fit), num_init=2)
        opt = create_optimizer(bounds=bounds, config=cfg, rng=np.random.default_rng(seed))
        told = np.asarray(opt.tell(x, y), dtype=float)
        at_query = np.asarray(opt._inner.posterior_mu(query), dtype=float)
        return told, at_query

    told_a, frozen_a = query_mu(1, None)
    told_b, frozen_b = query_mu(99, None)
    assert told_a.shape == y.shape
    assert np.allclose(told_a, told_b)
    assert np.allclose(frozen_a, frozen_b)
    _, fitted = query_mu(1, 10)
    assert not np.allclose(frozen_a, fitted)

    zero = create_optimizer(
        bounds=bounds,
        config=turbo_zero_config(num_init=2),
        rng=np.random.default_rng(0),
    )
    echoed = np.asarray(zero.tell(x, y), dtype=float)
    assert np.allclose(echoed, y)


def test_none_fit_params_freeze_the_scales():
    """None means do not search. Rust would otherwise keep the factory counts of 10 and 30."""
    config = _make_enn_config(num_fit_samples=None, num_fit_candidates=None)
    overrides = _config_to_rust_overrides(config)

    assert overrides is not None
    assert overrides["freeze_params"] is True
    assert "num_fit_samples" not in overrides
    assert "num_fit_candidates" not in overrides
