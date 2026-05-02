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


def test_none_fit_params_not_in_overrides():
    """When fit params are None, they should not appear in overrides."""
    config = _make_enn_config(num_fit_samples=None, num_fit_candidates=None)
    overrides = _config_to_rust_overrides(config)

    # overrides may be None or a dict without these keys
    if overrides is not None:
        assert (
            "num_fit_samples" not in overrides or overrides["num_fit_samples"] is None
        )
        assert (
            "num_fit_candidates" not in overrides
            or overrides["num_fit_candidates"] is None
        )
