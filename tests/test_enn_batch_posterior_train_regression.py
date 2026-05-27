"""Regression: batch_posterior must agree with posterior when x is training data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams, PosteriorFlags

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUST_EMPTY_NEIGHBOR_TEST = "batch_posterior_train_regression"
_RUST_EMPTY_NEIGHBOR_FN = "batch_posterior_on_train_matches_posterior_when_no_neighbors"


def test_batch_posterior_matches_posterior_on_train_x():
    """Shared-neighbor batch path on x_train must match row-wise posterior (fit LOO flags)."""
    rng = np.random.default_rng(17)
    n, d = 12, 3
    train_x = rng.standard_normal((n, d))
    train_y = train_x.sum(axis=1, keepdims=True)
    train_yvar = 0.1 * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    paramss = [
        ENNParams(3, epistemic_variance_scale=0.5, aleatoric_variance_scale=0.1),
        ENNParams(3, epistemic_variance_scale=2.0, aleatoric_variance_scale=0.0),
    ]
    flags = PosteriorFlags(exclude_nearest=True, observation_noise=True)
    post_batch = model.batch_posterior(train_x, paramss, flags=flags)
    for i, params in enumerate(paramss):
        post = model.posterior(train_x, params=params, flags=flags)
        np.testing.assert_allclose(post_batch.mu[i], post.mu, rtol=0, atol=0)
        np.testing.assert_allclose(post_batch.se[i], post.se, rtol=0, atol=0)


def test_batch_posterior_on_train_empty_neighbor_path_matches_posterior():
    """When neighbor lookup is empty, batch_posterior must use empty-posterior (se=1), not zeros.

    Reproduces query x = x_train with k_num_neighbors = 0 (invalid at the Python API; covered in
    Rust). Before the fix, batch_posterior left se at 0 while posterior() returned se = 1.
    """
    if sys.platform == "win32":
        pytest.skip("cargo test not run on win32 in this regression guard")
    proc = subprocess.run(
        [
            "cargo",
            "test",
            "-p",
            "ennbo",
            "--test",
            _RUST_EMPTY_NEIGHBOR_TEST,
            _RUST_EMPTY_NEIGHBOR_FN,
            "--",
            "--nocapture",
        ],
        cwd=_REPO_ROOT / "rust",
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
