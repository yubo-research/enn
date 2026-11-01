"""Streaming scenarios for semantic parameters on the ENN workflow."""

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors, SemanticOutputProjection
from enn.enn.enn_fitter import ENNStatefulFitter
from evals.harness.adapt_problem import make_problem
from evals.harness.work_dirs import fresh_work_dir


def stream(problem, **kwargs):
    return EpistemicNearestNeighbors(
        problem.x[:0],
        problem.y[:0],
        input_transform=problem.input_transform,
        output_projection=problem.output_projection,
        work_dir=fresh_work_dir(),
        **kwargs,
    )


def add_and_fit(model, x, y, seed=0):
    model.add(x, y)
    fitter = ENNStatefulFitter(
        k=8,
        seed=int(np.random.default_rng(seed).integers(2**63 - 1)),
    )
    return fitter.ask(model, num_fit_candidates=3)


def test_failed_add_preserves_rows_and_predictions():
    problem = make_problem("xx", "misleading", 13, count=48)
    model = stream(problem, weighting=True)
    params = add_and_fit(model, problem.x[:30], problem.y[:30])
    before = model.posterior(problem.x[-4:], params=params).mu.copy()
    with pytest.raises(ValueError):
        model.add(problem.x[30:40], np.ones((10, 2)))
    assert len(model) == 30
    np.testing.assert_array_equal(
        before, model.posterior(problem.x[-4:], params=params).mu
    )
    assert model.posterior(problem.x[:0], params=params).mu.shape == (0, 1)


def test_signed_output_learning_and_deterministic_replay():
    problem = make_problem("yy", "opposing", 6, count=128)
    results = []
    for _ in range(2):
        model = stream(problem)
        params = add_and_fit(model, problem.x[:100], problem.y[:100], seed=3)
        results.append(model.posterior(problem.x[100:], params=params))
    left, right = results
    np.testing.assert_array_equal(left.mu, right.mu)
    assert np.corrcoef(left.mu[:, 0], left.mu[:, 1])[0, 1] < -0.9
    assert np.isfinite(left.se).all() and (left.se > 0).all()


def test_constant_outputs_with_zero_embeddings():
    x = np.random.default_rng(2).normal(size=(30, 3))
    y = np.full((30, 2), [3.0, -1.0])
    projection = SemanticOutputProjection.from_embeddings(["a", "b"], np.zeros((2, 2)))
    model = EpistemicNearestNeighbors(
        x[:0], y[:0], output_projection=projection, work_dir=fresh_work_dir()
    )
    params = add_and_fit(model, x, y)
    post = model.posterior(x, params=params)
    np.testing.assert_allclose(post.mu, y, atol=1e-12)
    assert np.isfinite(post.se).all()


def test_invalid_data_is_rejected_without_changing_the_model():
    model = EpistemicNearestNeighbors(
        np.empty((0, 1)), np.empty((0, 1)), work_dir=fresh_work_dir()
    )
    with pytest.raises(ValueError):
        model.add(np.ones((12, 1)), np.full((12, 1), np.nan))
    assert len(model) == 0
    params = add_and_fit(model, np.ones((12, 1)), np.ones((12, 1)))
    with pytest.raises(ValueError):
        model.posterior(np.array([[np.nan]]), params=params)
