"""Online replay, leakage, and recovery tests for optional semantic adaptation."""

import numpy as np
import pytest

from enn import AdaptiveSemanticENN, SemanticOutputProjection
from enn.evals.adapt_problem import make_problem


def test_bounded_stream_and_failed_updates_preserve_predictions():
    problem = make_problem("xx", "misleading", 13)
    model = AdaptiveSemanticENN(input_transform=problem.input_transform, capacity=30)
    model.add(problem.x[:30], problem.y[:30])
    before = model.posterior(problem.x[-4:]).mu.copy()
    with pytest.raises(ValueError):
        model.add(problem.x[30:40], np.ones((10, 2)))
    np.testing.assert_array_equal(before, model.posterior(problem.x[-4:]).mu)
    assert model.num_seen == model.num_retained == 30
    model.add(problem.x[30:100], problem.y[30:100])
    assert model.num_seen == 100 and model.num_retained == 30
    assert model.num_refits == 2
    assert np.isfinite(model.posterior(problem.x[-4:]).se).all()
    assert model.posterior(problem.x[:0]).mu.shape == (0, 1)


def test_geometry_learns_rotated_response_without_linear_mean():
    problem = make_problem("xx", "misleading", 7)
    model = AdaptiveSemanticENN(input_transform=problem.input_transform)
    model.add(problem.x[:100], problem.y[:100])
    model.add(problem.x[100:300], problem.y[100:300])
    prediction = model.posterior(problem.x[300:])
    error = np.sqrt(np.mean((prediction.mu - problem.y[300:]) ** 2))
    assert error < problem.y[300:].std() * 0.4
    assert model.choice.weight > 0


def test_signed_output_learning_and_deterministic_replay():
    problem = make_problem("yy", "opposing", 6)
    models = [
        AdaptiveSemanticENN(output_projection=problem.output_projection, seed=3)
        for _ in range(2)
    ]
    for model in models:
        model.add(problem.x[:100], problem.y[:100])
    left, right = [model.posterior(problem.x[300:]) for model in models]
    np.testing.assert_array_equal(left.mu, right.mu)
    assert np.corrcoef(left.mu[:, 0], left.mu[:, 1])[0, 1] < -0.9
    assert np.isfinite(left.se).all() and (left.se > 0).all()


def test_constants_and_zero_embeddings_with_linear_mean():
    x = np.random.default_rng(2).normal(size=(30, 3))
    y = np.full((30, 2), [3.0, -1.0])
    projection = SemanticOutputProjection.from_embeddings(["a", "b"], np.zeros((2, 2)))
    model = AdaptiveSemanticENN(output_projection=projection, linear_mean=True).add(
        x, y
    )
    np.testing.assert_array_equal(model.posterior(x).mu, y)
    assert np.isfinite(model.posterior(x).se).all()


def test_invalid_data_and_unfitted_state():
    model = AdaptiveSemanticENN()
    with pytest.raises(RuntimeError):
        model.posterior([[0]])
    with pytest.raises(ValueError):
        AdaptiveSemanticENN(capacity=3)
    with pytest.raises(ValueError):
        model.add([[0]], [[0]])
    with pytest.raises(ValueError):
        model.add(np.ones((12, 1)), np.full((12, 1), np.nan))
    assert model.num_seen == 0
    model.add(np.ones((12, 1)), np.ones((12, 1)))
    with pytest.raises(ValueError):
        model.posterior([[np.nan]])


@pytest.mark.parametrize("adaptation", ["inputs", "outputs", "none"])
def test_ablation_restricts_candidate_family(adaptation):
    problem = make_problem("yy", "opposing", 6)
    model = AdaptiveSemanticENN(
        output_projection=problem.output_projection, adaptation=adaptation
    ).add(problem.x[:30], problem.y[:30])
    if adaptation in ("outputs", "none"):
        assert model.choice.geometry == "fixed"
    if adaptation == "inputs":
        assert model.choice.output == "independent"
    if adaptation == "none":
        assert model.choice.output == "semantic"


@pytest.mark.parametrize(
    "weight",
    [
        0,
        -1,
        True,
        np.bool_(False),
        np.inf,
        np.nan,
        "30",
        pytest.param(10**1000, id="oversized-integer"),
    ],
)
def test_invalid_weight_preserves_replay_and_random_state(weight):
    x = np.arange(30.0)[:, None]
    models = [AdaptiveSemanticENN().add(x, np.sin(x)) for _ in range(2)]
    with pytest.raises(ValueError, match="geometry_weight"):
        models[0].add([[31.0]], [[1.0]], geometry_weight=weight)
    for model in models:
        model.add([[31.0]], [[1.0]])
    assert models[0].choice == models[1].choice
    np.testing.assert_array_equal(models[0].posterior(x).mu, models[1].posterior(x).mu)


@pytest.mark.parametrize("adaptation", ["all", "inputs", "outputs", "none"])
@pytest.mark.parametrize("linear_mean", [False, True])
def test_fixed_weight_and_immutable_metadata(adaptation, linear_mean):
    problem = make_problem("yy", "opposing", 6, count=100)
    projection = problem.output_projection
    before = projection.loadings.copy()
    model = AdaptiveSemanticENN(
        output_projection=projection, adaptation=adaptation, linear_mean=linear_mean
    ).add(problem.x, problem.y, geometry_weight=30)
    assert model.choice.weight in (0, 30)
    assert 0 <= model.choice.alpha <= 1
    np.testing.assert_array_equal(projection.loadings, before)
    assert np.isfinite(model.posterior(problem.x).mu).all()


def test_opposing_outputs_learn_zero_semantic_mixing():
    rng = np.random.default_rng(30)
    x = rng.uniform(size=(300, 2))
    signal = np.sin(3 * x[:, :1])
    y = np.hstack([signal, -signal]) + rng.normal(0, 0.03, size=(300, 2))
    projection = SemanticOutputProjection.from_embeddings(["a", "b"], [[1, 0], [1, 0]])
    model = AdaptiveSemanticENN(output_projection=projection, adaptation="none").add(
        x, y
    )
    assert model.choice.output == "semantic"
    assert model.choice.alpha == 0
