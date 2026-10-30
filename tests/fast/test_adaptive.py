"""Online replay, leakage, and recovery tests for optional semantic adaptation."""

import numpy as np
import pytest

from enn import AdaptiveSemanticENN, SemanticOutputProjection
from enn.adaptive_model import (
    AdaptChoice,
    AdaptiveBasis,
    AdaptivePredictor,
    _selection_means,
    select_predictor,
)
from enn.enn.enn_params import ENNParams
from enn.evals.adapt_problem import make_problem
from enn.semantic_multi_output import OutputPosterior


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


def test_calibration_labels_cannot_change_fitted_mean_or_choice():
    problem = make_problem("yy", "opposing", 4, count=100)
    changed = problem.y.copy()
    changed[80:] += 100
    args = (None, problem.output_projection, False)
    model, _, calibration = select_predictor(problem.x, problem.y, *args)
    other, _, changed_calibration = select_predictor(problem.x, changed, *args)
    assert model.choice == other.choice
    np.testing.assert_array_equal(
        model.posterior(problem.x).mu, other.posterior(problem.x).mu
    )
    assert np.max(changed_calibration) > np.max(calibration) * 10


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


def _separate_predictions(model, x, neighbors):
    previous = model.params
    predictions = []
    for k in neighbors:
        model.params = ENNParams(min(k, len(model.model)), 1, 0)
        predictions.append(model.posterior(x))
    model.params = previous
    return OutputPosterior(
        np.stack([p.mu for p in predictions]), np.stack([p.se for p in predictions])
    )


@pytest.mark.parametrize(
    "geometry,linear_mean",
    [(g, trend) for g in ("fixed", "linear", "quadratic") for trend in (False, True)],
)
def test_batched_candidate_predictions_match_separate_calls(geometry, linear_mean):
    problem = make_problem("yy", "opposing", 6, count=30)
    x, y = problem.x.copy(), problem.y.copy()
    x[1] = x[0]  # Equal distances and self queries exercise neighbor tie handling.
    y[:, -1] = 3
    basis = AdaptiveBasis(x[:18], y[:18], None, problem.output_projection)
    choice = AdaptChoice(geometry, 3, "learned", 3)
    model = AdaptivePredictor(basis, x[:18], y[:18], choice, linear_mean)
    batch = model.selection_predictions(x, (3, 10, 30))
    separate = _separate_predictions(model, x, (3, 10, 30))
    np.testing.assert_allclose(batch.mu, separate.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch.se, separate.se, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(batch.mu[..., -1], 3)
    np.testing.assert_array_equal(batch.se[..., -1], 0)


@pytest.mark.parametrize("adaptation", ["all", "inputs", "outputs", "none"])
def test_shared_index_preserves_selection_and_calibration(monkeypatch, adaptation):
    problem = make_problem("yy", "opposing", 6, count=30)
    args = (problem.x, problem.y, None, problem.output_projection, True, adaptation)
    model, floor, calibration = select_predictor(*args)
    monkeypatch.setattr(
        "enn.adaptive_model._selection_means", _reference_selection_means
    )
    other, other_floor, other_calibration = select_predictor(*args)
    assert model.choice == other.choice
    np.testing.assert_array_equal(floor, other_floor)
    np.testing.assert_allclose(calibration, other_calibration, rtol=1e-12)
    np.testing.assert_array_equal(
        model.posterior(problem.x).mu, other.posterior(problem.x).mu
    )


def _reference_selection_means(basis, rows, choice, linear_mean, neighbors):
    x, y, query = rows
    model = AdaptivePredictor(basis, x, y, choice, linear_mean)
    return _separate_predictions(model, query, neighbors).mu


@pytest.mark.parametrize("linear_mean", [False, True])
def test_index_free_selection_matches_full_predictor_and_fallback(
    monkeypatch, linear_mean
):
    from enn import enn_rust

    problem = make_problem("yy", "opposing", 6, count=100)
    basis = AdaptiveBasis(
        problem.x[:60], problem.y[:60], None, problem.output_projection
    )
    rows = (problem.x[:60], problem.y[:60], problem.x[60:])
    neighbors = (3, 10, 30)
    for geometry in ("fixed", "linear", "quadratic"):
        choice = AdaptChoice(geometry, 3, "independent", 3)
        expected = _reference_selection_means(
            basis, rows, choice, linear_mean, neighbors
        )
        actual = _selection_means(basis, rows, choice, linear_mean, neighbors)
        np.testing.assert_array_equal(actual, expected)
    monkeypatch.setattr(
        enn_rust.model, "small_flat_selection_means", lambda *args: None
    )
    fallback = _selection_means(basis, rows, choice, linear_mean, neighbors)
    np.testing.assert_array_equal(fallback, expected)


def test_semantic_column_major_features_use_index_free_selection():
    from enn import enn_rust

    problem = make_problem("xx", "misleading", 6, count=100)
    basis = AdaptiveBasis(problem.x[:60], problem.y[:60], problem.input_transform, None)
    rows = (problem.x[:60], problem.y[:60], problem.x[60:])
    choice = AdaptChoice("fixed", 0, "independent", 3)
    train, query = [basis.features(x, choice) for x in (rows[0], rows[2])]
    target = (rows[1] - basis.y_mean) / basis.y_scale
    train, query = np.asfortranarray(train), np.asfortranarray(query)
    assert not train.flags.c_contiguous
    assert (
        enn_rust.model.small_flat_selection_means(train, target, query, [3, 10, 30])
        is not None
    )
    np.testing.assert_array_equal(
        _selection_means(basis, rows, choice, False, (3, 10, 30)),
        _reference_selection_means(basis, rows, choice, False, (3, 10, 30)),
    )


@pytest.mark.parametrize("metrics", [1, 6])
def test_batched_selection_scores_match_individual_reductions(metrics):
    from enn.adaptive_model import _selection_scores

    rng = np.random.default_rng(30)
    means = rng.normal(size=(3, 60, metrics))
    means[1] = means[0]
    target = rng.normal(size=(60, metrics))
    outputs = {
        "identity": np.eye(metrics),
        "signed": rng.normal(size=(metrics, metrics)),
    }
    expected = np.array(
        [
            [np.mean((mean @ operator - target) ** 2) for operator in outputs.values()]
            for mean in means
        ]
    )
    actual = _selection_scores(means, target, outputs)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[0], actual[1])


@pytest.mark.parametrize("dimensions,metrics", [(3, 1), (3, 6), (16, 1), (16, 6)])
def test_quadratic_direction_preserves_gradient_moment(dimensions, metrics):
    from enn.adaptive_model import _quadratic_direction

    rng = np.random.default_rng(30)
    z = rng.normal(size=(100, dimensions))
    target = rng.normal(size=(100, metrics))
    design = np.column_stack((z, z**2 - 1))
    coefficients = np.linalg.solve(
        design.T @ design + len(z) * 0.03 * np.eye(2 * dimensions),
        design.T @ target,
    )
    linear, quadratic = np.split(coefficients, 2)
    gradient = linear[None] + 2 * z[:, :, None] * quadratic[None]
    moment = np.einsum("ndm,nem->de", gradient, gradient) / len(z)
    expected = moment * dimensions / np.trace(moment)
    direction = _quadratic_direction(z, target)
    np.testing.assert_allclose(
        direction @ direction.T, expected, rtol=1e-12, atol=1e-12
    )


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
def test_invalid_fixed_weight_preserves_replay_and_random_state(weight):
    model = AdaptiveSemanticENN().add(np.arange(30.0)[:, None], np.ones((30, 1)))
    before = model._rng.bit_generator.state
    with pytest.raises(ValueError, match="geometry_weight"):
        model.add([[31.0]], [[1.0]], geometry_weight=weight)
    assert model.num_seen == 30 and model.num_refits == 1
    assert model._rng.bit_generator.state == before


@pytest.mark.parametrize("linear_mean", [False, True])
@pytest.mark.parametrize("adaptation", ["all", "inputs", "outputs", "none"])
def test_fixed_weight_selection_matches_full_predictors(
    monkeypatch, adaptation, linear_mean
):
    problem = make_problem("yy", "opposing", 6, count=100)
    args = (
        problem.x,
        problem.y,
        None,
        problem.output_projection,
        linear_mean,
        adaptation,
    )
    model, floor, calibration = select_predictor(*args, geometry_weight=30)
    monkeypatch.setattr(
        "enn.adaptive_model._selection_means", _reference_selection_means
    )
    other, other_floor, other_calibration = select_predictor(*args, geometry_weight=30)
    assert model.choice == other.choice
    assert model.choice.weight in (0, 30)
    np.testing.assert_array_equal(floor, other_floor)
    np.testing.assert_allclose(calibration, other_calibration, rtol=1e-12)
    np.testing.assert_array_equal(
        model.posterior(problem.x).mu, other.posterior(problem.x).mu
    )


def test_fixed_weight_evaluates_all_geometries_and_preserves_factor_shapes(monkeypatch):
    from enn import adaptive_model

    problem = make_problem("yy", "opposing", 6, count=100)
    original = adaptive_model._selection_means
    calls = []

    def record(basis, rows, choice, linear_mean, neighbors):
        calls.append((choice.geometry, choice.weight))
        return original(basis, rows, choice, linear_mean, neighbors)

    monkeypatch.setattr(adaptive_model, "_selection_means", record)
    projection = problem.output_projection
    loadings = projection.loadings.copy()
    model = AdaptiveSemanticENN(output_projection=projection).add(
        problem.x, problem.y, geometry_weight=30
    )
    assert calls == [("fixed", 0), ("linear", 30), ("quadratic", 30)]
    np.testing.assert_array_equal(projection.loadings, loadings)
    basis = AdaptiveBasis(problem.x, problem.y, None, projection)
    assert model.output_projection is projection
    for geometry in ("linear", "quadratic"):
        shapes = [
            basis.features(problem.x, AdaptChoice(geometry, w, "independent", 3)).shape
            for w in (3, 30)
        ]
        assert shapes[0] == shapes[1]


@pytest.mark.parametrize("dimensions,offset", [(3, 0), (16, 0), (3, 1e12), (16, 1e12)])
def test_reused_centering_preserves_numpy_standard_deviations(dimensions, offset):
    rng = np.random.default_rng(30)
    x = rng.normal(size=(100, dimensions)) + offset
    y = rng.normal(size=(100, 6)) + offset
    y[:, -1] = 3
    basis = AdaptiveBasis(x, y, None, None)
    np.testing.assert_array_equal(basis.x_mean, x.mean(axis=0))
    np.testing.assert_array_equal(basis.y_mean, y.mean(axis=0))
    np.testing.assert_array_equal(basis.x_scale, np.maximum(x.std(axis=0), 1e-12))
    np.testing.assert_array_equal(
        basis.y_scale,
        np.where(np.ptp(y, axis=0) == 0, 1, np.maximum(y.std(axis=0), 1e-12)),
    )


@pytest.mark.parametrize("geometry", ["fixed", "linear", "quadratic"])
@pytest.mark.parametrize("output", ["independent", "semantic", "learned", "lowrank"])
@pytest.mark.parametrize("linear_mean", [False, True])
def test_selected_refit_matches_complete_basis(geometry, output, linear_mean):
    problem = make_problem("yy", "opposing", 6, count=100)
    args = (problem.x, problem.y, None, problem.output_projection)
    complete = AdaptiveBasis(*args)
    selected = AdaptiveBasis(
        *args, geometry=geometry, output=output, linear_mean=linear_mean
    )
    choice = AdaptChoice(geometry, 30, output, 3)
    np.testing.assert_array_equal(
        selected.features(problem.x, choice), complete.features(problem.x, choice)
    )
    np.testing.assert_array_equal(
        selected.operators[output], complete.operators[output]
    )
    if linear_mean or geometry == "linear":
        np.testing.assert_array_equal(
            selected.trend(problem.x), complete.trend(problem.x)
        )
    else:
        with pytest.raises(RuntimeError, match="without a linear trend"):
            selected.trend(problem.x)


@pytest.mark.parametrize("geometry", ["fixed", "linear", "quadratic"])
def test_square_features_preserve_distance_and_metadata_rank(geometry):
    problem = make_problem("xx", "misleading", 6, count=100)
    transform = problem.input_transform
    basis = AdaptiveBasis(problem.x[:60], problem.y[:60], transform, None)
    choice = AdaptChoice(geometry, 30, "independent", 3)
    actual = basis.features(problem.x, choice)
    if geometry == "fixed":
        expected = transform.transform(problem.x)
    else:
        direction = basis.directions[geometry]
        assert direction.shape[1] == transform.projection.shape[1]
        z = (problem.x - basis.x_mean) / basis.x_scale
        expected = np.column_stack((z, np.sqrt(choice.weight) * z @ direction))
    assert actual.shape == problem.x.shape
    np.testing.assert_allclose(
        np.sum((actual[1:] - actual[0]) ** 2, axis=1),
        np.sum((expected[1:] - expected[0]) ** 2, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_metadata_rank_stays_zero_during_learning():
    from enn.adaptive_model import _fixed_rank

    direction = _fixed_rank(np.eye(3), 0, 3)
    assert direction.shape == (3, 0)
    np.testing.assert_array_equal(direction @ direction.T, np.zeros((3, 3)))


def test_raw_candidate_has_zero_semantic_mixing():
    problem = make_problem("xx", "misleading", 3, count=100)
    basis = AdaptiveBasis(problem.x, problem.y, problem.input_transform, None)
    features = basis.features(problem.x, AdaptChoice("raw", 0, "independent", 3))
    np.testing.assert_array_equal(features, problem.x)


@pytest.mark.parametrize("geometry", ["linear", "quadratic"])
def test_constant_input_uses_unit_scale_when_it_changes_at_query(geometry):
    x = np.column_stack((np.linspace(0, 1, 30), np.zeros(30)))
    basis = AdaptiveBasis(x, np.sin(x[:, :1]), None, None)
    choice = AdaptChoice(geometry, 3, "independent", 10)
    query = np.array([[0.5, 0.0], [0.5, 1.0]])
    features = basis.features(query, choice)
    np.testing.assert_allclose(np.sum((features[1] - features[0]) ** 2), 1.0)
    model = AdaptivePredictor(basis, x, np.sin(x[:, :1]), choice, False)
    posterior = model.posterior(query)
    assert np.isfinite(posterior.mu).all()
    assert np.max(posterior.se) < 10
