"""Semantic features use metadata and bounds, never observations or targets."""

import json
import sys

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors, SemanticInputTransform
from enn.enn.enn_params import ENNParams


class Encoder:
    def encode(self, texts):
        return np.array([[len(t), t.count("a"), t.count("b")] for t in texts])


def make_transform(**kwargs):
    return SemanticInputTransform(
        ["a", "b", "ab"], [[0, 2]] * 3, encoder=Encoder(), **kwargs
    )


def test_projection_geometry_and_order():
    transform = make_transform(variance_cutoff=1)
    assert transform.projection.shape == (3, 2)
    np.testing.assert_allclose(np.sum(transform.projection**2), 2)
    np.testing.assert_allclose(np.sum(transform.weighted_projection**2), 0.75)
    x = np.array([[0.1, 0.8, 1.5], [0.4, 1.9, 1.1]])
    z = transform.transform(x)
    np.testing.assert_array_equal(z, transform.transform(x))
    np.testing.assert_array_equal(
        z, transform.transform(x[:, [2, 0, 1]], names=["ab", "a", "b"])
    )
    np.testing.assert_allclose(transform.inverse(z), x)
    with pytest.raises(ValueError):
        transform.transform(x, names=["a", "a", "b"])


@pytest.mark.parametrize("legacy", [False, True])
def test_cache_load_without_sentence_runtime(tmp_path, monkeypatch, legacy):
    transform = make_transform(variance_cutoff=1)
    path = tmp_path / "projection.npz"
    transform.save(path)
    with np.load(path) as data:
        metadata = json.loads(str(data["metadata"]))
    assert metadata["variance_cutoff"] == 1
    if legacy:
        metadata.pop("variance_cutoff")
        np.savez(
            path,
            bounds=transform.bounds,
            projection=transform.projection,
            metadata=json.dumps(metadata),
        )
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    restored = SemanticInputTransform.load(path)
    np.testing.assert_array_equal(
        restored.transform([[1, 0, 2]]), transform.transform([[1, 0, 2]])
    )
    assert restored.names == transform.names
    assert restored.variance_cutoff == (0.9 if legacy else 1)
    assert not restored.projection.flags.writeable


def test_descriptions_files_fallback_and_degeneracy(tmp_path):
    path = tmp_path / "description.txt"
    path.write_text("aaa", encoding="utf-8")
    actual = make_transform(descriptions=[path.as_uri(), None, "  "])
    expected = make_transform(descriptions=["aaa", "b", "ab"])
    np.testing.assert_array_equal(actual.projection, expected.projection)
    degenerate = make_transform(descriptions=["same"] * 3)
    assert degenerate.projection.shape == (3, 0)
    assert degenerate.transform([[0, 1, 2]]).shape == (1, 3)
    limited = make_transform(variance_cutoff=0.9, relative_weight=0)
    np.testing.assert_array_equal(limited.transform([[1, 1, 1]])[:, 3:], 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"variance_cutoff": 0},
        {"variance_cutoff": -0.1},
        {"variance_cutoff": float("nan")},
        {"variance_cutoff": float("inf")},
        {"variance_cutoff": float("-inf")},
        {"variance_cutoff": 1.01},
        {"relative_weight": -1},
        {"relative_weight": float("nan")},
        {"descriptions": [None]},
    ],
)
def test_invalid_metadata_configuration(kwargs):
    with pytest.raises(ValueError):
        make_transform(**kwargs)


@pytest.mark.parametrize("cutoff,columns", [(0.5, 1), (0.9, 1), (0.98, 2), (1, 2)])
def test_variance_cutoff_selects_shortest_prefix(cutoff, columns):
    # The normalized, centered encoder vectors have 97.66% in the first factor.
    transform = make_transform(variance_cutoff=cutoff)
    assert transform.projection.shape == (3, columns)
    np.testing.assert_allclose(np.sum(transform.projection**2), columns)
    np.testing.assert_allclose(np.sum(transform.weighted_projection**2), 0.75)
    assert transform.transform([[0, 1, 2]]).shape == (1, 3 + columns)
    assert make_transform().variance_cutoff == 0.9


def test_full_cutoff_keeps_small_numerically_nonzero_components():
    class UnequalEncoder:
        def encode(self, texts):
            return np.array(
                [[1, 1e-10, 0], [1, -1e-10, 0], [1, 0, 1e-5], [1, 0, -1e-5]]
            )

    transform = SemanticInputTransform(
        ["a", "b", "c", "d"],
        [[0, 1]] * 4,
        variance_cutoff=1,
        encoder=UnequalEncoder(),
    )
    assert transform.projection.shape == (4, 3)


def test_enn_paths_match_explicit_augmentation():
    transform = make_transform()
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 2, (12, 3))
    y = rng.normal(size=(12, 1))
    model = EpistemicNearestNeighbors(x[:10], y[:10], input_transform=transform)
    manual = EpistemicNearestNeighbors(transform.transform(x[:10]), y[:10])
    model.add(x[10:], y[10:])
    manual.add(transform.transform(x[10:]), y[10:])
    params = ENNParams(3, 1.0, 0.0)
    query = rng.uniform(0, 2, (3, 3))
    np.testing.assert_allclose(
        model.posterior(query, params=params).mu,
        manual.posterior(transform.transform(query), params=params).mu,
    )
    np.testing.assert_allclose(
        model.batch_posterior(query, [params]).se,
        manual.batch_posterior(transform.transform(query), [params]).se,
    )
    np.testing.assert_array_equal(
        model.neighbors(query[0], 3), manual.neighbors(transform.transform(query[0]), 3)
    )
    np.testing.assert_allclose(model.train_rows_at([0, 11])[0], x[[0, 11]])
    a = model.conditional_posterior(x[:1], y[:1], query, params=params)
    b = manual.conditional_posterior(
        transform.transform(x[:1]), y[:1], transform.transform(query), params=params
    )
    np.testing.assert_allclose(a.mu, b.mu)
    draws = model.posterior_function_draw(query, params, function_seeds=[4])[0]
    expected = manual.posterior_function_draw(
        transform.transform(query), params, function_seeds=[4]
    )[0]
    np.testing.assert_array_equal(draws, expected)


def test_conditional_draws_and_fitter_use_same_features():
    from enn import ENNStatefulFitter

    transform = make_transform()
    x = np.random.default_rng(3).uniform(0, 2, (15, 3))
    y = np.sin(x[:, :1])
    model = EpistemicNearestNeighbors(x, y, input_transform=transform)
    manual = EpistemicNearestNeighbors(transform.transform(x), y)
    params = ENNParams(3, 1.0, 0.0)
    kwargs = {"params": params, "function_seeds": [19]}
    actual = model.conditional_posterior_function_draw(x[:1], y[:1], x[2:4], **kwargs)
    expected = manual.conditional_posterior_function_draw(
        transform.transform(x[:1]), y[:1], transform.transform(x[2:4]), **kwargs
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    fitter = ENNStatefulFitter(3, seed=1, input_transform=transform)
    fitter.tell(x, y)
    fitted = fitter.ask(model, num_fit_candidates=3, num_fit_samples=5)
    assert np.isfinite(fitter.posterior(model, x[2:4], fitted).mu).all()
    assert np.isfinite(
        fitter.posterior_function_draw(model, x[2:4], fitted, function_seeds=[1])[0]
    ).all()
    with pytest.raises(ValueError, match="same input transform"):
        fitter.ask(manual)


@pytest.mark.parametrize(
    "names,bounds",
    [
        ([], []),
        (["a", "a"], [[0, 1]] * 2),
        ([""], [[0, 1]]),
        (["a"], [[1, 1]]),
        (["a"], [[0, float("inf")]]),
    ],
)
def test_invalid_problem_definition(names, bounds):
    with pytest.raises(ValueError):
        SemanticInputTransform(names, bounds, encoder=Encoder())


def test_single_variable_and_zero_embeddings():
    single = SemanticInputTransform(["a"], [[0, 1]], encoder=Encoder())
    assert single.projection.shape == (1, 0)

    class ZeroEncoder:
        def encode(self, texts):
            return np.zeros((len(texts), 2))

    zero = SemanticInputTransform(["a", "b"], [[0, 1]] * 2, encoder=ZeroEncoder())
    np.testing.assert_array_equal(zero.transform([[0, 1]]), [[-0.5, 0.5]])
    with pytest.raises(ValueError):
        zero.transform([[float("nan"), 0]])


def test_optimizer_keeps_original_bounds_and_observations():
    from enn import create_optimizer
    from enn.turbo.config.acq_type import AcqType
    from enn.turbo.config.enn_fit_config import ENNFitConfig
    from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig
    from enn.turbo.config.factory import turbo_enn_config

    transform = make_transform()
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            input_transform=transform,
            fit=ENNFitConfig(num_fit_samples=5, num_fit_candidates=3),
        ),
        num_init=4,
        acq_type=AcqType.THOMPSON,
    )
    optimizer = create_optimizer(
        bounds=transform.bounds, config=config, rng=np.random.default_rng(9)
    )
    from dataclasses import replace

    baseline = create_optimizer(
        bounds=transform.bounds,
        config=replace(
            config, surrogate=replace(config.surrogate, input_transform=None)
        ),
        rng=np.random.default_rng(9),
    )
    initial = optimizer.ask(4)
    np.testing.assert_array_equal(initial, baseline.ask(4))
    assert initial.shape == (4, 3)
    optimizer.tell(initial, np.sin(initial[:, 0]))
    query = optimizer.ask(2)
    assert query.shape == (2, 3)
    assert np.all((query >= 0) & (query <= 2))
    np.testing.assert_allclose(optimizer._x_obs, initial)
    with pytest.raises(ValueError, match="bounds must match"):
        create_optimizer(
            bounds=np.array([[0, 3]] * 3), config=config, rng=np.random.default_rng(9)
        )


def test_fit_helpers_accept_semantic_model():
    from enn.enn.enn_fit import enn_fit, subsample_loglik

    transform = make_transform()
    x = np.random.default_rng(4).uniform(0, 2, (15, 3))
    y = np.cos(x[:, :1])
    model = EpistemicNearestNeighbors(x, y, input_transform=transform)
    params = enn_fit(
        model,
        k=3,
        num_fit_candidates=3,
        num_fit_samples=5,
        rng=np.random.default_rng(5),
    )
    scores = subsample_loglik(
        model, x, y, paramss=[params], P=5, rng=np.random.default_rng(6)
    )
    assert np.isfinite(scores).all()


def test_encoder_called_once_and_invalid_embeddings_rejected():
    class CountingEncoder:
        def __init__(self):
            self.calls = 0

        def encode(self, texts):
            self.calls += 1
            return np.eye(len(texts))

    encoder = CountingEncoder()
    transform = SemanticInputTransform(["a", "b"], [[0, 1]] * 2, encoder=encoder)
    for _ in range(3):
        transform.transform([[0.2, 0.3]])
    assert encoder.calls == 1

    class InvalidEncoder:
        def encode(self, texts):
            return np.array(1)

    with pytest.raises(ValueError):
        SemanticInputTransform(["a"], [[0, 1]], encoder=InvalidEncoder())


def test_transform_configuration_is_fixed():
    from dataclasses import FrozenInstanceError

    transform = make_transform()
    with pytest.raises(FrozenInstanceError):
        transform.relative_weight = 2.0
    with pytest.raises(ValueError):
        transform.projection[0, 0] = 100


def test_surrogate_metadata_preserves_positional_y_bounds():
    import inspect

    from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig

    parameters = inspect.signature(ENNSurrogateConfig).parameters
    assert parameters["input_transform"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["y_bounds"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
