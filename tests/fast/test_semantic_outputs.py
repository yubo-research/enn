"""Metadata geometry, blend limits, and held-out calibration invariants."""

import numpy as np
import pytest

from enn import (
    EpistemicNearestNeighbors,
    SemanticMultiOutputENN,
    SemanticOutputProjection,
)
from enn.enn.enn_params import ENNParams


class OutputEncoder:
    def encode(self, texts):
        return np.array(
            [[len(text), text.count("a"), 1] for text in texts], dtype=float
        )


def projection():
    return SemanticOutputProjection.from_embeddings(
        ["a", "b", "c"], [[1, 0], [1, 0], [0, 1]]
    )


def data():
    x = np.random.default_rng(1).uniform(size=(18, 2))
    y = np.column_stack((np.sin(x[:, 0] * 3), -np.sin(x[:, 0] * 3), x[:, 1]))
    return x, y


def test_uncentered_projection_and_degeneracy(tmp_path):
    p = projection()
    np.testing.assert_allclose(p.loadings.T @ p.loadings, np.eye(2))
    np.testing.assert_allclose(np.linalg.norm(p.embeddings, axis=1), 1)
    same = SemanticOutputProjection.from_embeddings(["a", "b"], [[1, 2]] * 2)
    np.testing.assert_allclose(same.loadings @ same.loadings.T, np.full((2, 2), 0.5))
    zero = SemanticOutputProjection.from_embeddings(["a", "b"], np.zeros((2, 2)))
    assert zero.loadings.shape == (2, 0)
    path = tmp_path / "outputs.npz"
    p.save(path)
    cached = SemanticOutputProjection.load(path)
    np.testing.assert_array_equal(cached.loadings, p.loadings)
    assert not cached.loadings.flags.writeable


def test_metadata_fallback_and_disk_cache(tmp_path, monkeypatch):
    from enn import semantic_outputs

    calls = []

    def encode(texts, model):
        calls.append(texts)
        return OutputEncoder().encode(texts)

    monkeypatch.setattr(semantic_outputs, "semantic_encode", encode)
    kwargs = {"variance_cutoff": 0.9, "cache_dir": tmp_path}
    first = SemanticOutputProjection.from_metadata(["a", "b"], [None, ""], **kwargs)
    second = SemanticOutputProjection.from_metadata(["a", "b"], **kwargs)
    assert calls == [["a", "b"]]
    np.testing.assert_array_equal(first.loadings, second.loadings)
    SemanticOutputProjection.from_metadata(["a", "b"], ["new", None], **kwargs)
    assert len(calls) == 2


@pytest.mark.parametrize("variance_cutoff", [0, 3, -1, np.nan, np.inf])
def test_invalid_cutoffs(variance_cutoff):
    with pytest.raises(ValueError):
        SemanticOutputProjection.from_embeddings(
            ["a", "b", "c"], np.eye(3), variance_cutoff=variance_cutoff
        )


@pytest.mark.parametrize(
    "embeddings", [np.ones((2, 2)), [[1], [np.nan], [2]], [1, 2, 3]]
)
def test_invalid_embeddings(embeddings):
    with pytest.raises(ValueError):
        SemanticOutputProjection.from_embeddings(["a", "b", "c"], embeddings)


def test_alpha_zero_matches_independent_and_constant_outputs():
    x, y = data()
    y[:, 2] = 7
    params = ENNParams(3, 1, 0)
    model = SemanticMultiOutputENN(projection(), alpha=0, params=params, folds=3).fit(
        x, y
    )
    actual = model.posterior(x[:4])
    mean, scale = y.mean(axis=0), y.std(axis=0)
    scale[2] = 1
    independent = EpistemicNearestNeighbors(x, (y - mean) / scale)
    expected = independent.posterior(x[:4], params=params).mu * scale + mean
    np.testing.assert_allclose(actual.mu, expected, atol=1e-12)
    assert np.isfinite(actual.se).all() and (actual.se > 0).all()
    shared = SemanticMultiOutputENN(projection(), alpha=1, folds=3).fit(x, y)
    np.testing.assert_array_equal(shared.posterior(x[:4]).mu[:, 2], 7)


def test_shared_limit_and_opposing_outputs():
    x, y = data()
    params = ENNParams(3, 1, 0)
    model = SemanticMultiOutputENN(projection(), alpha=1, params=params, folds=3).fit(
        x, y
    )
    mean, scale = y.mean(axis=0), y.std(axis=0)
    h = ((y - mean) / scale) @ projection().loadings
    latent = EpistemicNearestNeighbors(x, h).posterior(x[:4], params=params)
    expected = latent.mu @ projection().loadings.T * scale + mean
    independent = EpistemicNearestNeighbors(x, (y - mean) / scale)
    projected = independent.posterior(x[:4], params=params).mu @ projection().loadings
    np.testing.assert_allclose(latent.mu, projected, atol=1e-12)
    np.testing.assert_allclose(model.posterior(x[:4]).mu, expected, atol=1e-12)
    # Opposite standardized outputs cancel in the common semantic direction.
    np.testing.assert_allclose(
        model.posterior(x[:4]).mu[:, :2], np.tile(mean[:2], (4, 1))
    )


def test_fold_refit_and_gaussian_calibration():
    x, y = data()
    model = SemanticMultiOutputENN(projection(), folds=3, seed=4).fit(x, y)
    held = model.fold_ids_ == 0
    changed = y.copy()
    changed[held] += 100
    other = SemanticMultiOutputENN(projection(), folds=3, seed=4).fit(x, changed)
    np.testing.assert_array_equal(model.oof_mean_[held], other.oof_mean_[held])
    np.testing.assert_allclose(
        np.mean((y - model.oof_mean_) ** 2 / model.oof_variance_, axis=0), 1
    )
    assert np.isfinite(model.posterior(x[:4]).se).all()


def test_zero_embeddings_fall_back_and_refit():
    x, y = data()
    zero = SemanticOutputProjection.from_embeddings(["a", "b", "c"], np.zeros((3, 2)))
    first = SemanticMultiOutputENN(zero, alpha=1, folds=3).fit(x, y)
    second = SemanticMultiOutputENN(zero, alpha=0, folds=3).fit(x, y)
    np.testing.assert_array_equal(first.posterior(x).mu, second.posterior(x).mu)
    first.fit(x, y + 1)
    np.testing.assert_allclose(first.posterior(x).mu, second.posterior(x).mu + 1)
    assert first.posterior(x[:0]).mu.shape == (0, 3)


def test_validation_and_unfitted_state():
    model = SemanticMultiOutputENN(projection(), folds=3)
    with pytest.raises(RuntimeError):
        model.posterior([[0, 0]])
    x, y = data()
    with pytest.raises(ValueError):
        model.fit(x, y[:, :2])
    with pytest.raises(ValueError):
        model.fit(x[:2], y[:2])
    with pytest.raises(ValueError):
        SemanticMultiOutputENN(projection(), alpha=np.nan)
    with pytest.raises(ValueError):
        SemanticMultiOutputENN(projection(), folds=1)
    model.fit(x, y)
    with pytest.raises(ValueError):
        model.posterior([[np.nan, 1]])


def test_batched_scalar_predictions_match_each_output():
    x, y = data()
    params = ENNParams(3, 1, 0)
    batched = EpistemicNearestNeighbors(x, y).posterior(x[:4], params=params)
    for output in range(y.shape[1]):
        scalar = EpistemicNearestNeighbors(x, y[:, output : output + 1]).posterior(
            x[:4], params=params
        )
        np.testing.assert_allclose(batched.mu[:, output], scalar.mu.ravel())
        np.testing.assert_allclose(batched.se[:, output], scalar.se.ravel())


def test_two_outputs_have_a_valid_default_rank():
    names = ["temperature", "pressure"]
    direct = SemanticOutputProjection.from_embeddings(names, np.eye(2))
    metadata = SemanticOutputProjection.from_metadata(names, encoder=OutputEncoder())
    assert direct.loadings.shape == metadata.loadings.shape == (2, 1)
    assert SemanticOutputProjection.from_embeddings(
        names, np.eye(2), variance_cutoff=1
    ).loadings.shape == (2, 1)


def test_cutoff_changes_width_and_cache_identity(tmp_path, monkeypatch):
    from enn import semantic_outputs

    calls = []

    def encode(texts, model):
        calls.append(texts)
        return np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(semantic_outputs, "semantic_encode", encode)
    names = ["a", "b", "c"]
    narrow = SemanticOutputProjection.from_metadata(
        names, variance_cutoff=0.3, cache_dir=tmp_path
    )
    wide = SemanticOutputProjection.from_metadata(
        names, variance_cutoff=0.9, cache_dir=tmp_path
    )
    cached = SemanticOutputProjection.from_metadata(
        names, variance_cutoff=0.9, cache_dir=tmp_path
    )
    assert narrow.loadings.shape == (3, 1)
    assert wide.loadings.shape == cached.loadings.shape == (3, 2)
    assert len(calls) == 2
    assert len(list(tmp_path.glob("*.npz"))) == 2
