"""Metadata geometry and blend limits."""

import numpy as np
import pytest

from enn import EpistemicNearestNeighbors, SemanticOutputProjection
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import ENNParams
from evals.harness.work_dirs import fresh_work_dir


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
    assert calls == [["a::", "b::"]]
    np.testing.assert_array_equal(first.loadings, second.loadings)
    SemanticOutputProjection.from_metadata(["a", "b"], ["new", None], **kwargs)
    assert len(calls) == 2


def test_corrupt_cache_file_fails_fast(tmp_path, monkeypatch):
    from enn import semantic_outputs

    calls = []

    def encode(texts, model):
        calls.append(texts)
        return OutputEncoder().encode(texts)

    monkeypatch.setattr(semantic_outputs, "semantic_encode", encode)
    kwargs = {"cache_dir": tmp_path}
    SemanticOutputProjection.from_metadata(["a", "b"], **kwargs)
    (path,) = tmp_path.glob("*.npz")
    path.write_bytes(b"not an npz file")
    with pytest.raises(ValueError):
        SemanticOutputProjection.from_metadata(["a", "b"], **kwargs)
    assert len(calls) == 1


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


def fit_pinned(x, y, alpha, p=None):
    model = EpistemicNearestNeighbors(
        x[:0],
        y[:0],
        output_projection=p or projection(),
        output_alpha=alpha,
        work_dir=fresh_work_dir(),
    )
    params = enn_fit(
        model,
        k=3,
        num_fit_candidates=3,
        rng=np.random.default_rng(0),
        incremental=model.add(x, y),
    )
    return model, params


def test_alpha_zero_matches_independent_and_constant_outputs():
    x, y = data()
    y[:, 2] = 7
    model, params = fit_pinned(x, y, 0)
    plain = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir())
    expected = plain.posterior(x[:4], params=ENNParams(3, 1, 0))
    fixed = ENNParams(3, 1, 0, output_blend=params.output_blend)
    actual = model.posterior(x[:4], params=fixed)
    np.testing.assert_allclose(actual.mu, expected.mu, atol=1e-12)
    np.testing.assert_allclose(actual.se, expected.se, atol=1e-12)
    _, shared = fit_pinned(x, y, 1)
    assert shared.output_blend.alpha[2] == 0
    np.testing.assert_allclose(model.posterior(x[:4], params=shared).mu[:, 2], 7)


def test_shared_limit_and_opposing_outputs():
    x, y = data()
    model, params = fit_pinned(x, y, 1)
    blend = params.output_blend
    fixed = ENNParams(3, 1, 0, output_blend=blend)
    plain = model.posterior(x[:4], params=ENNParams(3, 1, 0))
    loadings = projection().loadings
    latent = ((plain.mu - blend.center) / blend.scale) @ loadings
    expected = latent @ loadings.T * blend.scale + blend.center
    shared = model.posterior(x[:4], params=fixed)
    np.testing.assert_allclose(shared.mu, expected, atol=1e-12)
    # Opposite standardized outputs cancel in the common semantic direction.
    np.testing.assert_allclose(
        shared.mu[:, :2], np.tile(blend.center[:2], (4, 1)), atol=1e-12
    )


def test_projection_rejects_nonorthonormal_loadings_and_bad_names():
    with pytest.raises(ValueError, match="orthonormal"):
        SemanticOutputProjection(("a", "b", "c"), np.eye(3), np.ones((3, 2)))
    with pytest.raises(ValueError, match="output names"):
        SemanticOutputProjection((1, "b"), np.eye(2), np.zeros((2, 0)))


def test_zero_embeddings_fall_back_to_no_blend():
    x, y = data()
    zero = SemanticOutputProjection.from_embeddings(["a", "b", "c"], np.zeros((3, 2)))
    model, params = fit_pinned(x, y, 1, zero)
    assert params.output_blend is None
    assert model.posterior(x[:0], params=params).mu.shape == (0, 3)


def test_output_alpha_requires_a_projection_and_a_valid_value():
    x, y = data()
    with pytest.raises(ValueError, match="requires output_projection"):
        EpistemicNearestNeighbors(x, y, output_alpha=0.3, work_dir=fresh_work_dir())
    for alpha in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="output_alpha"):
            EpistemicNearestNeighbors(
                x,
                y,
                output_projection=projection(),
                output_alpha=alpha,
                work_dir=fresh_work_dir(),
            )
    with pytest.raises(ValueError):
        EpistemicNearestNeighbors(
            x, y[:, :2], output_projection=projection(), work_dir=fresh_work_dir()
        )


def test_batched_scalar_predictions_match_each_output():
    x, y = data()
    params = ENNParams(3, 1, 0)
    batched = EpistemicNearestNeighbors(x, y, work_dir=fresh_work_dir()).posterior(
        x[:4], params=params
    )
    for output in range(y.shape[1]):
        scalar = EpistemicNearestNeighbors(
            x, y[:, output : output + 1], work_dir=fresh_work_dir()
        ).posterior(x[:4], params=params)
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
