"""Input-output priors depend on both sets of metadata, never observed targets."""

import numpy as np
import pytest

from enn import SemanticInputTransform


class RelationshipEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = {"heat": [2, 0, 0], "cost": [0, 3, 0], "noise": [0, 0, 4]}
        return np.array([vectors[t] for t in texts], dtype=float)


def make_relationships(**kwargs):
    return SemanticInputTransform.from_input_output(
        ["heat", "cost", "noise"],
        [[0, 2], [10, 20], [-1, 1]],
        ["temperature", "price"],
        output_descriptions=["heat", "cost"],
        encoder=RelationshipEncoder(),
        **kwargs,
    )


def test_shared_output_features_have_expected_geometry_and_keep_inputs():
    transform = make_relationships(relative_weight=2 / 3)
    p = transform.projection
    assert p.shape == (3, 2)
    np.testing.assert_allclose(p.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum(p**2), 2)
    x = np.array([[2, 10, 0], [0, 20, 1]])
    z = np.array([[0.5, -0.5, 0], [-0.5, 0.5, 0.5]])
    expected = np.column_stack((z, z @ transform.weighted_projection))
    np.testing.assert_allclose(transform.transform(x), expected)
    np.testing.assert_allclose(
        transform.transform(x[:, [2, 0, 1]], names=["noise", "heat", "cost"]), expected
    )
    np.testing.assert_allclose(np.sum(transform.weighted_projection**2), 2)
    disabled = make_relationships(relative_weight=0)
    np.testing.assert_array_equal(disabled.transform(x)[:, 3:], 0)


def test_joint_deduplication_files_fallback_and_one_native_encode(
    tmp_path, monkeypatch
):
    from enn import _rust

    encoder = RelationshipEncoder()
    monkeypatch.setattr(
        _rust, "semantic_encode", lambda texts, model: encoder.encode(texts)
    )
    path = tmp_path / "temperature description.txt"
    path.write_text(" heat \n")
    transform = SemanticInputTransform.from_input_output(
        ["heat", "cost", "noise"],
        [[0, 1]] * 3,
        ["temperature", "cost"],
        output_descriptions=[path.as_uri(), "  "],
    )
    assert encoder.calls == [["heat", "cost", "noise"]]
    transform.transform([[0, 0, 1]])
    assert len(encoder.calls) == 1
    np.testing.assert_array_equal(transform.projection, make_relationships().projection)


def test_output_metadata_changes_prior_and_identical_inputs_have_no_variance():
    encoder = RelationshipEncoder()
    a = SemanticInputTransform.from_input_output(
        ["heat", "cost"], [[0, 1]] * 2, ["heat"], encoder=encoder
    )
    b = SemanticInputTransform.from_input_output(
        ["heat", "cost"], [[0, 1]] * 2, ["cost"], encoder=encoder
    )
    np.testing.assert_allclose(
        a.projection @ a.projection.T, [[0.5, -0.5], [-0.5, 0.5]]
    )
    np.testing.assert_allclose(
        b.projection @ b.projection.T, a.projection @ a.projection.T
    )
    same = SemanticInputTransform.from_input_output(
        ["heat"], [[0, 1]], ["heat"], encoder=encoder
    )
    assert same.projection.shape == (1, 0)
    xx = SemanticInputTransform(["heat"], [[0, 1]], encoder=encoder)
    assert xx.projection.shape == (1, 0)


def test_relationship_cache_needs_no_encoder(tmp_path, monkeypatch):
    from enn import _rust

    transform = make_relationships()
    path = tmp_path / "relationships.npz"
    transform.save(path)
    monkeypatch.setattr(_rust, "semantic_encode", None)
    cached = SemanticInputTransform.load(path)
    np.testing.assert_array_equal(
        cached.transform([[1, 15, 0]]), transform.transform([[1, 15, 0]])
    )
    assert not cached.projection.flags.writeable


@pytest.mark.parametrize("outputs", [[], [""], ["heat", "heat"], [None]])
def test_invalid_output_names(outputs):
    with pytest.raises(ValueError, match="output names"):
        SemanticInputTransform.from_input_output(["heat"], [[0, 1]], outputs)


@pytest.mark.parametrize("descriptions,output_descriptions", [([], None), (None, [])])
def test_description_lengths_validated_before_encoding(
    descriptions, output_descriptions
):
    with pytest.raises(ValueError, match="description"):
        SemanticInputTransform.from_input_output(
            ["heat"], [[0, 1]], ["cost"], descriptions, output_descriptions
        )


@pytest.mark.parametrize(
    "embeddings",
    [np.array(1), np.ones((1, 2)), np.ones((2, 0)), [[np.nan], [1]], [[np.inf], [1]]],
)
def test_invalid_encoder_results(embeddings):
    class InvalidEncoder:
        def encode(self, texts):
            return embeddings

    with pytest.raises(ValueError):
        SemanticInputTransform.from_input_output(
            ["heat"], [[0, 1]], ["cost"], encoder=InvalidEncoder()
        )


def test_zero_embeddings_add_zero_features():
    class ZeroEncoder:
        def encode(self, texts):
            return np.zeros((len(texts), 2))

    transform = SemanticInputTransform.from_input_output(
        ["heat"], [[0, 1]], ["cost"], encoder=ZeroEncoder()
    )
    np.testing.assert_array_equal(transform.transform([[0], [1]]), [[-0.5], [0.5]])


def test_cosine_cutoff_retains_magnitude_and_signed_contrasts():
    from enn._rust import semantic_input_output_projection

    # Two large opposite contrasts dominate two weaker orthogonal contrasts.
    embeddings = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0.1, np.sqrt(0.99)],
            [0, -0.1, np.sqrt(0.99)],
            [0, 1, 0],
        ]
    )
    rows = [0, 1, 2, 3, 0, 4]
    full = semantic_input_output_projection(embeddings, rows, 4, 1)
    truncated = semantic_input_output_projection(embeddings, rows, 4, 0.9)
    assert full.shape == (4, 2) and truncated.shape == (4, 1)
    np.testing.assert_allclose(np.sum(full**2), 2)
    np.testing.assert_allclose(np.sum(truncated**2), 1)
    assert truncated[0, 0] * truncated[1, 0] < 0
    np.testing.assert_allclose(truncated[2:], 0, atol=1e-12)
