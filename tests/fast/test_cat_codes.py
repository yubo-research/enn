from __future__ import annotations

import numpy as np
import pytest

from enn.enn.cat_codes import (
    CatEncoder,
    CatEncoding,
    DimKind,
    column_plan,
    log_code_width,
)
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams


def test_log_code_width_tracks_domain_size() -> None:
    assert log_code_width(1) == 1
    assert log_code_width(2) == 9
    assert log_code_width(1000) == 27
    assert log_code_width(2**30, dim_cap=40) == 40
    with pytest.raises(ValueError, match="card"):
        log_code_width(0)
    with pytest.raises(ValueError, match="dim_cap"):
        log_code_width(4, dim_cap=0)


def test_auto_switches_at_sixteen_categories() -> None:
    assert column_plan(15) == (15, "onehot")
    assert column_plan(16) == (log_code_width(16), "code")
    assert column_plan(3, CatEncoding.ONE_HOT) == (3, "onehot")
    assert column_plan(3, CatEncoding.RADEMACHER) == (log_code_width(3), "code")
    assert column_plan(1000, CatEncoding.ONE_HOT) == (1000, "onehot")


def test_same_category_matches_and_different_categories_are_equidistant() -> None:
    encoder = CatEncoder(
        (DimKind.CATEGORICAL,),
        encoding=CatEncoding.RADEMACHER,
        code_dim=16,
        seed=3,
        separation=1,
    )
    labels = np.arange(32).reshape(-1, 1)
    coded = encoder.encode(labels)
    same = coded[0] - encoder.encode(np.array([[0]]))[0]
    assert np.linalg.norm(same) == pytest.approx(0.0)
    gaps = np.linalg.norm(coded[0] - coded[1:], axis=1)
    assert np.mean(np.square(gaps)) == pytest.approx(2.0, abs=0.35)
    assert gaps.min() > 0.0


def test_hash_width_ignores_cardinality_and_column_count() -> None:
    kinds = (DimKind.CONTINUOUS, *(DimKind.CATEGORICAL,) * 40)
    wide = CatEncoder(kinds, hash_dim=16, seed=1)
    values = np.zeros((5, 41))
    values[:, 0] = 1.0
    encoded = wide.encode(values)
    assert encoded.shape == (5, 17)
    assert wide.tied_dims() == (tuple(range(1, 17)),)
    assert encoded.tied_dims == wide.tied_dims()
    assert wide.styles == ()
    np.testing.assert_array_equal(encoded[:, 0], 1.0)
    np.testing.assert_array_equal(encoded, wide.encode(values))
    empty = wide.encode(np.zeros((0, 41)))
    assert empty.shape == (0, 17)


def test_tied_groups_follow_variables_and_snap_recovers_ids() -> None:
    kinds = (
        DimKind.CONTINUOUS,
        DimKind.CATEGORICAL,
        DimKind.CONTINUOUS,
        DimKind.CATEGORICAL,
    )
    encoder = CatEncoder(kinds, encoding=CatEncoding.RADEMACHER, code_dim=5, seed=9)
    assert encoder.tied_dims() == (tuple(range(1, 6)), tuple(range(7, 12)))
    raw = np.array([[0.2, 10, -3.0, -4], [1.0, 7, 1.0, 8], [0.0, 10, 0.5, 8]])
    encoded = encoder.encode(raw)
    recovered = encoder.snap(encoded, [np.array([10, 7, 3]), np.array([-4, 8])])
    np.testing.assert_array_equal(recovered, raw[:, [1, 3]])
    noisy = encoded.copy()
    noisy[:, 1:] += 0.05
    np.testing.assert_array_equal(
        encoder.snap(noisy, [np.array([10, 7, 3]), np.array([-4, 8])]),
        raw[:, [1, 3]],
    )


def test_encoder_rejects_mixed_layouts_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="code_dim does not use cards"):
        CatEncoder(
            (DimKind.CATEGORICAL,),
            cards=(4,),
            encoding=CatEncoding.RADEMACHER,
            code_dim=3,
        )
    with pytest.raises(ValueError, match="need cards"):
        CatEncoder((DimKind.CATEGORICAL, DimKind.CATEGORICAL))
    with pytest.raises(ValueError, match="no categorical"):
        CatEncoder((DimKind.CONTINUOUS,), hash_dim=4)
    with pytest.raises(ValueError, match="seed"):
        CatEncoder(
            (DimKind.CATEGORICAL,),
            encoding=CatEncoding.RADEMACHER,
            code_dim=2,
            seed=-1,
        )
    with pytest.raises(ValueError, match="kinds"):
        CatEncoder(())
    with pytest.raises(TypeError, match="DimKind"):
        CatEncoder(("categorical",))
    encoder = CatEncoder(
        (DimKind.CONTINUOUS, DimKind.CATEGORICAL),
        encoding=CatEncoding.RADEMACHER,
        code_dim=4,
    )
    with pytest.raises(ValueError, match="integers"):
        encoder.encode(np.array([[1.0, 1.5]]))
    with pytest.raises(ValueError, match="shape"):
        encoder.encode(np.array([[1.0]]))
    with pytest.raises(ValueError, match="hashed"):
        CatEncoder((DimKind.CATEGORICAL,), hash_dim=8).snap(
            np.zeros((1, 8)), [np.array([0])]
        )
    with pytest.raises(ValueError, match="non-empty"):
        raw = np.array([[0.0, 0]])
        encoder.snap(encoder.encode(raw), [np.array([])])
    with pytest.raises(TypeError, match="encoding"):
        CatEncoder((DimKind.CONTINUOUS,), encoding="auto")
    with pytest.raises(ValueError, match="only one"):
        CatEncoder(
            (DimKind.CATEGORICAL,),
            encoding=CatEncoding.RADEMACHER,
            code_dim=4,
            hash_dim=4,
        )
    with pytest.raises(ValueError, match="hash_dim already"):
        CatEncoder((DimKind.CATEGORICAL,), encoding=CatEncoding.ONE_HOT, hash_dim=4)
    with pytest.raises(ValueError, match="Rademacher width"):
        CatEncoder((DimKind.CATEGORICAL,), (4,), code_dim=4)


def test_codes_ignore_arbitrary_id_magnitudes() -> None:
    """A wide irrelevant id must not swamp binary columns the way raw integers do."""
    rng = np.random.default_rng(0)
    n = 800
    bits = rng.integers(0, 2, size=(n, 4))
    noise = rng.integers(0, 10_000, size=(n, 1))
    raw = np.hstack([bits, noise])
    y = bits.sum(axis=1) + 0.05 * rng.standard_normal(n)
    query_bits = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    query = np.hstack([query_bits, np.zeros((2, 1), dtype=np.int64)])
    coded = CatEncoder(
        (DimKind.CATEGORICAL,) * 5,
        (2, 2, 2, 2, 10_000),
        encoding=CatEncoding.AUTO,
        seed=2,
    )
    model = EpistemicNearestNeighbors(
        coded.encode(raw), y.reshape(-1, 1), tied_dims=coded.tied_dims()
    )
    mu = model.posterior(coded.encode(query), params=ENNParams(15, 0.05, 0.01)).mu[:, 0]
    ordinal = EpistemicNearestNeighbors(raw.astype(float), y.reshape(-1, 1))
    mu_ord = ordinal.posterior(
        query.astype(float), params=ENNParams(15, 0.05, 0.01)
    ).mu[:, 0]
    target = query_bits.sum(axis=1).astype(float)
    assert np.mean((mu - target) ** 2) < 0.15
    assert np.mean((mu - target) ** 2) < np.mean((mu_ord - target) ** 2)


def test_user_chooses_one_hot_rademacher_or_auto() -> None:
    kinds = (DimKind.CONTINUOUS, DimKind.CATEGORICAL, DimKind.CATEGORICAL)
    cards = (None, 3, 1000)
    auto = CatEncoder(kinds, cards, seed=4)
    assert auto.encoding is CatEncoding.AUTO
    assert auto.styles == ("onehot", "code")
    assert auto.code_dims == (3, log_code_width(1000))
    row = np.array([[1.5, 2, 50]])
    encoded = auto.encode(row)
    np.testing.assert_array_equal(encoded[0, 1:4], [0.0, 0.0, auto.separation])
    assert encoded[0, 0] == pytest.approx(1.5)
    snapped = auto.snap(encoded, [np.array([0, 2]), np.array([50, 7, 9])])
    np.testing.assert_array_equal(snapped, [[2, 50]])
    forced = CatEncoder(kinds, cards, encoding=CatEncoding.ONE_HOT, separation=1)
    assert forced.styles == ("onehot", "onehot")
    assert forced.num_dim == 1 + 3 + 1000
    wide = CatEncoder(kinds, cards, encoding=CatEncoding.RADEMACHER, seed=4)
    assert wide.styles == ("code", "code")
    plain = EpistemicNearestNeighbors(np.ones((4, 3)), np.ones((4, 1)))
    assert plain.tied_dims == ()


def test_default_separation_lengthens_a_mismatch() -> None:
    unit = CatEncoder((DimKind.CATEGORICAL,), (1000,), seed=0, separation=1)
    default = CatEncoder((DimKind.CATEGORICAL,), (1000,), seed=0)
    labels = np.array([[0], [1]])
    gap = np.sum((default.encode(labels)[0] - default.encode(labels)[1]) ** 2)
    unit_gap = np.sum((unit.encode(labels)[0] - unit.encode(labels)[1]) ** 2)
    assert gap == pytest.approx(100 * unit_gap)
    with pytest.raises(ValueError, match="separation"):
        CatEncoder(
            (DimKind.CATEGORICAL,),
            encoding=CatEncoding.RADEMACHER,
            code_dim=4,
            separation=0,
        )


def test_one_row_and_empty_batch() -> None:
    encoder = CatEncoder((DimKind.CATEGORICAL, DimKind.CATEGORICAL), (3, 1), seed=0)
    one = encoder.encode(np.array([[2, 0]]))
    assert one.shape == (1, encoder.num_dim)
    empty = encoder.encode(np.zeros((0, 2), dtype=np.int64))
    assert empty.shape == (0, encoder.num_dim)
    identity = CatEncoder((DimKind.CONTINUOUS, DimKind.CONTINUOUS))
    copied = identity.encode(np.array([[1.25, -2.0]]))
    np.testing.assert_array_equal(copied, [[1.25, -2.0]])
    assert identity.tied_dims() == ()


def test_rejects_bad_cards_and_one_hot_ids() -> None:
    with pytest.raises(ValueError, match="continuous column"):
        CatEncoder((DimKind.CONTINUOUS, DimKind.CATEGORICAL), (2, 2))
    with pytest.raises(ValueError, match="length"):
        CatEncoder((DimKind.CATEGORICAL,), (2, 3))
    with pytest.raises(ValueError, match="card"):
        CatEncoder((DimKind.CATEGORICAL,), (0,))
    encoder = CatEncoder((DimKind.CATEGORICAL,), (3,), encoding=CatEncoding.ONE_HOT)
    with pytest.raises(ValueError, match="one-hot"):
        encoder.encode(np.array([[3]]))
    with pytest.raises(ValueError, match="one-hot"):
        encoder.snap(encoder.encode(np.array([[1]])), [np.array([0, 4])])
    with pytest.raises(ValueError, match="domains"):
        encoder.snap(encoder.encode(np.array([[1]])), [])
    with pytest.raises(ValueError, match="encoded shape"):
        encoder.snap(np.zeros((1, 2)), [np.array([0])])
    with pytest.raises(ValueError, match="hash_dim"):
        CatEncoder((DimKind.CATEGORICAL,), hash_dim=0)
    with pytest.raises(ValueError, match="code_dim"):
        CatEncoder((DimKind.CATEGORICAL,), encoding=CatEncoding.RADEMACHER, code_dim=0)


def test_public_names() -> None:
    import enn

    assert enn.CatEncoder is CatEncoder
    assert enn.CatEncoding.AUTO is CatEncoding.AUTO
    assert enn.DimKind.CONTINUOUS is DimKind.CONTINUOUS
    assert enn.DimKind.CATEGORICAL is DimKind.CATEGORICAL


def test_integer_category_identity_survives_values_above_float_precision():
    labels = np.array([[2**53], [2**53 + 1], [-(2**63)], [2**63 - 1]], dtype=np.int64)
    encoder = CatEncoder(
        [DimKind.CATEGORICAL], encoding=CatEncoding.RADEMACHER, code_dim=64
    )
    codes = encoder.encode(labels)
    assert not np.array_equal(codes[0], codes[1])
    np.testing.assert_array_equal(encoder.snap(codes, [labels[:, 0]]), labels)


def test_mixed_object_rows_preserve_large_category_labels():
    labels = np.array([2**53, 2**53 + 1, -(2**63), 2**63 - 1], dtype=np.int64)
    rows = np.empty((4, 2), dtype=object)
    rows[:, 0] = [0.25, 0.5, 0.75, 1.0]
    rows[:, 1] = labels
    encoder = CatEncoder(
        [DimKind.CONTINUOUS, DimKind.CATEGORICAL],
        encoding=CatEncoding.RADEMACHER,
        code_dim=64,
    )
    codes = encoder.encode(rows)
    np.testing.assert_array_equal(codes[:, 0], rows[:, 0])
    assert not np.array_equal(codes[0, 1:], codes[1, 1:])
    np.testing.assert_array_equal(encoder.snap(codes, [labels])[:, 0], labels)
