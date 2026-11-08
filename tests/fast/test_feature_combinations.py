"""Feature combinations obey the same contract in supervised and optimizer use."""

import numpy as np
import pytest

from enn import (
    CatEncoder,
    CatEncoding,
    DimKind,
    EpistemicNearestNeighbors,
    create_optimizer,
)
from enn.enn.enn_params import ENNParams
from enn.turbo.config import ENNFitConfig, ENNSurrogateConfig
from enn.turbo.config.factory import turbo_enn_config
from evals.harness.work_dirs import fresh_work_dir


def problem(encoding):
    rng = np.random.default_rng(15)
    x = rng.uniform(0, 1, (24, 3))
    if encoding is not None:
        x[:, 2] = np.arange(len(x)) % 3
        encoder = CatEncoder(
            [DimKind.CONTINUOUS] * 2 + [DimKind.CATEGORICAL],
            [None, None, 3],
            encoding=encoding,
            separation=1,
        )
        x = encoder.encode(x)
    bounds = np.column_stack((np.full(x.shape[1], -1.0), np.full(x.shape[1], 2.0)))
    return x, np.sin(x[:, :1] * 3), bounds


def tied_groups(x, use_ties, weighting):
    return getattr(x, "tied_dims", ((0, 1),)) if use_ties and weighting else ()


@pytest.mark.parametrize("weighting", [False, True])
@pytest.mark.parametrize("use_ties", [False, True])
@pytest.mark.parametrize(
    "encoding", [None, CatEncoding.ONE_HOT, CatEncoding.RADEMACHER]
)
def test_supervised_feature_combinations(tmp_path, weighting, encoding, use_ties):
    x, y, _ = problem(encoding)
    tied = tied_groups(x, use_ties, weighting)
    if not use_ties:
        x = np.asarray(x)
    actual = EpistemicNearestNeighbors(
        x[:16], y[:16], weighting=weighting, tied_dims=tied, work_dir=tmp_path
    )
    actual.add(x[16:], y[16:])
    posterior = actual.posterior(x[:3], params=ENNParams(3, 1.0, 0.1))
    assert np.isfinite(posterior.mu).all() and np.isfinite(posterior.se).all()
    original, values, _ = actual.train_rows_at([23, 0, 17])
    np.testing.assert_allclose(original, x[[23, 0, 17]], atol=1e-14)
    np.testing.assert_allclose(values, y[[23, 0, 17]])
    assert actual.weighting is weighting
    assert actual.tied_dims == tied


@pytest.mark.parametrize("weighting", [False, True])
@pytest.mark.parametrize("use_ties", [False, True])
@pytest.mark.parametrize(
    "encoding", [None, CatEncoding.ONE_HOT, CatEncoding.RADEMACHER]
)
def test_optimizer_feature_combinations(weighting, encoding, use_ties):
    x, y, bounds = problem(encoding)
    tied = tied_groups(x, use_ties, weighting)
    if not use_ties:
        x = np.asarray(x)
    surrogate = ENNSurrogateConfig(
        weighting=weighting,
        tied_dims=tied,
        fit=ENNFitConfig(num_fit_samples=5, num_fit_candidates=3),
    )
    optimizer = create_optimizer(
        bounds=bounds,
        config=turbo_enn_config(enn=surrogate, num_init=3),
        seed=int(np.random.default_rng(4).integers(2**63 - 1)),
        work_dir=fresh_work_dir(),
    )
    optimizer.tell(x[:16], y[:16])
    proposed = optimizer.ask(2)
    assert proposed.shape == (2, x.shape[1]) and np.isfinite(proposed).all()
    assert np.all((proposed >= -1) & (proposed <= 2))
    optimizer.tell(x[16:], y[16:])
    assert np.isfinite(optimizer.ask(1)).all()
    np.testing.assert_allclose(optimizer.x_obs, x, atol=1e-14)


def test_work_dir_is_required_even_when_the_environment_names_one(
    tmp_path, monkeypatch
):
    from enn._rust import EpistemicNearestNeighbors as RustENN

    directory = tmp_path / "environment-only"
    monkeypatch.setenv("ENN_WORK_DIR", str(directory))
    x, y = np.zeros((2, 1)), np.ones((2, 1))
    for wrapper in [EpistemicNearestNeighbors, RustENN]:
        with pytest.raises(TypeError, match="work_dir"):
            wrapper(x, y)
        assert len(wrapper(x, y, work_dir=str(tmp_path / wrapper.__module__))) == 2
    assert not directory.exists()


@pytest.mark.parametrize("obsolete", ["scale_x", "metric_learning", "enn_storage"])
def test_native_optimizer_rejects_replaced_options(obsolete):
    from enn._rust import create_optimizer_enn

    with pytest.raises(ValueError, match="Unknown config override key"):
        create_optimizer_enn(
            np.array([[0.0, 1.0]]), fresh_work_dir(), config_overrides={obsolete: True}
        )
