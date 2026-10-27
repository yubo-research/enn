"""Feature combinations obey the same contract in supervised and optimizer use."""

import numpy as np
import pytest

from enn import (
    CatEncoder,
    CatEncoding,
    DimKind,
    EpistemicNearestNeighbors,
    SemanticInputTransform,
    create_optimizer,
)
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import ENNParams
from enn.turbo.config import ENNFitConfig, ENNIndexDriver, ENNSurrogateConfig
from enn.turbo.config.factory import turbo_enn_config


class Encoder:
    def encode(self, texts):
        return np.eye(len(texts))


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
    transform = SemanticInputTransform(
        [f"x{i}" for i in range(x.shape[1])], bounds, encoder=Encoder()
    )
    return x, np.sin(x[:, :1] * 3), transform


@pytest.mark.parametrize("driver", list(ENNIndexDriver))
@pytest.mark.parametrize("weighting", [False, True])
@pytest.mark.parametrize("use_ties", [False, True])
@pytest.mark.parametrize("semantics", ["none", "both", "only"])
@pytest.mark.parametrize(
    "encoding", [None, CatEncoding.ONE_HOT, CatEncoding.RADEMACHER]
)
def test_supervised_feature_combinations(
    tmp_path, driver, weighting, semantics, encoding, use_ties
):
    x, y, transform = problem(encoding)
    tied = (
        (((0, 1),) if semantics == "only" else getattr(x, "tied_dims", ((0, 1),)))
        if use_ties
        else ()
    )
    if not use_ties:
        x = np.asarray(x)
    kwargs = {"index_driver": driver, "weighting": weighting, "tied_dims": tied}
    if driver is ENNIndexDriver.BPANN_DISK:
        kwargs["work_dir"] = tmp_path / "actual"
    if semantics != "none":
        kwargs.update(input_transform=transform, semantic_only=semantics == "only")
    actual = EpistemicNearestNeighbors(x[:16], y[:16], **kwargs)
    features = (
        x
        if semantics == "none"
        else transform.transform(x, semantic_only=semantics == "only")
    )
    if driver is ENNIndexDriver.BPANN_DISK:
        kwargs["work_dir"] = tmp_path / "manual"
    kwargs.pop("input_transform", None)
    kwargs.pop("semantic_only", None)
    manual = EpistemicNearestNeighbors(np.asarray(features[:16]), y[:16], **kwargs)
    actual.add(x[16:], y[16:])
    manual.add(features[16:], y[16:])
    params = ENNParams(3, 1.0, 0.1)
    a = actual.posterior(x[:3], params=params)
    b = manual.posterior(features[:3], params=params)
    np.testing.assert_allclose(a.mu, b.mu)
    np.testing.assert_allclose(a.se, b.se)
    original, values, _ = actual.train_rows_at([23, 0, 17])
    np.testing.assert_allclose(original, x[[23, 0, 17]], atol=1e-14)
    np.testing.assert_allclose(values, y[[23, 0, 17]])
    assert actual.weighting is weighting
    assert actual.tied_dims == tied


@pytest.mark.parametrize("driver", list(ENNIndexDriver))
@pytest.mark.parametrize("weighting", [False, True])
@pytest.mark.parametrize("use_ties", [False, True])
@pytest.mark.parametrize("semantics", ["none", "both", "only"])
@pytest.mark.parametrize(
    "encoding", [None, CatEncoding.ONE_HOT, CatEncoding.RADEMACHER]
)
def test_optimizer_feature_combinations(
    tmp_path, driver, weighting, semantics, encoding, use_ties
):
    x, y, transform = problem(encoding)
    tied = (
        (((0, 1),) if semantics == "only" else getattr(x, "tied_dims", ((0, 1),)))
        if use_ties
        else ()
    )
    if not use_ties:
        x = np.asarray(x)
    surrogate = ENNSurrogateConfig(
        index_driver=driver,
        weighting=weighting,
        work_dir=tmp_path if driver is ENNIndexDriver.BPANN_DISK else None,
        tied_dims=tied,
        input_transform=transform if semantics != "none" else None,
        semantic_only=semantics == "only",
        fit=ENNFitConfig(num_fit_samples=5, num_fit_candidates=3),
    )
    optimizer = create_optimizer(
        bounds=transform.bounds,
        config=turbo_enn_config(enn=surrogate, num_init=3),
        rng=np.random.default_rng(4),
    )
    optimizer.tell(x[:16], y[:16])
    proposed = optimizer.ask(2)
    assert proposed.shape == (2, x.shape[1]) and np.isfinite(proposed).all()
    assert np.all((proposed >= -1) & (proposed <= 2))
    optimizer.tell(x[16:], y[16:])
    assert np.isfinite(optimizer.ask(1)).all()
    np.testing.assert_allclose(optimizer._x_obs, x, atol=1e-14)


def test_semantic_only_fitting_and_invalid_add_preserve_original_rows():
    x, y, transform = problem(None)
    model = EpistemicNearestNeighbors(
        x[:12], y[:12], input_transform=transform, semantic_only=True, weighting=True
    )
    with pytest.raises(ValueError):
        model.add(x[12:13], np.zeros((1, 2)))
    assert len(model) == 12
    np.testing.assert_allclose(model.train_rows_at([11])[0], x[11:12])
    params = enn_fit(
        model,
        k=3,
        num_fit_samples=4,
        num_fit_candidates=3,
        rng=np.random.default_rng(1),
    )
    assert np.isfinite(model.posterior(x[:2], params=params).mu).all()


def test_disk_directory_is_explicit_even_when_environment_is_set(tmp_path, monkeypatch):
    from enn._rust import EpistemicNearestNeighbors as RustENN

    directory = tmp_path / "environment-only"
    monkeypatch.setenv("ENN_WORK_DIR", str(directory))
    x, y = np.zeros((2, 1)), np.ones((2, 1))
    for wrapper, driver in [
        (EpistemicNearestNeighbors, ENNIndexDriver.BPANN_DISK),
        (RustENN, "BPANN_DISK"),
    ]:
        with pytest.raises(ValueError, match="explicit work_dir"):
            wrapper(x, y, index_driver=driver)
        assert len(wrapper(x, y)) == 2
    assert not directory.exists()


@pytest.mark.parametrize("obsolete", ["scale_x", "metric_learning", "enn_storage"])
def test_native_optimizer_rejects_replaced_options(obsolete):
    from enn._rust import create_optimizer_enn

    with pytest.raises(ValueError, match="Unknown config override key"):
        create_optimizer_enn(np.array([[0.0, 1.0]]), config_overrides={obsolete: True})


@pytest.mark.parametrize("weighting", [False, True])
def test_semantic_only_disk_reopen_preserves_original_observations(tmp_path, weighting):
    x, y, transform = problem(None)
    kwargs = {
        "index_driver": ENNIndexDriver.BPANN_DISK,
        "work_dir": tmp_path,
        "weighting": weighting,
        "input_transform": transform,
        "semantic_only": True,
    }
    model = EpistemicNearestNeighbors(x[:12], y[:12], **kwargs)
    model.add(x[12:18], y[12:18])
    model.persist_index_to_disk()
    del model
    reopened = EpistemicNearestNeighbors(
        np.empty((0, x.shape[1])), np.empty((0, 1)), **kwargs
    )
    original, observations, _ = reopened.train_rows_at([17, 0, 12])
    np.testing.assert_array_equal(original, x[[17, 0, 12]])
    np.testing.assert_allclose(observations, y[[17, 0, 12]])
    reopened.add(x[18:], y[18:])
    reopened.persist_index_to_disk()
    params = enn_fit(
        reopened,
        k=3,
        num_fit_samples=4,
        num_fit_candidates=3,
        rng=np.random.default_rng(1),
    )
    assert np.isfinite(reopened.posterior(x[:2], params=params).mu).all()
    del reopened
    again = EpistemicNearestNeighbors(
        np.empty((0, x.shape[1])), np.empty((0, 1)), **kwargs
    )
    np.testing.assert_array_equal(again.train_rows_at([23, 0])[0], x[[23, 0]])
