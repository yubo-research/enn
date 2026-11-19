from __future__ import annotations

import numpy as np
import pytest

from evals.harness.work_dirs import fresh_work_dir


def test_root_exports_resolve() -> None:
    import enn

    for name in (
        "ENNParams",
        "PosteriorFlags",
        "ENNNormal",
        "ENNSurrogateConfig",
        "ENNFitConfig",
        "NoSurrogateConfig",
        "MultiObjectiveConfig",
        "CandidateGenConfig",
        "TRLengthConfig",
        "Rescalarize",
        "RAASPDriver",
        "ENNMetricLearning",
    ):
        assert name in enn.__all__
        assert getattr(enn, name).__name__ == name


def test_model_workflow_uses_only_root_imports(tmp_path) -> None:
    from enn import (
        ENNNormal,
        ENNParams,
        ENNStatefulFitter,
        EpistemicNearestNeighbors,
        PosteriorFlags,
    )

    rng = np.random.default_rng(0)
    train_x = rng.uniform(size=(30, 2))
    train_y = train_x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(train_x, train_y, work_dir=tmp_path / "store")
    params = ENNStatefulFitter(3, seed=0).ask(
        model, num_fit_candidates=4, num_fit_samples=4
    )
    assert isinstance(params, ENNParams)
    post = model.posterior(train_x[:2], params=params, flags=PosteriorFlags())
    assert isinstance(post, ENNNormal)
    assert post.mu.shape == (2, 1)


@pytest.mark.parametrize("surrogate", ["enn", "none"])
def test_optimizer_workflow_uses_only_root_imports(tmp_path, surrogate) -> None:
    from enn import (
        CandidateGenConfig,
        ENNFitConfig,
        ENNSurrogateConfig,
        create_optimizer,
        turbo_enn_config,
        turbo_zero_config,
    )

    candidates = CandidateGenConfig(max_candidates=64)
    if surrogate == "enn":
        enn = ENNSurrogateConfig(k=3, fit=ENNFitConfig())
        config = turbo_enn_config(enn=enn, num_init=4, candidates=candidates)
    else:
        config = turbo_zero_config(num_init=4, candidates=candidates)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    optimizer = create_optimizer(
        bounds=bounds, config=config, seed=0, work_dir=fresh_work_dir()
    )
    for _ in range(3):
        x = optimizer.ask(2)
        optimizer.tell(x, -np.sum((x - 0.5) ** 2, axis=1, keepdims=True))


def test_one_public_optimizer_type() -> None:
    import enn

    for name in (
        "create_optimizer_enn",
        "create_optimizer_zero",
        "create_optimizer_lhd",
    ):
        assert name not in enn.__all__
        assert not hasattr(enn, name)
    assert enn.Optimizer.__name__ == "Optimizer"


CONFIGS = ["turbo_enn", "turbo_zero", "lhd_only"]


def _config(kind: str):
    from enn import (
        ENNSurrogateConfig,
        lhd_only_config,
        turbo_enn_config,
        turbo_zero_config,
    )

    if kind == "turbo_enn":
        return turbo_enn_config(enn=ENNSurrogateConfig(k=3), num_init=2)
    if kind == "turbo_zero":
        return turbo_zero_config(num_init=2)
    return lhd_only_config(num_init=2)


@pytest.mark.parametrize("kind", CONFIGS)
def test_work_dir_is_required_in_every_mode(kind) -> None:
    from enn import create_optimizer

    with pytest.raises(TypeError, match="work_dir"):
        create_optimizer(bounds=np.array([[0.0, 1.0]]), config=_config(kind), seed=0)


@pytest.mark.parametrize("kind", CONFIGS)
def test_public_optimizer_surface(tmp_path, kind) -> None:
    from enn import Optimizer, Telemetry, create_optimizer

    bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
    optimizer = create_optimizer(
        bounds=bounds, config=_config(kind), seed=0, work_dir=tmp_path / "w"
    )
    assert isinstance(optimizer, Optimizer)
    np.testing.assert_array_equal(optimizer.bounds, bounds)
    assert optimizer.x_obs.shape == (0, 2)
    assert optimizer.y_obs.shape == (0, 1)
    assert optimizer.incumbent_x is None
    x = optimizer.ask(2)
    optimizer.tell(x, np.ones((2, 1)))
    assert optimizer.tr_obs_count == 2
    np.testing.assert_allclose(optimizer.x_obs, x)
    assert optimizer.y_obs.shape == (2, 1)
    assert isinstance(optimizer.telemetry(), Telemetry)
    assert isinstance(optimizer.tr_length, float)
    assert optimizer.init_progress is None or len(optimizer.init_progress) == 2


def test_building_a_config_does_not_touch_the_filesystem(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _config("turbo_enn")
    assert list(tmp_path.iterdir()) == []
