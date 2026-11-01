from __future__ import annotations

import numpy as np
import pytest


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
        enn = ENNSurrogateConfig(k=3, fit=ENNFitConfig(), work_dir=tmp_path / "store")
        config = turbo_enn_config(enn=enn, num_init=4, candidates=candidates)
    else:
        config = turbo_zero_config(num_init=4, candidates=candidates)
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    optimizer = create_optimizer(bounds=bounds, config=config, seed=0)
    for _ in range(3):
        x = optimizer.ask(2)
        optimizer.tell(x, -np.sum((x - 0.5) ** 2, axis=1, keepdims=True))
