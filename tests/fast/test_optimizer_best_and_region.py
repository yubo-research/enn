import numpy as np
import pytest

from enn import (
    ENNFitConfig,
    ENNSurrogateConfig,
    Optimizer,
    create_optimizer,
    lhd_only_config,
    turbo_enn_config,
    turbo_zero_config,
)
from enn.turbo.config import (
    MorboTRConfig,
    MultiObjectiveConfig,
    NoTRConfig,
    TRLengthConfig,
    TurboTRConfig,
)

BOUNDS = np.array([[0.0, 1.0]] * 3)
SHORT_REGION = TurboTRConfig(TRLengthConfig(length_init=0.1, length_min=0.05))
SMALL_REGION = TurboTRConfig(TRLengthConfig(length_init=0.03, length_min=0.01))


def _objective(x):
    return -np.linalg.norm(x - 0.3, axis=1, keepdims=True)


def _enn(**kwargs):
    enn = ENNSurrogateConfig(k=3, fit=ENNFitConfig(num_fit_samples=5))
    return turbo_enn_config(enn=enn, num_init=4, **kwargs)


def _run(optimizer, steps):
    best, best_x = -np.inf, None
    for _ in range(steps):
        x = optimizer.ask(2)
        y = _objective(x)
        optimizer.tell(x, y)
        j = int(np.argmax(y))
        if y[j, 0] > best:
            best, best_x = y[j, 0], x[j].copy()
    return best, best_x


@pytest.mark.parametrize(
    "config",
    [
        turbo_zero_config(num_init=4, trust_region=SHORT_REGION),
        _enn(trust_region=SHORT_REGION),
    ],
    ids=["zero", "enn"],
)
def test_best_observation_outlives_restarts_and_a_reopen(tmp_path, config):
    work_dir = tmp_path / "run"
    optimizer = create_optimizer(
        bounds=BOUNDS, config=config, seed=1, work_dir=work_dir
    )
    assert optimizer.best_x is None
    assert optimizer.best_y is None
    best, best_x = _run(optimizer, 60)
    assert len(optimizer.y_obs) < 120, "the run never restarted"
    assert optimizer.best_y[0] == best
    np.testing.assert_allclose(optimizer.best_x, best_x, atol=1e-12)
    del optimizer
    reopened = Optimizer.open(work_dir)
    assert reopened.best_y[0] == best
    np.testing.assert_allclose(reopened.best_x, best_x, atol=1e-12)


def test_multi_objective_runs_have_no_single_best(tmp_path):
    config = turbo_zero_config(
        num_init=2,
        trust_region=MorboTRConfig(MultiObjectiveConfig(num_metrics=2)),
    )
    optimizer = create_optimizer(
        bounds=BOUNDS, config=config, seed=1, work_dir=tmp_path / "run"
    )
    x = optimizer.ask(2)
    optimizer.tell(x, np.column_stack([x[:, 0], -x[:, 0]]))
    assert optimizer.best_x is None
    assert optimizer.best_y is None


@pytest.mark.parametrize(
    "regional, global_",
    [
        (
            turbo_zero_config(num_init=4, trust_region=SMALL_REGION),
            turbo_zero_config(num_init=4, trust_region=NoTRConfig()),
        ),
        (_enn(trust_region=SMALL_REGION), _enn(trust_region=NoTRConfig())),
    ],
    ids=["zero", "enn"],
)
def test_no_trust_region_searches_the_whole_box(tmp_path, regional, global_):
    def late_spread(config, name):
        optimizer = create_optimizer(
            bounds=BOUNDS, config=config, seed=1, work_dir=tmp_path / name
        )
        spreads = []
        for _ in range(25):
            x = optimizer.ask(4)
            optimizer.tell(x, _objective(x))
            spreads.append(float(x.std(axis=0).mean()))
        return optimizer, np.mean(spreads[-5:])

    shrunk, shrunk_spread = late_spread(regional, "regional")
    unshrunk, unshrunk_spread = late_spread(global_, "global")
    assert shrunk.tr_length < 1.0
    assert unshrunk.tr_length == 1.0
    assert unshrunk_spread > 3 * shrunk_spread
    assert len(unshrunk.x_obs) == 100, "a run without a region must not restart"


def test_no_trust_region_run_resumes_exactly(tmp_path):
    config = _enn(trust_region=NoTRConfig(noise_aware=True))
    kept = create_optimizer(
        bounds=BOUNDS, config=config, seed=2, work_dir=tmp_path / "a"
    )
    path = tmp_path / "b"
    resumed = create_optimizer(bounds=BOUNDS, config=config, seed=2, work_dir=path)
    for _ in range(8):
        x = kept.ask(2)
        np.testing.assert_array_equal(resumed.ask(2), x)
        kept.tell(x, _objective(x))
        resumed.tell(x, _objective(x))
        del resumed
        resumed = Optimizer.open(path)
    assert resumed.tr_length == 1.0


@pytest.mark.parametrize(
    "region",
    [TurboTRConfig(), MorboTRConfig(MultiObjectiveConfig(num_metrics=2))],
    ids=["turbo", "morbo"],
)
def test_lhd_only_rejects_a_trust_region(region):
    with pytest.raises(ValueError, match="requires NoTRConfig"):
        lhd_only_config(num_init=4, trust_region=region)


def test_lhd_only_accepts_an_explicit_absence_of_a_region(tmp_path):
    config = lhd_only_config(num_init=4, trust_region=NoTRConfig())
    optimizer = create_optimizer(
        bounds=BOUNDS, config=config, seed=1, work_dir=tmp_path / "run"
    )
    x = optimizer.ask(4)
    optimizer.tell(x, _objective(x))
    assert optimizer.tr_length == 1.0
