from __future__ import annotations

import numpy as np
import pytest

from enn import (
    AcqType,
    CandidateGenConfig,
    CandidateRV,
    ENNFitConfig,
    ENNSurrogateConfig,
    MorboTRConfig,
    MultiObjectiveConfig,
    Optimizer,
    TRLengthConfig,
    TurboTRConfig,
    create_optimizer,
    lhd_only_config,
    turbo_enn_config,
    turbo_zero_config,
)

BOUNDS = np.array([[0.0, 1.0], [-1.0, 2.0], [0.0, 3.0]])
SEARCHED = ENNFitConfig(num_fit_samples=8, num_fit_candidates=4)
CANDIDATES = CandidateGenConfig(max_candidates=64)


def _morbo():
    return MorboTRConfig(MultiObjectiveConfig(num_metrics=2))


def _enn(surrogate, acq=AcqType.PARETO, **kwargs):
    return turbo_enn_config(
        enn=surrogate, num_init=4, acq_type=acq, candidates=CANDIDATES, **kwargs
    )


CONFIGS = {
    "frozen": lambda: _enn(ENNSurrogateConfig(k=3)),
    "pareto": lambda: _enn(ENNSurrogateConfig(k=3, fit=SEARCHED)),
    "ucb": lambda: _enn(ENNSurrogateConfig(k=3, fit=SEARCHED), AcqType.UCB),
    "thompson": lambda: _enn(ENNSurrogateConfig(k=3, fit=SEARCHED), AcqType.THOMPSON),
    "calibrated": lambda: _enn(
        ENNSurrogateConfig(
            k=3,
            fit=ENNFitConfig(
                num_fit_samples=8, num_fit_candidates=4, affine_calibrate=True
            ),
        )
    ),
    "weighting": lambda: _enn(ENNSurrogateConfig(k=3, weighting=True)),
    "morbo": lambda: _enn(ENNSurrogateConfig(k=3), trust_region=_morbo()),
    "sobol": lambda: turbo_enn_config(
        enn=ENNSurrogateConfig(k=3),
        num_init=4,
        candidates=CandidateGenConfig(
            candidate_rv=CandidateRV.SOBOL, max_candidates=64
        ),
    ),
    "zero": lambda: turbo_zero_config(num_init=4, candidates=CANDIDATES),
    "lhd": lambda: lhd_only_config(num_init=4),
}
SHORT_REGION = TurboTRConfig(TRLengthConfig(length_init=0.1, length_min=0.05))
CONFIGS["restarting_enn"] = lambda: _enn(
    ENNSurrogateConfig(k=3), trust_region=SHORT_REGION
)
CONFIGS["restarting_zero"] = lambda: turbo_zero_config(
    num_init=4, candidates=CANDIDATES, trust_region=SHORT_REGION
)
MULTI = {"morbo"}


def _objective(kind, x, noise):
    y = -np.sum((x - 0.4) ** 2, axis=1, keepdims=True) + 0.01 * noise[:, :1]
    if kind in MULTI:
        return np.hstack([y, 0.5 * y + 0.01 * noise[:, 1:2]])
    return y


def _observe(optimizer):
    return {
        "x_obs": optimizer.x_obs,
        "y_obs": optimizer.y_obs,
        "incumbent_x": optimizer.incumbent_x,
        "tr_length": optimizer.tr_length,
        "tr_obs_count": optimizer.tr_obs_count,
        "init_progress": optimizer.init_progress,
    }


def _assert_same(a, b):
    for key in a:
        if a[key] is None or isinstance(a[key], (float, int, tuple)):
            assert a[key] == b[key], key
        else:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def _create(kind, work_dir, seed=0):
    return create_optimizer(
        bounds=BOUNDS, config=CONFIGS[kind](), seed=seed, work_dir=work_dir
    )


def _run_pair(
    tmp_path, kind, rounds, *, between_ask_and_tell=False, worsening=False, arms=2
):
    kept = _create(kind, tmp_path / "kept")
    resumed = _create(kind, tmp_path / "resumed")
    noise_rng = np.random.default_rng(5)
    restarts = 0
    for round_index in range(rounds):
        before = kept.tr_obs_count
        proposed = kept.ask(arms)
        np.testing.assert_array_equal(resumed.ask(arms), proposed)
        restarts += int(kept.tr_obs_count < before)
        if between_ask_and_tell:
            del resumed
            resumed = Optimizer.open(tmp_path / "resumed")
        noise = noise_rng.standard_normal((proposed.shape[0], 2))
        y = _objective(kind, proposed, noise)
        if worsening:
            y = y - float(round_index)
        told = kept.tell(proposed, y)
        np.testing.assert_array_equal(resumed.tell(proposed, y), told)
        del resumed
        resumed = Optimizer.open(tmp_path / "resumed")
        _assert_same(_observe(kept), _observe(resumed))
    return restarts


@pytest.mark.parametrize("kind", list(CONFIGS))
def test_reopening_every_round_matches_a_single_optimizer(tmp_path, kind):
    _run_pair(tmp_path, kind, rounds=10)


@pytest.mark.parametrize("kind", ["morbo", "weighting", "sobol"])
def test_long_runs_with_large_batches_stay_exact(tmp_path, kind):
    _run_pair(tmp_path, kind, rounds=20, arms=8)


@pytest.mark.parametrize("kind", ["pareto", "zero"])
def test_reopening_between_ask_and_tell_matches_a_single_optimizer(tmp_path, kind):
    _run_pair(tmp_path, kind, rounds=8, between_ask_and_tell=True)


@pytest.mark.parametrize("kind", ["restarting_enn", "restarting_zero"])
def test_resume_crosses_the_init_transition_and_a_restart(tmp_path, kind):
    assert _run_pair(tmp_path, kind, rounds=30, worsening=True) > 0


def test_create_rejects_a_used_work_dir(tmp_path):
    _create("zero", tmp_path / "w")
    with pytest.raises(ValueError, match=r"Optimizer\.open"):
        _create("zero", tmp_path / "w")


def test_open_rejects_an_empty_work_dir(tmp_path):
    with pytest.raises(ValueError, match="create_optimizer"):
        Optimizer.open(tmp_path)


def test_open_restores_bounds_and_config(tmp_path):
    optimizer = _create("frozen", tmp_path / "w")
    x = optimizer.ask(2)
    optimizer.tell(x, np.ones((2, 1)))
    del optimizer
    reopened = Optimizer.open(tmp_path / "w")
    np.testing.assert_array_equal(reopened.bounds, BOUNDS)


@pytest.mark.parametrize("kind", ["frozen", "pareto", "zero"])
def test_extra_rows_on_disk_are_replayed(tmp_path, kind):
    kept = _create(kind, tmp_path / "kept")
    crashed = _create(kind, tmp_path / "crashed")
    state = tmp_path / "crashed" / "optimizer_state.json"
    noise_rng = np.random.default_rng(5)
    for round_index in range(6):
        arms = kept.ask(2)
        np.testing.assert_array_equal(crashed.ask(2), arms)
        y = _objective(kind, arms, noise_rng.standard_normal((2, 2)))
        before_tell = state.read_bytes()
        kept.tell(arms, y)
        crashed.tell(arms, y)
        if round_index == 4:
            del crashed
            state.write_bytes(before_tell)
            crashed = Optimizer.open(tmp_path / "crashed")
    _assert_same(_observe(kept), _observe(crashed))
    np.testing.assert_array_equal(kept.ask(2), crashed.ask(2))


def test_fewer_rows_than_the_snapshot_is_an_error(tmp_path):
    optimizer = _create("zero", tmp_path / "w")
    x = optimizer.ask(2)
    optimizer.tell(x, np.ones((2, 1)))
    del optimizer
    (tmp_path / "w" / "optimizer_observations.bin").unlink()
    with pytest.raises(ValueError, match="holds 0 rows"):
        Optimizer.open(tmp_path / "w")


def test_torn_state_file_is_a_clear_error(tmp_path):
    optimizer = _create("zero", tmp_path / "w")
    optimizer.ask(1)
    del optimizer
    state = tmp_path / "w" / "optimizer_state.json"
    state.write_text(state.read_text()[:40])
    with pytest.raises(ValueError, match="optimizer_state"):
        Optimizer.open(tmp_path / "w")


def test_a_leftover_temporary_state_file_is_ignored(tmp_path):
    optimizer = _create("zero", tmp_path / "w")
    arms = optimizer.ask(2)
    del optimizer
    (tmp_path / "w" / "optimizer_state.json.tmp").write_text("{")
    reopened = Optimizer.open(tmp_path / "w")
    reopened.tell(arms, np.ones((2, 1)))
    assert reopened.tr_obs_count == 2


@pytest.mark.parametrize("kind", ["restarting_enn", "restarting_zero"])
def test_crash_inside_a_restart_matches_an_uninterrupted_run(tmp_path, kind):
    kept = _create(kind, tmp_path / "kept")
    crashed = _create(kind, tmp_path / "crashed")
    state = tmp_path / "crashed" / "optimizer_state.json"
    noise_rng = np.random.default_rng(5)
    crossed_restart = False
    for round_index in range(30):
        before = kept.tr_obs_count
        arms = kept.ask(2)
        if kept.tr_obs_count < before:
            crossed_restart = True
            del crashed
            state.write_text(
                state.read_text().replace(
                    '"restart_pending":false', '"restart_pending":true'
                )
            )
            crashed = Optimizer.open(tmp_path / "crashed")
            assert crashed.tr_obs_count == 0
        np.testing.assert_array_equal(crashed.ask(2), arms)
        noise = noise_rng.standard_normal((arms.shape[0], 2))
        y = _objective(kind, arms, noise) - float(round_index)
        kept.tell(arms, y)
        crashed.tell(arms, y)
        _assert_same(_observe(kept), _observe(crashed))
    assert crossed_restart


def test_restart_pending_snapshot_is_finished_on_open(tmp_path):
    optimizer = _create("frozen", tmp_path / "w")
    x = optimizer.ask(2)
    optimizer.tell(x, np.ones((2, 1)))
    del optimizer
    state = tmp_path / "w" / "optimizer_state.json"
    text = state.read_text().replace(
        '"restart_pending":false', '"restart_pending":true'
    )
    state.write_text(text)
    reopened = Optimizer.open(tmp_path / "w")
    assert reopened.tr_obs_count == 0
    assert '"restart_pending":false' in state.read_text()
    assert np.isfinite(reopened.ask(2)).all()
