"""A tell() that raises must leave the optimizer as if the call never happened."""

import numpy as np
import pytest

from enn import (
    ENNFitConfig,
    ENNSurrogateConfig,
    create_optimizer,
    lhd_only_config,
    turbo_enn_config,
)

BOUNDS = np.array([[0.0, 1.0]] * 3)


def _objective(x):
    return -np.linalg.norm(x - 0.3, axis=1, keepdims=True)


def _optimizer(path, num_init, y_bounds=None):
    enn = ENNSurrogateConfig(
        k=3, fit=ENNFitConfig(num_fit_samples=5), y_bounds=y_bounds
    )
    config = turbo_enn_config(enn=enn, num_init=num_init)
    return create_optimizer(bounds=BOUNDS, config=config, seed=1, work_dir=path)


def _with_nan(y):
    y = y.copy()
    y[0] = np.nan
    return y


def _nan_y(x, _yvar):
    return _with_nan(_objective(x)), None


def _inf_y(x, _yvar):
    y = _objective(x)
    y[0] = np.inf
    return y, None


def _extra_metric(x, _yvar):
    return np.hstack([_objective(x)] * 2), None


def _negative_yvar(x, _yvar):
    return _objective(x), -np.ones((len(x), 1))


def _wrong_yvar_shape(x, _yvar):
    return _objective(x), np.ones((len(x), 2))


def _omitted_yvar(x, yvar):
    assert yvar is not None
    return _objective(x), None


def _advance(optimizer, num_steps, num_arms=2, yvar=None):
    for _ in range(num_steps):
        x = optimizer.ask(num_arms)
        optimizer.tell(
            x, _objective(x), None if yvar is None else yvar * np.ones((num_arms, 1))
        )


def test_rejected_tell_does_not_consume_initial_design(tmp_path):
    optimizer = _optimizer(tmp_path / "run", num_init=8)
    x = optimizer.ask(4)
    for _ in range(3):
        with pytest.raises(ValueError):
            optimizer.tell(x, _with_nan(_objective(x)))
    assert optimizer.init_progress == (0, 8)
    optimizer.tell(x, _objective(x))
    assert optimizer.init_progress == (4, 8)


def test_rejected_first_tell_does_not_fix_the_yvar_mode(tmp_path):
    optimizer = _optimizer(tmp_path / "run", num_init=4)
    x = optimizer.ask(4)
    with pytest.raises(ValueError):
        optimizer.tell(x, _objective(x), -np.ones((4, 1)))
    optimizer.tell(x, _objective(x))
    assert len(optimizer.x_obs) == 4


def test_accepted_tell_still_fixes_the_yvar_mode(tmp_path):
    optimizer = _optimizer(tmp_path / "run", num_init=4)
    x = optimizer.ask(2)
    optimizer.tell(x, _objective(x), np.ones((2, 1)))
    with pytest.raises(ValueError, match="train_yvar must be provided"):
        optimizer.tell(x, _objective(x))


@pytest.mark.parametrize("reject", [_nan_y, _inf_y, _extra_metric])
def test_rejected_tell_does_not_change_the_search(tmp_path, reject):
    faulted = _optimizer(tmp_path / "faulted", num_init=4)
    clean = _optimizer(tmp_path / "clean", num_init=4)
    _advance(faulted, 6)
    _advance(clean, 6)
    x = faulted.ask(2)
    assert np.array_equal(x, clean.ask(2))
    y, yvar = reject(x, None)
    with pytest.raises(ValueError):
        faulted.tell(x, y, yvar)
    for optimizer in (faulted, clean):
        optimizer.tell(x, _objective(x))
    assert np.array_equal(faulted.ask(2), clean.ask(2))
    assert np.array_equal(faulted.best_y, clean.best_y)


@pytest.mark.parametrize("reject", [_negative_yvar, _wrong_yvar_shape, _omitted_yvar])
def test_rejected_noisy_tell_does_not_change_the_search(tmp_path, reject):
    faulted = _optimizer(tmp_path / "faulted", num_init=4)
    clean = _optimizer(tmp_path / "clean", num_init=4)
    yvar = 0.01
    _advance(faulted, 6, yvar=yvar)
    _advance(clean, 6, yvar=yvar)
    x = faulted.ask(2)
    assert np.array_equal(x, clean.ask(2))
    y, bad_yvar = reject(x, yvar * np.ones((2, 1)))
    with pytest.raises(ValueError):
        faulted.tell(x, y, bad_yvar)
    for optimizer in (faulted, clean):
        optimizer.tell(x, _objective(x), yvar * np.ones((2, 1)))
    assert np.array_equal(faulted.ask(2), clean.ask(2))


@pytest.mark.parametrize("num_steps", [0, 6])
def test_y_outside_y_bounds_does_not_change_the_search(tmp_path, num_steps):
    y_bounds = np.array([[-10.0, 10.0]])
    faulted = _optimizer(tmp_path / "faulted", num_init=4, y_bounds=y_bounds)
    clean = _optimizer(tmp_path / "clean", num_init=4, y_bounds=y_bounds)
    _advance(faulted, num_steps)
    _advance(clean, num_steps)
    x = faulted.ask(2)
    assert np.array_equal(x, clean.ask(2))
    with pytest.raises(ValueError, match="strictly"):
        faulted.tell(x, _objective(x) + 100.0)
    for optimizer in (faulted, clean):
        optimizer.tell(x, _objective(x))
    assert faulted.init_progress == clean.init_progress
    assert np.array_equal(faulted.ask(2), clean.ask(2))


def test_first_tell_with_two_metrics_is_rejected_by_a_scalar_trust_region(tmp_path):
    faulted = _optimizer(tmp_path / "faulted", num_init=4)
    clean = _optimizer(tmp_path / "clean", num_init=4)
    x = faulted.ask(2)
    assert np.array_equal(x, clean.ask(2))
    with pytest.raises(ValueError, match="trust region expects 1"):
        faulted.tell(x, np.hstack([_objective(x)] * 2))
    assert faulted.init_progress == (0, 4)
    for optimizer in (faulted, clean):
        optimizer.tell(x, _objective(x))
    assert np.array_equal(faulted.ask(2), clean.ask(2))


def test_metric_width_stays_fixed_across_a_trust_region_restart(tmp_path):
    optimizer = _optimizer(tmp_path / "run", num_init=2)
    restarts = 0
    for step in range(40):
        rows_before_ask = len(optimizer.x_obs)
        x = optimizer.ask(2)
        restarts += len(optimizer.x_obs) < rows_before_ask
        worsening = -float(step) * np.ones((2, 1))
        with pytest.raises(ValueError, match="metric columns"):
            optimizer.tell(x, np.hstack([worsening] * 2))
        optimizer.tell(x, worsening)
    assert restarts > 0


def test_init_only_run_keeps_every_metric_column(tmp_path):
    optimizer = create_optimizer(
        bounds=BOUNDS, config=lhd_only_config(), seed=1, work_dir=tmp_path / "run"
    )
    x = optimizer.ask(2)
    optimizer.tell(x, np.hstack([_objective(x)] * 2))
    assert optimizer.y_obs.shape == (2, 2)
