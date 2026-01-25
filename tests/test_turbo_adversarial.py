from __future__ import annotations
import conftest
import numpy as np
import pytest
from enn import create_optimizer
from enn.turbo.components import NoSurrogate, SurrogateResult
from enn.turbo.config.enums import CandidateRV
from enn.turbo.optimizer import Optimizer
from enn.turbo.optimizer_config import turbo_enn_config, turbo_zero_config


def test_turbo_enn_affine_invariance_under_dynamic_y_range() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms, num_steps = 4, 20
    config = turbo_enn_config()

    def run(global_scale: float, global_shift: float) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        unit_trajectory: list[np.ndarray] = []
        tr_lengths: list[float] = []
        for t in range(num_steps):
            x = opt.ask(num_arms=num_arms)
            u = x.copy()
            base_y = conftest.sphere_objective(2.0 * u - 1.0)
            a_t = 10.0 ** float(np.sin(t + 1.0)) + 1e-6
            b_t = 1e6 * float(np.cos(t + 1.0))
            y = global_scale * (a_t * base_y + b_t) + global_shift
            opt.tell(x, y)
            unit_trajectory.append(u)
            tr_lengths.append(opt.tr_length)
        return np.stack(unit_trajectory, axis=0), np.asarray(tr_lengths, dtype=float)

    traj_a, lengths_a = run(global_scale=1.0, global_shift=0.0)
    traj_b, lengths_b = run(global_scale=3.7, global_shift=-0.2)
    assert np.allclose(traj_a, traj_b)
    assert np.allclose(lengths_a, lengths_b)


def test_trailing_obs_preserves_unique_best_and_is_deterministic_under_ties() -> None:
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    config = turbo_zero_config(
        trailing_obs=5, num_init=2, candidate_rv=CandidateRV.UNIFORM
    )
    num_steps = 30

    def run(y_fn) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(0)
        opt = create_optimizer(bounds=bounds, config=config, rng=rng)
        snapshots: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(num_steps):
            x = opt.ask(num_arms=1)
            y = np.asarray([float(y_fn(t))], dtype=float)
            opt.tell(x, y)
            snapshots.append(
                (
                    opt._x_obs.view().copy(),
                    opt._y_obs.view().copy(),
                )
            )
        return snapshots

    snaps_unique_best = run(lambda t: 1_000.0 if t == 0 else -float(t))
    for _, y_obs in snaps_unique_best[10:]:
        assert float(np.max(y_obs)) == 1_000.0
    snaps_ties_1 = run(lambda t: 0.0)
    snaps_ties_2 = run(lambda t: 0.0)
    assert len(snaps_ties_1) == len(snaps_ties_2)
    for (x1, y1), (x2, y2) in zip(snaps_ties_1, snaps_ties_2, strict=True):
        assert np.array_equal(y1, y2)
        assert np.allclose(x1, x2)


class _NoSurrogateWithLengthscales(NoSurrogate):
    def __init__(self, lengthscales: np.ndarray) -> None:
        super().__init__()
        self._lengthscales = np.asarray(lengthscales, dtype=float)

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: object | None = None,
    ) -> SurrogateResult:
        super().fit(
            x_obs,
            y_obs,
            y_var,
            num_steps=num_steps,
            rng=rng,
        )
        return SurrogateResult(model=None, lengthscales=self._lengthscales)


class _FirstNAcqOptimizer:
    def select(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        surrogate: object,
        rng: object,
        *,
        tr_state: object | None = None,
    ) -> np.ndarray:
        return np.asarray(x_cand, dtype=float)[: int(num_arms)]


def test_candidate_generation_with_extreme_lengthscales_stays_in_bounds() -> None:
    bounds = np.array([[0.0, 1.0]] * 5, dtype=float)
    rng = np.random.default_rng(0)
    config = turbo_zero_config(
        num_init=1,
        num_candidates=64,
        candidate_rv=CandidateRV.UNIFORM,
    )
    surrogate = _NoSurrogateWithLengthscales(
        lengthscales=np.array([1e-6, 1e6, 1e-3, 1e3, 1.0], dtype=float)
    )
    opt = Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=surrogate,
        acquisition_optimizer=_FirstNAcqOptimizer(),
    )
    x0 = opt.ask(num_arms=1)
    opt.tell(x0, np.array([0.0], dtype=float))
    x_obs = opt._x_obs.view()
    y_obs = opt._y_obs.view()
    x_center = opt._find_x_center(x_obs, y_obs)
    assert x_center is not None
    x = opt.ask(num_arms=4)
    assert x.shape == (4, bounds.shape[0])
    assert np.all(np.isfinite(x))
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(np.any(np.abs(x - x_center.reshape(1, -1)) > 0.0, axis=1))


def test_candidate_generation_raises_on_nonfinite_lengthscales() -> None:
    bounds = np.array([[0.0, 1.0]] * 3, dtype=float)
    rng = np.random.default_rng(0)
    config = turbo_zero_config(
        num_init=1,
        num_candidates=16,
        candidate_rv=CandidateRV.UNIFORM,
    )
    surrogate = _NoSurrogateWithLengthscales(
        lengthscales=np.array([np.nan, 1.0, 1.0], dtype=float)
    )
    opt = Optimizer(
        bounds=bounds,
        config=config,
        rng=rng,
        surrogate=surrogate,
        acquisition_optimizer=_FirstNAcqOptimizer(),
    )
    x0 = opt.ask(num_arms=1)
    opt.tell(x0, np.array([0.0], dtype=float))
    with pytest.raises(ValueError, match="lengthscales"):
        _ = opt.ask(num_arms=2)
