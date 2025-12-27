from __future__ import annotations

import numpy as np
import pytest

import conftest

from enn.turbo.turbo_mode import TurboMode
from enn.turbo.turbo_utils import to_unit


@pytest.mark.parametrize(
    "mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ONE, TurboMode.TURBO_ENN]
)
def test_turbo_behavior_independent_of_affine_x(mode: TurboMode) -> None:
    from enn import Turbo

    bounds1 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    bounds2 = np.array([[2.0, 4.0], [-3.0, 1.0]], dtype=float)
    num_arms, num_steps = 4, 8
    rng1, rng2 = np.random.default_rng(0), np.random.default_rng(0)
    opt1 = Turbo(bounds=bounds1, mode=mode, rng=rng1)
    opt2 = Turbo(bounds=bounds2, mode=mode, rng=rng2)
    for _ in range(num_steps):
        x1, x2 = opt1.ask(num_arms=num_arms), opt2.ask(num_arms=num_arms)
        u1, u2 = to_unit(x1, bounds1), to_unit(x2, bounds2)
        assert np.allclose(u1, u2)
        y1 = conftest.sphere_objective(2.0 * u1 - 1.0)
        y2 = conftest.sphere_objective(2.0 * u2 - 1.0)
        assert np.allclose(y1, y2)
        opt1.tell(x1, y1)
        opt2.tell(x2, y2)


@pytest.mark.parametrize("mode", [TurboMode.TURBO_ZERO, TurboMode.TURBO_ENN])
def test_turbo_behavior_independent_of_affine_y(mode: TurboMode) -> None:
    from enn import Turbo

    bounds = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float)
    num_arms, num_steps = 4, 8

    def run_with_transform(scale: float, shift: float) -> np.ndarray:
        rng = np.random.default_rng(0)
        opt = Turbo(bounds=bounds, mode=mode, rng=rng)
        unit_trajectory = []
        for _ in range(num_steps):
            x = opt.ask(num_arms=num_arms)
            u = x.copy()
            base_y = conftest.sphere_objective(2.0 * u - 1.0)
            opt.tell(x, scale * base_y + shift)
            unit_trajectory.append(u)
        return np.stack(unit_trajectory, axis=0)

    traj_base = run_with_transform(scale=1.0, shift=0.0)
    traj_affine = run_with_transform(scale=2.0, shift=0.5)
    assert np.allclose(traj_base, traj_affine)
