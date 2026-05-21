from __future__ import annotations

SPHERE_Y_RANGE = 1.0
SPHERE_BEST_Y_TOL = 0.02 * SPHERE_Y_RANGE
TR_LENGTH_MEAN_DIFF_TOL = 0.15


def assert_sphere_best_y_parity(rust_best: float, py_best: float) -> None:
    assert abs(rust_best - py_best) <= SPHERE_BEST_Y_TOL
