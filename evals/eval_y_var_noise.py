from __future__ import annotations

from ops.qa import run_y_var_noise


def evaluate() -> None:
    """Compare matched vs mismatched train_yvar under observation_noise."""
    run_y_var_noise()
