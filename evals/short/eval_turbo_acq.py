from __future__ import annotations

from ops.qa import run_turbo_acq


def evaluate() -> None:
    """Compare TuRBO acquisition types on Ackley (noiseless primary)."""
    run_turbo_acq()
