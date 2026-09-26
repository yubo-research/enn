from __future__ import annotations

from ops.qa import run_y_bounds


def evaluate() -> None:
    """Compare unbounded vs (0, inf) y-bounds ENN on a positive 1-d DGP."""
    run_y_bounds()
