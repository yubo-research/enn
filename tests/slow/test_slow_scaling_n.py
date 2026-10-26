"""Process-isolation integration coverage, including fresh interpreter startup."""

import pytest

from evals import scaling_n as mod


@pytest.mark.slow
def test_run_seed_isolated_matches_grid() -> None:
    cfg = mod.ScalingConfig(
        n_grid=(20, 40),
        num_test=5,
        batch=20,
        k=4,
        num_fit_candidates=2,
        num_fit_samples=2,
    )
    rows = mod.run_seed_isolated(cfg)
    assert [r["n"] for r in rows] == [20.0, 40.0]
    assert all(r["disk_mib"] > 0 for r in rows)
