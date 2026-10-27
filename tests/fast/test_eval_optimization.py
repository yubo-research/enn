"""Optimization metrics must score actual proposals and use an equal evaluation budget."""

import numpy as np
import pytest

from enn.evals import optimization as mod


@pytest.mark.parametrize("problem", ("sphere", "ackley"))
def test_known_optimum_and_nonnegative_regret(problem):
    assert mod.objective(np.zeros((1, mod.NUM_DIM)), problem)[0, 0] == pytest.approx(0)
    assert np.all(mod.objective(np.ones((2, mod.NUM_DIM)), problem) < 0)
    with pytest.raises(ValueError, match="unknown problem"):
        mod.objective(np.zeros((1, mod.NUM_DIM)), "missing")


def test_closed_loop_scores_and_tells_every_proposal(monkeypatch):
    told = []

    class Optimizer:
        def ask(self, num_arms):
            assert num_arms == 1
            return np.zeros((1, mod.NUM_DIM))

        def tell(self, x, y):
            told.append((x.copy(), y.copy()))

    monkeypatch.setattr(mod, "build_optimizer", lambda seed: Optimizer())
    monkeypatch.setattr(mod, "BUDGET", 12)
    monkeypatch.setattr(mod, "CHECKPOINTS", (10, 12))
    rows = mod.run_seed("sphere", 4)
    assert sum(len(x) for x, _ in told) == 12
    assert len(told) == 3
    for x, y in told:
        np.testing.assert_array_equal(y, mod.objective(x, "sphere"))
    enn = [r for r in rows if r["method"] == "enn"]
    random = [r for r in rows if r["method"] == "random"]
    assert enn[0]["regret"] == random[0]["regret"]
    assert enn[-1]["regret"] == 0
    assert enn[0]["regret"] > 0
    assert random[-1]["regret"] <= random[0]["regret"]
    assert all(r["wall_s"] > 0 for r in rows)


def test_real_optimizer_repeats_quality_with_fixed_seed(monkeypatch):
    monkeypatch.setattr(mod, "NUM_INIT", 4)
    monkeypatch.setattr(mod, "BUDGET", 8)
    monkeypatch.setattr(mod, "CHECKPOINTS", (4, 8))
    first = mod.run_seed("sphere", 21)
    second = mod.run_seed("sphere", 21)
    assert [(r["method"], r["n"], r["regret"]) for r in first] == [
        (r["method"], r["n"], r["regret"]) for r in second
    ]
    assert all(r["regret"] >= 0 for r in first)
