import numpy as np
import pytest

from uhd.thompson_sampler import ThompsonSampler


def test_thompson_sampler_cold_start_returns_uninitialized_arms():
    arms = ["a", "b", "c"]

    returned = set()
    for i in range(100):
        ts = ThompsonSampler(arms, np.random.default_rng(i))
        arm = ts.ask()
        returned.add(arm)
        if returned == {"a", "b", "c"}:
            break

    assert returned == {"a", "b", "c"}


def test_thompson_sampler_cold_start_cycles_through_uninitialized():
    arms = ["a", "b", "c"]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    arm1 = ts.ask()
    ts.tell(success=True)

    arm2 = ts.ask()
    ts.tell(success=True)

    assert arm1 != arm2

    arm3 = ts.ask()
    ts.tell(success=True)

    assert {arm1, arm2, arm3} == {"a", "b", "c"}


def test_thompson_sampler_ask_uses_thompson_embedding():
    good = ["good"]
    bad = ["bad"]
    arms = [good, bad]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng, min_observations=1)

    # Give good arm many successes, bad arm many failures
    for _ in range(10):
        arm = ts.ask()
        ts.tell(success=(arm is good))

    good_count = sum(1 for _ in range(100) if ts.ask() is good)
    assert good_count > 80


def test_thompson_sampler_tell_updates_mvue():
    arm = ["a"]
    arms = [arm]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    ts.ask()
    ts.tell(success=True)
    assert ts._mvues[0].n == 1
    assert ts._mvues[0].mean == 1.0

    ts.ask()
    ts.tell(success=False)
    assert ts._mvues[0].n == 2
    assert ts._mvues[0].mean == 0.5


def test_thompson_sampler_empty_arms_raises():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="arms must be non-empty"):
        ThompsonSampler([], rng)


def test_thompson_sampler_tell_before_ask_raises():
    arms = ["a", "b"]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    with pytest.raises(RuntimeError, match="tell\\(\\) called before ask\\(\\)"):
        ts.tell(success=True)


def test_thompson_sampler_deterministic_with_same_seed():
    arms = ["a", "b", "c"]

    ts1 = ThompsonSampler(arms, np.random.default_rng(42))
    ts2 = ThompsonSampler(arms, np.random.default_rng(42))

    # Initialize both samplers the same way
    for ts in [ts1, ts2]:
        for _ in range(3):
            ts.ask()
            ts.tell(success=True)

    results1 = [ts1.ask() for _ in range(20)]
    results2 = [ts2.ask() for _ in range(20)]

    assert results1 == results2
