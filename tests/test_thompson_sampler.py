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
    ts.tell(arm1, y=1.0, y_var=1.0)

    arm2 = ts.ask()
    ts.tell(arm2, y=1.0, y_var=1.0)

    assert arm1 != arm2

    arm3 = ts.ask()
    ts.tell(arm3, y=1.0, y_var=1.0)

    assert {arm1, arm2, arm3} == {"a", "b", "c"}


def test_thompson_sampler_ask_uses_thompson_sampling():
    good = ["good"]
    bad = ["bad"]
    arms = [good, bad]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    for _ in range(3):
        ts.tell(good, y=10.0, y_var=0.01)
        ts.tell(bad, y=0.0, y_var=0.01)

    good_count = sum(1 for _ in range(100) if ts.ask() is good)
    assert good_count > 90


def test_thompson_sampler_tell_updates_mvue():
    arm = ["a"]
    arms = [arm]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    ts.tell(arm, y=5.0, y_var=1.0)
    assert ts._mvues[0].n == 1
    assert ts._mvues[0].mean == 5.0

    ts.tell(arm, y=7.0, y_var=1.0)
    assert ts._mvues[0].n == 2
    assert ts._mvues[0].mean == 6.0


def test_thompson_sampler_uses_identity_not_equality():
    class Arm:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Arm):
                return self.name == other.name
            return False

    arm1 = Arm("x")
    arm2 = Arm("x")
    assert arm1 == arm2
    assert arm1 is not arm2

    rng = np.random.default_rng(42)
    ts = ThompsonSampler([arm1, arm2], rng)

    ts.tell(arm1, y=10.0, y_var=1.0)
    assert ts._mvues[0].n == 1
    assert ts._mvues[1].n == 0

    ts.tell(arm2, y=5.0, y_var=1.0)
    assert ts._mvues[0].n == 1
    assert ts._mvues[1].n == 1


def test_thompson_sampler_empty_arms_raises():
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError, match="arms must be non-empty"):
        ThompsonSampler([], rng)


def test_thompson_sampler_unknown_arm_raises():
    arms = ["a", "b"]
    rng = np.random.default_rng(42)
    ts = ThompsonSampler(arms, rng)

    with pytest.raises(ValueError, match="arm not found"):
        ts.tell("c", y=1.0, y_var=1.0)


def test_thompson_sampler_deterministic_with_same_seed():
    arms = ["a", "b", "c"]

    ts1 = ThompsonSampler(arms, np.random.default_rng(42))
    ts2 = ThompsonSampler(arms, np.random.default_rng(42))

    for ts in [ts1, ts2]:
        ts.tell("a", y=5.0, y_var=1.0)
        ts.tell("b", y=5.0, y_var=1.0)
        ts.tell("c", y=5.0, y_var=1.0)

    results1 = [ts1.ask() for _ in range(20)]
    results2 = [ts2.ask() for _ in range(20)]

    assert results1 == results2
