import numpy as np
import pytest

from uhd.simple_adapter import SimpleAdapter


def test_simple_adapter_ask_returns_middle():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng)
    assert adapter.ask() == 0.1  # Middle of 3 elements


def test_simple_adapter_tell_success_can_increase():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng, p_success=1.0)
    adapter.tell(success=True)
    assert adapter.ask() == 1.0


def test_simple_adapter_tell_failure_can_decrease():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng, p_failure=1.0)
    adapter.tell(success=False)
    assert adapter.ask() == 0.01


def test_simple_adapter_stays_at_max():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng, p_success=1.0)
    adapter.tell(success=True)  # Move to 1.0
    adapter.tell(success=True)  # Should stay at 1.0
    assert adapter.ask() == 1.0


def test_simple_adapter_stays_at_min():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng, p_failure=1.0)
    adapter.tell(success=False)  # Move to 0.01
    adapter.tell(success=False)  # Should stay at 0.01
    assert adapter.ask() == 0.01


def test_simple_adapter_probability_respected():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng, p_success=0.0)
    adapter.tell(success=True)
    assert adapter.ask() == 0.1  # Should not move with p_success=0


def test_simple_adapter_sorts_step_sizes():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[1.0, 0.01, 0.1], rng=rng)
    assert adapter._step_sizes == [0.01, 0.1, 1.0]


def test_simple_adapter_invalid_empty_step_sizes():
    rng = np.random.default_rng(seed=42)
    with pytest.raises(ValueError, match="step_sizes must be non-empty"):
        SimpleAdapter(step_sizes=[], rng=rng)


def test_simple_adapter_invalid_p_success():
    rng = np.random.default_rng(seed=42)
    with pytest.raises(ValueError, match="p_success must be in"):
        SimpleAdapter(step_sizes=[0.1], rng=rng, p_success=-0.1)


def test_simple_adapter_invalid_p_failure():
    rng = np.random.default_rng(seed=42)
    with pytest.raises(ValueError, match="p_failure must be in"):
        SimpleAdapter(step_sizes=[0.1], rng=rng, p_failure=1.5)


def test_simple_adapter_step_size_property():
    rng = np.random.default_rng(seed=42)
    adapter = SimpleAdapter(step_sizes=[0.01, 0.1, 1.0], rng=rng)
    assert adapter.step_size == 0.1
