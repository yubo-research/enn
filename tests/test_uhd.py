import numpy as np
import torch
from torch import nn

from uhd.simple_adapter import SimpleAdapter
from uhd.simple_perturbator import SimplePerturbator
from uhd.uhd import UHD


def _make_simple_model() -> nn.Module:
    return nn.Linear(10, 5)


def _get_params_flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in module.parameters()])


def test_uhd_ask_returns_seed_and_perturbs_module():
    rng = np.random.default_rng(42)
    model = _make_simple_model()
    params_before = _get_params_flat(model).clone()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    seed = uhd.ask(model)

    assert isinstance(seed, int)
    params_after = _get_params_flat(model)
    assert not torch.allclose(params_before, params_after)


def test_uhd_tell_accepts_improvement():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    seed = uhd.ask(model)
    params_after_ask = _get_params_flat(model).clone()

    accepted = uhd.tell(model, seed, y=1.0, y_var=0.1)

    assert accepted is True
    assert uhd.incumbent_y == 1.0
    assert uhd.accepts == 1
    assert uhd.races == 1
    params_after_tell = _get_params_flat(model)
    assert torch.allclose(params_after_ask, params_after_tell)


def test_uhd_tell_rejects_and_unperturbs():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    seed1 = uhd.ask(model)
    uhd.tell(model, seed1, y=1.0, y_var=0.1)

    params_after_first = _get_params_flat(model).clone()

    seed2 = uhd.ask(model)
    accepted = uhd.tell(model, seed2, y=0.5, y_var=0.1)

    assert accepted is False
    assert uhd.incumbent_y == 1.0
    assert uhd.accepts == 1
    assert uhd.races == 2
    params_after_reject = _get_params_flat(model)
    assert torch.allclose(params_after_first, params_after_reject)


def test_uhd_tell_validates_y():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    seed = uhd.ask(model)

    try:
        uhd.tell(model, seed, y=float("nan"), y_var=0.1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "y must be finite" in str(e)


def test_uhd_tell_validates_y_var():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    seed = uhd.ask(model)

    try:
        uhd.tell(model, seed, y=1.0, y_var=-0.1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "y_var must be finite and positive" in str(e)


def test_uhd_tell_validates_seed_mismatch():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    uhd.ask(model)

    try:
        uhd.tell(model, seed=99999, y=1.0, y_var=0.1)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "seed mismatch" in str(e)


def test_uhd_step_size_property():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.05, 0.1, 0.2], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    assert uhd.step_size is None

    uhd.ask(model)

    assert uhd.step_size == 0.1


def test_uhd_multiple_iterations():
    rng = np.random.default_rng(42)
    model = _make_simple_model()

    adapter = SimpleAdapter([0.1], rng=rng)
    perturbator = SimplePerturbator(rng=rng)
    uhd = UHD(step_size_adapter=adapter, perturbator=perturbator)

    y_values = [0.1, 0.2, 0.15, 0.3, 0.25]
    for y in y_values:
        seed = uhd.ask(model)
        uhd.tell(model, seed, y=y, y_var=0.1)

    assert uhd.races == 5
    assert uhd.accepts == 3
    assert uhd.incumbent_y == 0.3
