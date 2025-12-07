import numpy as np
import pytest
import torch
import torch.nn as nn

from uhd.hoeffding_racer import CURRENT_SEED, HoeffdingRacer


class SimpleModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)


def test_hoeffding_racer_first_ask_returns_current_seed():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()
    seed = racer.ask(module)
    assert seed == CURRENT_SEED


def test_hoeffding_racer_second_ask_returns_int_and_perturbs():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    params_before = {n: p.clone() for n, p in module.named_parameters()}

    seed1 = racer.ask(module)
    assert seed1 == CURRENT_SEED

    racer.tell(seed1, y=-2.3, y_var=0.01)

    seed2 = racer.ask(module)
    assert seed2 != CURRENT_SEED
    assert seed2 >= 0

    for n, p in module.named_parameters():
        assert not torch.allclose(p, params_before[n])


def test_hoeffding_racer_challenger_wins():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-2.0, y_var=0.01)

    seed = racer.ask(module)
    assert seed != CURRENT_SEED
    racer.tell(seed, y=-1.0, y_var=0.01)

    seed = racer.ask(module)
    assert seed == CURRENT_SEED

    assert racer._incumbent.n == 1
    assert racer._incumbent.mean == -1.0


def test_hoeffding_racer_current_wins():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    params_initial = {n: p.clone() for n, p in module.named_parameters()}

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.0, y_var=0.01)

    seed = racer.ask(module)
    assert seed != CURRENT_SEED
    racer.tell(seed, y=-2.0, y_var=0.01)

    seed = racer.ask(module)
    assert seed == CURRENT_SEED

    for n, p in module.named_parameters():
        assert torch.allclose(p, params_initial[n])


def test_hoeffding_racer_overlapping_bounds_continues_race():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.5, y_var=1.0)

    challenger_seed = racer.ask(module)
    racer.tell(challenger_seed, y=-1.4, y_var=1.0)

    seed = racer.ask(module)
    assert seed == challenger_seed


def test_hoeffding_racer_incumbent_tighter_bounds_continues_race():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.5, y_var=0.01)

    challenger_seed = racer.ask(module)
    racer.tell(challenger_seed, y=-1.5, y_var=1.0)

    seed = racer.ask(module)
    assert seed == challenger_seed


def test_hoeffding_racer_accumulates_multiple_observations():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.5, y_var=1.0)
    racer.tell(seed, y=-1.6, y_var=1.0)
    racer.tell(seed, y=-1.4, y_var=1.0)

    assert racer._incumbent.n == 3

    challenger_seed = racer.ask(module)
    racer.tell(challenger_seed, y=-1.0, y_var=1.0)
    racer.tell(challenger_seed, y=-0.9, y_var=1.0)

    assert racer._challenger.n == 2


def test_hoeffding_racer_challenger_nested_swaps():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.5, y_var=1.0)

    challenger_seed = racer.ask(module)
    racer.tell(challenger_seed, y=-1.5, y_var=0.01)

    seed = racer.ask(module)
    assert seed == challenger_seed

    assert racer._incumbent.n == 1
    assert racer._incumbent.var == 0.01
    assert racer._challenger.n == 1
    assert racer._challenger.var == 1.0
    assert racer._challenger_seed < 0  # negative seed indicates swapped state


def test_hoeffding_racer_swap_correctly_navigates_module():
    """After a swap, the signed seed should correctly reverse perturb/unperturb."""
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    # Record initial incumbent params
    initial_params = module.linear.weight.clone()

    # First ask returns CURRENT_SEED (incumbent)
    seed = racer.ask(module)
    assert seed == CURRENT_SEED
    racer.tell(seed, y=-1.5, y_var=1.0)

    # Second ask perturbs to challenger
    challenger_seed = racer.ask(module)
    challenger_params = module.linear.weight.clone()
    assert not torch.allclose(initial_params, challenger_params)  # params changed

    # Tell with tight variance to trigger swap (challenger nested)
    racer.tell(challenger_seed, y=-1.5, y_var=0.01)

    # Third ask should trigger swap and return same seed
    seed = racer.ask(module)
    assert seed == challenger_seed
    assert racer._challenger_seed < 0  # seed is now negative

    # After swap: old challenger is new incumbent, module moved to new challenger
    # New challenger = old incumbent, so params should match initial
    params_after_swap = module.linear.weight.clone()
    assert torch.allclose(params_after_swap, initial_params)

    # Tell on new challenger (old incumbent), then ask again
    racer.tell(seed, y=-1.5, y_var=1.0)
    seed = racer.ask(module)

    # Should still be at new challenger (old incumbent params)
    assert torch.allclose(module.linear.weight, initial_params)


def test_hoeffding_racer_tell_seed_mismatch_raises():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)
    racer.tell(seed, y=-1.5, y_var=1.0)

    challenger_seed = racer.ask(module)
    racer.tell(challenger_seed, y=-1.0, y_var=1.0)

    wrong_seed = challenger_seed + 1
    with pytest.raises(ValueError, match="Seed mismatch"):
        racer.tell(wrong_seed, y=-1.0, y_var=1.0)


def test_hoeffding_racer_tell_invalid_y_var_raises():
    rng = np.random.default_rng(42)
    racer = HoeffdingRacer(step_size=0.01, k=2.0, rng=rng)
    module = SimpleModule()

    seed = racer.ask(module)

    with pytest.raises(ValueError, match="y_var must be positive"):
        racer.tell(seed, y=-1.0, y_var=0.0)

    with pytest.raises(ValueError, match="y_var must be positive"):
        racer.tell(seed, y=-1.0, y_var=-0.1)
