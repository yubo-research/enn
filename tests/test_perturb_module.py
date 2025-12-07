import time

import torch
from torch import nn

from uhd.perturb_module import perturb_module, unperturb_module


def test_perturb_module_changes_parameters():
    module = nn.Linear(10, 5)
    original_weight = module.weight.data.clone()
    original_bias = module.bias.data.clone()

    perturb_module(module, seed=42, sigma=0.1)

    assert not torch.allclose(module.weight.data, original_weight)
    assert not torch.allclose(module.bias.data, original_bias)


def test_perturb_module_deterministic_with_same_seed():
    module1 = nn.Linear(10, 5)
    module2 = nn.Linear(10, 5)

    with torch.no_grad():
        module2.weight.copy_(module1.weight)
        module2.bias.copy_(module1.bias)

    perturb_module(module1, seed=123, sigma=0.5)
    perturb_module(module2, seed=123, sigma=0.5)

    assert torch.allclose(module1.weight.data, module2.weight.data)
    assert torch.allclose(module1.bias.data, module2.bias.data)


def test_perturb_module_different_seeds_produce_different_results():
    module1 = nn.Linear(10, 5)
    module2 = nn.Linear(10, 5)

    with torch.no_grad():
        module2.weight.copy_(module1.weight)
        module2.bias.copy_(module1.bias)

    perturb_module(module1, seed=1, sigma=0.5)
    perturb_module(module2, seed=2, sigma=0.5)

    assert not torch.allclose(module1.weight.data, module2.weight.data)


def test_unperturb_module_restores_original():
    module = nn.Linear(10, 5)
    original_weight = module.weight.data.clone()
    original_bias = module.bias.data.clone()

    perturb_module(module, seed=42, sigma=0.1)
    unperturb_module(module, seed=42, sigma=0.1)

    assert torch.allclose(module.weight.data, original_weight)
    assert torch.allclose(module.bias.data, original_bias)


def test_perturb_module_timing():
    import numpy as np

    module = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    num_params = sum(p.numel() for p in module.parameters())

    perturb_module(module, seed=999, sigma=0.01)  # warm-up

    times_ms = []
    for i in range(10):
        start = time.perf_counter()
        perturb_module(module, seed=i, sigma=0.01)
        times_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = np.mean(times_ms)
    std_ms = np.std(times_ms)
    print(
        f"\nPerturbed {num_params:,} parameters: {mean_ms:.3f} ± {std_ms:.3f} ms (n=10)"
    )
