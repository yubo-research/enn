import torch
from torch import nn


def perturb_module(module: nn.Module, seed: int, sigma: float) -> None:
    _perturb_module(module, seed, sigma, sign=1)


def unperturb_module(module: nn.Module, seed: int, sigma: float) -> None:
    _perturb_module(module, seed, sigma, sign=-1)


def _perturb_module(module: nn.Module, seed: int, sigma: float, sign: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for param in module.parameters():
        noise = torch.randn(param.shape, generator=generator, dtype=param.dtype)
        if param.device.type != "cpu":
            noise = noise.to(param.device)
        param.data.add_(sign * noise * sigma)
