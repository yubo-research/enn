from __future__ import annotations

import torch
from torch import nn


def rembo_perturb(module: nn.Module, seed: int, z: torch.Tensor, s: int = 4) -> None:
    _rembo_perturb(module, seed, z, s, sign=1)


def rembo_unperturb(module: nn.Module, seed: int, z: torch.Tensor, s: int = 4) -> None:
    _rembo_perturb(module, seed, z, s, sign=-1)


def _rembo_perturb(
    module: nn.Module, seed: int, z: torch.Tensor, s: int, sign: int
) -> None:
    from embedding.sparse_jl_t import block_sparse_jl_transform_t

    if z.ndim != 1:
        raise ValueError(f"z must be 1D, got ndim={z.ndim}")

    total_params = sum(p.numel() for p in module.parameters())
    if total_params == 0:
        return

    delta = block_sparse_jl_transform_t(z, d=total_params, s=s, seed=seed)

    offset = 0
    for param in module.parameters():
        numel = param.numel()
        param_delta = delta[offset : offset + numel].reshape(param.shape)
        if param.device != param_delta.device:
            param_delta = param_delta.to(param.device)
        if param.dtype != param_delta.dtype:
            param_delta = param_delta.to(param.dtype)
        param.data.add_(sign * param_delta)
        offset += numel
