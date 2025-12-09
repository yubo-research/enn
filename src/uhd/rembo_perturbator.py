from __future__ import annotations

import torch
from torch import nn

from uhd.rembo import rembo_perturb, rembo_unperturb


class REMBOPerturbator:
    def __init__(
        self,
        num_dim_z: int,
        seed_A: int,
        s_rembo: int = 4,
        seed: int | None = None,
    ) -> None:
        self._num_dim_z = num_dim_z
        self._seed_A = seed_A
        self._s_rembo = s_rembo
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)
        self._incumbent_y: float = float("-inf")
        self._dz: torch.Tensor | None = None
        self._counter: int = 0

    def ask(self, module: nn.Module, step_size: float) -> int:
        self._dz = torch.randn(self._num_dim_z, generator=self._generator) * step_size
        rembo_perturb(module, self._seed_A, self._dz, s=self._s_rembo)
        self._counter += 1
        return self._counter

    def tell(self, module: nn.Module, seed: int, y: float, y_var: float) -> None:
        assert seed == self._counter
        accepted = y > self._incumbent_y
        if accepted:
            self._incumbent_y = y
        else:
            rembo_unperturb(module, self._seed_A, self._dz, s=self._s_rembo)
        self._dz = None
