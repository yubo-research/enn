from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from embedding.sparse_jl_contribution import get_sparse_jl_contribution
from uhd.perturb_module import perturb_module, unperturb_module

if TYPE_CHECKING:
    pass


def embed_perturbation_streaming(
    module: nn.Module,
    seed: int,
    step_size: float,
    num_dim_embed: int,
    embed_seed: int,
    s: int = 4,
) -> np.ndarray:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    embedding = np.zeros(num_dim_embed, dtype=np.float64)
    input_idx = 0

    for param in module.parameters():
        noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
        noise_flat = noise.flatten().numpy()

        for val in noise_flat:
            scaled_val = float(val) * step_size
            if scaled_val != 0.0:
                contributions = get_sparse_jl_contribution(
                    input_idx, scaled_val, num_dim_embed, s, embed_seed
                )
                for row, contrib in contributions:
                    embedding[row] += contrib
            input_idx += 1

    return embedding


class ENNPerturbator:
    def __init__(
        self,
        num_dim_embed: int,
        num_candidates: int = 10,
        seed: int | None = None,
        k: int = 5,
        var_scale: float = 1.0,
        s_embed: int = 4,
    ) -> None:
        self._num_dim_embed = num_dim_embed
        self._num_candidates = num_candidates
        self._k = k
        self._var_scale = var_scale
        self._s_embed = s_embed

        self._rng = np.random.default_rng(seed)
        self._embed_seed = int(self._rng.integers(1, 2**31))

        self._observations: list[tuple[int, float, float, float]] = []
        self._incumbent_y: float = float("-inf")
        self._last_seed: int | None = None
        self._last_step_size: float | None = None

    def ask(self, module: nn.Module, step_size: float) -> int:
        from enn.enn import EpistemicNearestNeighbors
        from enn.enn_params import ENNParams
        from enn.enn_util import arms_from_pareto_fronts

        candidate_seeds = [
            int(self._rng.integers(1, 2**31)) for _ in range(self._num_candidates)
        ]

        if len(self._observations) == 0:
            chosen_seed = candidate_seeds[0]
            perturb_module(module, chosen_seed, step_size)
            self._last_seed = chosen_seed
            self._last_step_size = step_size
            return chosen_seed

        train_x = []
        train_y = []
        train_yvar = []
        for obs_seed, obs_step_size, obs_y, obs_yvar in self._observations:
            x = embed_perturbation_streaming(
                module,
                obs_seed,
                obs_step_size,
                self._num_dim_embed,
                self._embed_seed,
                self._s_embed,
            )
            train_x.append(x)
            train_y.append([obs_y])
            train_yvar.append([obs_yvar])

        train_x_arr = np.array(train_x, dtype=np.float64)
        train_y_arr = np.array(train_y, dtype=np.float64)
        train_yvar_arr = np.array(train_yvar, dtype=np.float64)

        enn = EpistemicNearestNeighbors(train_x_arr, train_y_arr, train_yvar_arr)
        params = ENNParams(
            k=min(self._k, len(self._observations)), var_scale=self._var_scale
        )

        cand_x = []
        for cand_seed in candidate_seeds:
            x = embed_perturbation_streaming(
                module,
                cand_seed,
                step_size,
                self._num_dim_embed,
                self._embed_seed,
                self._s_embed,
            )
            cand_x.append(x)

        cand_x_arr = np.array(cand_x, dtype=np.float64)
        posterior = enn.batch_posterior(cand_x_arr, [params])
        mu = posterior.mu[0, :, 0]
        se = posterior.se[0, :, 0]

        chosen_x = arms_from_pareto_fronts(
            cand_x_arr, mu, se, num_arms=1, rng=self._rng
        )

        chosen_idx = None
        for i, x in enumerate(cand_x):
            if np.allclose(x, chosen_x[0]):
                chosen_idx = i
                break

        if chosen_idx is None:
            chosen_idx = 0

        chosen_seed = candidate_seeds[chosen_idx]
        perturb_module(module, chosen_seed, step_size)
        self._last_seed = chosen_seed
        self._last_step_size = step_size
        return chosen_seed

    def tell(self, module: nn.Module, seed: int, y: float, y_var: float) -> None:
        assert seed == self._last_seed

        self._observations.append((seed, self._last_step_size, y, y_var))

        accepted = y > self._incumbent_y
        if accepted:
            self._incumbent_y = y
        else:
            unperturb_module(module, self._last_seed, self._last_step_size)
