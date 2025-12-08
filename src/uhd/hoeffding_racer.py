import numpy as np
from torch import nn

from uhd.mvue_accumulator import MVUE
from uhd.perturb_module import perturb_module, unperturb_module

CURRENT_SEED: int = -1

_VERBOSE = False


class HoeffdingRacer:
    def __init__(self, step_size: float, k: float, rng: np.random.Generator) -> None:
        self._step_size = step_size
        self._k = k
        self._rng = rng

        self._incumbent = MVUE(decay=1.0)
        self._challenger = MVUE(decay=1.0)
        self._challenger_seed: int | None = None
        self._initialized = False
        self._accepts: int = 0
        self._races: int = 0
        self._squeeze = 0.9
        self._module_params = "incumbent"

    def _to_incumbent(self, module: nn.Module) -> None:
        if self._module_params == "challenger":
            unperturb_module(module, self._challenger_seed, self._step_size)
        self._module_params = "incumbent"

    def _to_challenger(self, module: nn.Module) -> None:
        if self._module_params == "incumbent":
            perturb_module(module, self._challenger_seed, self._step_size)
        self._module_params = "challenger"

    def ask(self, module: nn.Module) -> int:
        if not self._initialized:
            self._initialized = True
            if self._module_params != "incumbent":
                raise RuntimeError(
                    "Expected module_params to be 'incumbent' on first ask"
                )
            return CURRENT_SEED

        if self._challenger_seed is None:
            self._challenger_seed = int(self._rng.integers(1, 2**31))
            self._to_challenger(module)
            return self._challenger_seed

        if self._incumbent.n == 0:
            raise RuntimeError("Incumbent has no observations")
        if self._challenger.n > 0:
            inc_lcb, inc_ucb = self._incumbent.confidence_bounds(self._k)
            chall_lcb, chall_ucb = self._challenger.confidence_bounds(self._k)

            if chall_lcb > inc_ucb:
                if _VERBOSE:
                    print("CHALL:", inc_ucb, chall_lcb)
                self._to_challenger(module)
                self._incumbent = self._challenger
                self._challenger = MVUE(decay=1.0)
                self._challenger_seed = None
                self._accepts += 1
                self._races += 1
                self._module_params = "incumbent"
                return CURRENT_SEED
            elif inc_lcb > chall_ucb:
                if _VERBOSE:
                    print("INC:", inc_lcb, chall_ucb)
                self._to_incumbent(module)
                self._challenger = MVUE(decay=1.0)
                self._challenger_seed = None
                self._races += 1
                return CURRENT_SEED
            elif self._challenger.se < self._squeeze * self._incumbent.se:
                # Swap: challenger has tighter variance, becomes new incumbent
                # Old incumbent becomes new challenger
                old_challenger_seed = self._challenger_seed
                self._to_incumbent(module)  # Move to old incumbent position
                self._incumbent, self._challenger = self._challenger, self._incumbent
                self._challenger_seed = -abs(old_challenger_seed)
                self._module_params = "challenger"  # Module is now at new challenger
                return old_challenger_seed

        return self._challenger_seed

    @property
    def accepts(self) -> int:
        return self._accepts

    @property
    def races(self) -> int:
        return self._races

    @property
    def incumbent_se(self) -> float | None:
        if self._incumbent.n == 0:
            return None
        return self._incumbent.se

    @property
    def incumbent_mean(self) -> float | None:
        if self._incumbent.n == 0:
            return None
        return self._incumbent.mean

    @property
    def challenger_se(self) -> float | None:
        if self._challenger.n == 0:
            return None
        return self._challenger.se

    @property
    def challenger_mean(self) -> float | None:
        if self._challenger.n == 0:
            return None
        return self._challenger.mean

    def tell(self, seed: int, y: float, y_var: float) -> None:
        if not np.isfinite(y):
            raise ValueError("y must be finite")
        if not np.isfinite(y_var) or y_var <= 0:
            raise ValueError("y_var must be finite and positive")

        if seed == CURRENT_SEED:
            self._incumbent.update(y, y_var)
        else:
            if self._challenger_seed is None:
                raise RuntimeError("No challenger seed set")
            if seed != abs(self._challenger_seed):
                raise ValueError(
                    f"Seed mismatch: expected {abs(self._challenger_seed)}, got {seed}"
                )
            self._challenger.update(y, y_var)
