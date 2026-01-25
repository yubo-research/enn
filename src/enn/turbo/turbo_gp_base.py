from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpytorch.distributions import MultivariateNormal


def _get_exact_gp_base():
    from gpytorch.models import ExactGP

    return ExactGP


class TurboGPBase(_get_exact_gp_base()):
    mean_module: Any
    covar_module: Any

    def forward(self, x) -> MultivariateNormal:
        from gpytorch.distributions import MultivariateNormal

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self, x) -> MultivariateNormal:
        return self(x)
