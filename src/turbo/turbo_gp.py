from __future__ import annotations

from .turbo_gp_base import TurboGPBase


class TurboGP(TurboGPBase):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims: int,
    ) -> None:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_dims,
            lengthscale_constraint=lengthscale_constraint,
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=outputscale_constraint,
        )
