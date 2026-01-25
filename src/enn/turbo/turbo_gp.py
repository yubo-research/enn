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
        import torch
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.means import ConstantMean

        super().__init__(train_x, train_y, likelihood)
        batch_shape = (
            torch.Size(train_y.shape[:-1])
            if getattr(train_y, "ndim", 0) > 1
            else torch.Size()
        )
        self.mean_module = ConstantMean(batch_shape=batch_shape)
        base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=ard_dims,
            batch_shape=batch_shape,
            lengthscale_constraint=lengthscale_constraint,
        )
        self.covar_module = ScaleKernel(
            base_kernel,
            batch_shape=batch_shape,
            outputscale_constraint=outputscale_constraint,
        )
