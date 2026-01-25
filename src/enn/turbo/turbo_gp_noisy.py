from __future__ import annotations
from .turbo_gp_base import TurboGPBase


class TurboGPNoisy(TurboGPBase):
    def __init__(
        self,
        train_x,
        train_y,
        train_y_var,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims: int,
        *,
        learn_additional_noise: bool = True,
    ) -> None:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
        from gpytorch.means import ConstantMean

        likelihood = FixedNoiseGaussianLikelihood(
            noise=train_y_var,
            learn_additional_noise=learn_additional_noise,
        )
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
