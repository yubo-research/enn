from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from .posterior_result import PosteriorResult
from .surrogate_result import SurrogateResult

if TYPE_CHECKING:
    from numpy.random import Generator


class GPSurrogate:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._y_mean: float | Any = 0.0
        self._y_std: float | Any = 1.0
        self._lengthscales: np.ndarray | None = None

    @property
    def lengthscales(self) -> np.ndarray | None:
        return self._lengthscales

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        y_var: np.ndarray | None = None,
        *,
        num_steps: int = 0,
        rng: Generator | None = None,
    ) -> SurrogateResult:
        from ..turbo_gp_fit import fit_gp

        x_obs = np.asarray(x_obs, dtype=float)
        y_obs = np.asarray(y_obs, dtype=float)
        num_dim = x_obs.shape[1]
        if y_obs.ndim == 2 and y_obs.shape[1] == 1:
            y_obs = y_obs.ravel()
        gp_result = fit_gp(
            x_obs.tolist(),
            y_obs.tolist(),
            num_dim,
            yvar_obs_list=y_var.ravel().tolist() if y_var is not None else None,
            num_steps=num_steps,
        )
        self._model = gp_result.model
        if gp_result.y_mean is not None:
            self._y_mean = gp_result.y_mean
        if gp_result.y_std is not None:
            self._y_std = gp_result.y_std
        lengthscales = None
        if self._model is not None:
            lengthscale = (
                self._model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()
            )
            if lengthscale.ndim == 3:
                lengthscale = lengthscale.mean(axis=0)
            lengthscales = lengthscale.ravel()
            lengthscales_stabilized = lengthscales / lengthscales.mean()
            del lengthscales
            lengthscales_geom_normed = lengthscales_stabilized / np.prod(
                np.power(lengthscales_stabilized, 1.0 / len(lengthscales_stabilized))
            )
            self._lengthscales = lengthscales_geom_normed
        return SurrogateResult(model=self._model, lengthscales=self._lengthscales)

    def _as_2d(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim == 2:
            return a.T
        raise ValueError(a.shape)

    def _unstandardize(self, y_std_2d: np.ndarray) -> np.ndarray:
        y_std_2d = np.asarray(y_std_2d, dtype=float)
        if y_std_2d.ndim != 2:
            raise ValueError(y_std_2d.shape)
        y_mean = np.asarray(self._y_mean, dtype=float).reshape(-1)
        y_std = np.asarray(self._y_std, dtype=float).reshape(-1)
        num_metrics = y_std_2d.shape[1]
        if y_mean.size == 1 and num_metrics != 1:
            y_mean = np.full(num_metrics, float(y_mean[0]), dtype=float)
        if y_std.size == 1 and num_metrics != 1:
            y_std = np.full(num_metrics, float(y_std[0]), dtype=float)
        return y_mean.reshape(1, -1) + y_std.reshape(1, -1) * y_std_2d

    def predict(self, x: np.ndarray) -> PosteriorResult:
        import torch
        from ..turbo_utils import get_gp_posterior_suppress_warning

        if self._model is None:
            raise RuntimeError("GPSurrogate.predict requires a fitted model")
        x_torch = torch.as_tensor(x, dtype=torch.float64)
        with torch.no_grad():
            posterior = get_gp_posterior_suppress_warning(self._model, x_torch)
            mu_std = posterior.mean.cpu().numpy()
            var_std = posterior.variance.cpu().numpy()
        mu = self._unstandardize(self._as_2d(mu_std))
        sigma_std_2d = self._as_2d(var_std**0.5)
        y_std = np.asarray(self._y_std, dtype=float).reshape(-1)
        if y_std.size == 1 and sigma_std_2d.shape[1] != 1:
            y_std = np.full(sigma_std_2d.shape[1], float(y_std[0]), dtype=float)
        sigma = y_std.reshape(1, -1) * sigma_std_2d
        return PosteriorResult(mu=mu, sigma=sigma)

    def get_incumbent_candidate_indices(self, y_obs: np.ndarray) -> np.ndarray:
        return np.arange(len(y_obs), dtype=int)

    def sample(self, x: np.ndarray, num_samples: int, rng: Generator) -> np.ndarray:
        import gpytorch
        import torch
        from ..turbo_utils import torch_seed_context

        if self._model is None:
            raise RuntimeError("GPSurrogate.sample requires a fitted model")
        x_torch = torch.as_tensor(x, dtype=torch.float64)
        seed = int(rng.integers(2**31 - 1))
        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            torch_seed_context(seed, device=x_torch.device),
        ):
            posterior = self._model.posterior(x_torch)
            samples = posterior.sample(sample_shape=torch.Size([num_samples]))
        samples_np = samples.detach().cpu().numpy()
        num_candidates = len(x)
        num_metrics = len(self._y_mean) if hasattr(self._y_mean, "__len__") else 1
        if samples_np.ndim == 2:
            samples_np = samples_np[:, :, np.newaxis]
        else:
            assert samples_np.shape == (num_samples, num_metrics, num_candidates), (
                f"GP raw samples shape mismatch: got {samples_np.shape}, "
                f"expected ({num_samples}, {num_metrics}, {num_candidates})"
            )
            samples_np = np.transpose(samples_np, (0, 2, 1))
        assert samples_np.shape == (num_samples, num_candidates, num_metrics), (
            f"GP samples shape after transpose: got {samples_np.shape}, "
            f"expected ({num_samples}, {num_candidates}, {num_metrics})"
        )
        y_mean = np.asarray(self._y_mean, dtype=float).reshape(1, 1, -1)
        y_std = np.asarray(self._y_std, dtype=float).reshape(1, 1, -1)
        result = y_mean + y_std * samples_np
        assert result.shape == (num_samples, num_candidates, num_metrics)
        return result
