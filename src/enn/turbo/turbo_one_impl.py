from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator

from .base_turbo_impl import BaseTurboImpl
from .turbo_config import TurboOneConfig
from .turbo_utils import gp_thompson_sample


class TurboOneImpl(BaseTurboImpl):
    def __init__(self, config: TurboOneConfig) -> None:
        super().__init__(config)
        self._gp_model: Any | None = None
        self._gp_y_mean: float | Any = 0.0
        self._gp_y_std: float | Any = 1.0
        self._fitted_n_obs: int = 0

    def _as_2d(self, a: np.ndarray) -> np.ndarray:
        import numpy as np

        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            return a.reshape(-1, 1)
        if a.ndim == 2:
            return a.T
        raise ValueError(a.shape)

    def _broadcast_gp_mean_std(self, num_metrics: int) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        num_metrics = int(num_metrics)
        if num_metrics <= 0:
            raise ValueError(num_metrics)
        gp_y_mean = np.asarray(self._gp_y_mean, dtype=float).reshape(-1)
        gp_y_std = np.asarray(self._gp_y_std, dtype=float).reshape(-1)
        if gp_y_mean.size == 1 and num_metrics != 1:
            gp_y_mean = np.full(num_metrics, float(gp_y_mean[0]), dtype=float)
        if gp_y_std.size == 1 and num_metrics != 1:
            gp_y_std = np.full(num_metrics, float(gp_y_std[0]), dtype=float)
        if gp_y_mean.shape != (num_metrics,) or gp_y_std.shape != (num_metrics,):
            raise ValueError((gp_y_mean.shape, gp_y_std.shape, num_metrics))
        return gp_y_mean, gp_y_std

    def _unstandardize(self, y_std_2d: np.ndarray) -> np.ndarray:
        import numpy as np

        y_std_2d = np.asarray(y_std_2d, dtype=float)
        if y_std_2d.ndim != 2:
            raise ValueError(y_std_2d.shape)
        num_metrics = int(y_std_2d.shape[1])
        gp_y_mean, gp_y_std = self._broadcast_gp_mean_std(num_metrics)
        return gp_y_mean.reshape(1, -1) + gp_y_std.reshape(1, -1) * y_std_2d

    def get_x_center(
        self,
        x_obs_list: list,
        y_obs_list: list,
        rng: Generator,
        tr_state: Any = None,
    ) -> np.ndarray | None:
        import numpy as np
        import torch
        import warnings

        from .turbo_utils import argmax_random_tie

        if len(y_obs_list) == 0:
            return None
        if self._gp_model is None:
            if len(y_obs_list) <= 1:
                x_array = np.asarray(x_obs_list, dtype=float)
                y_array = np.asarray(y_obs_list, dtype=float)
                if y_array.ndim == 2:
                    if self._config.tr_type == "morbo" and tr_state is not None:
                        scores = tr_state.scalarize(y_array, clip=True)
                    else:
                        scores = y_array[:, 0]
                    best_idx = argmax_random_tie(scores, rng=rng)
                    return x_array[best_idx]
                return super().get_x_center(x_obs_list, y_obs_list, rng, tr_state)
            raise RuntimeError(
                "TurboOneImpl.get_x_center requires a fitted GP model for 2+ observations; "
                "call prepare_ask() first."
            )
        if self._fitted_n_obs != len(x_obs_list):
            raise RuntimeError(
                f"GP fitted on {self._fitted_n_obs} obs but get_x_center called with {len(x_obs_list)}"
            )

        x_array = np.asarray(x_obs_list, dtype=float)
        x_torch = torch.as_tensor(x_array, dtype=torch.float64)
        try:
            from gpytorch.utils.warnings import GPInputWarning
        except Exception:  # pragma: no cover
            GPInputWarning = None

        with torch.no_grad():
            if GPInputWarning is None:
                posterior = self._gp_model.posterior(x_torch)
            else:
                # We intentionally evaluate the GP posterior at the training inputs
                # (observed points) when choosing the center. GPyTorch warns about this
                # in debug mode, but it's expected for our usage.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"The input matches the stored training data\..*",
                        category=GPInputWarning,
                    )
                    posterior = self._gp_model.posterior(x_torch)
            mu_std = posterior.mean.cpu().numpy()

        mu = self._unstandardize(self._as_2d(mu_std))

        # For morbo: scalarize mu values
        if self._config.tr_type == "morbo" and tr_state is not None:
            scalarized = tr_state.scalarize(mu, clip=False)
            best_idx = argmax_random_tie(scalarized, rng=rng)
        else:
            if mu.shape[1] != 1:
                raise ValueError(mu.shape)
            best_idx = argmax_random_tie(mu[:, 0], rng=rng)

        return x_array[best_idx]

    def needs_tr_list(self) -> bool:
        return True

    def try_early_ask(
        self,
        num_arms: int,
        x_obs_list: list,
        draw_initial_fn: Callable[[int], np.ndarray],
        get_init_lhd_points_fn: Callable[[int], np.ndarray],
    ) -> np.ndarray | None:
        if len(x_obs_list) == 0:
            return get_init_lhd_points_fn(num_arms)
        return None

    def handle_restart(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        init_idx: int,
        num_init: int,
    ) -> tuple[bool, int]:
        x_obs_list.clear()
        y_obs_list.clear()
        yvar_obs_list.clear()
        return True, 0

    def prepare_ask(
        self,
        x_obs_list: list,
        y_obs_list: list,
        yvar_obs_list: list,
        num_dim: int,
        gp_num_steps: int,
        rng: Any | None = None,
    ) -> tuple[Any, float | None, float | None, np.ndarray | None]:
        import numpy as np

        from .turbo_utils import fit_gp

        if len(x_obs_list) == 0:
            return None, None, None, None
        self._gp_model, _likelihood, gp_y_mean_fitted, gp_y_std_fitted = fit_gp(
            x_obs_list,
            y_obs_list,
            num_dim,
            yvar_obs_list=yvar_obs_list if yvar_obs_list else None,
            num_steps=gp_num_steps,
        )
        self._fitted_n_obs = len(x_obs_list)
        if gp_y_mean_fitted is not None:
            self._gp_y_mean = gp_y_mean_fitted
        if gp_y_std_fitted is not None:
            self._gp_y_std = gp_y_std_fitted
        lengthscales = None
        if self._gp_model is not None:
            lengthscale = (
                self._gp_model.covar_module.base_kernel.lengthscale.cpu()
                .detach()
                .numpy()
            )
            if lengthscale.ndim == 3:
                lengthscale = lengthscale.mean(axis=0)
            lengthscales = lengthscale.ravel()
            # First line helps stabilize second line.
            lengthscales = lengthscales / lengthscales.mean()
            lengthscales = lengthscales / np.prod(
                np.power(lengthscales, 1.0 / len(lengthscales))
            )
        return self._gp_model, gp_y_mean_fitted, gp_y_std_fitted, lengthscales

    def select_candidates(
        self,
        x_cand: np.ndarray,
        num_arms: int,
        num_dim: int,
        rng: Generator,
        fallback_fn: Callable[[np.ndarray, int], np.ndarray],
        from_unit_fn: Callable[[np.ndarray], np.ndarray],
        tr_state: Any = None,
    ) -> np.ndarray:
        import numpy as np

        if self._gp_model is None:
            if self._fitted_n_obs >= 2:
                raise RuntimeError(
                    "TurboOneImpl.select_candidates requires a fitted GP model for 2+ observations; "
                    "call prepare_ask() first."
                )
            return fallback_fn(x_cand, num_arms)

        if self._config.tr_type == "morbo" and tr_state is not None:
            import gpytorch
            import torch

            from .turbo_utils import torch_seed_context

            x_torch = torch.as_tensor(x_cand, dtype=torch.float64)
            seed = int(rng.integers(2**31 - 1))
            with (
                torch.no_grad(),
                gpytorch.settings.fast_pred_var(),
                torch_seed_context(seed, device=x_torch.device),
            ):
                posterior = self._gp_model.posterior(x_torch)
                samples = posterior.sample(sample_shape=torch.Size([1]))

            if samples.ndim == 2:
                samples_std = samples[0].detach().cpu().numpy().reshape(-1, 1)
            elif samples.ndim == 3:
                samples_std = samples[0].detach().cpu().numpy().T
            else:
                raise ValueError(samples.shape)

            y_samples = self._unstandardize(samples_std)
            scores = tr_state.scalarize(y_samples, clip=False)
            shuffled_indices = rng.permutation(len(scores))
            shuffled_scores = scores[shuffled_indices]
            top_k_in_shuffled = np.argpartition(-shuffled_scores, num_arms - 1)[
                :num_arms
            ]
            idx = shuffled_indices[top_k_in_shuffled]
            return from_unit_fn(x_cand[idx])

        if (
            np.asarray(self._gp_y_mean).ndim != 0
            or np.asarray(self._gp_y_std).ndim != 0
        ):
            raise ValueError("multi-output GP requires tr_type='morbo'")
        idx = gp_thompson_sample(
            self._gp_model,
            x_cand,
            num_arms,
            rng,
            float(self._gp_y_mean),
            float(self._gp_y_std),
        )
        return from_unit_fn(x_cand[idx])

    def estimate_y(self, x_unit: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
        import torch

        if self._gp_model is None:
            raise RuntimeError(
                "TurboOneImpl.estimate_y requires a fitted GP model; call prepare_ask() first."
            )
        x_torch = torch.as_tensor(x_unit, dtype=torch.float64)
        with torch.no_grad():
            posterior = self._gp_model.posterior(x_torch)
            mu_std = posterior.mean.cpu().numpy()

        mu = self._unstandardize(self._as_2d(mu_std))
        if mu.shape[1] == 1:
            return mu[:, 0]
        return mu

    def get_mu_sigma(self, x_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        import torch

        if self._gp_model is None:
            return None
        x_torch = torch.as_tensor(x_unit, dtype=torch.float64)
        with torch.no_grad():
            posterior = self._gp_model.posterior(x_torch)
            mu_std = posterior.mean.cpu().numpy()
            sigma_std = posterior.variance.cpu().numpy() ** 0.5

        mu_std_2d = self._as_2d(mu_std)
        sigma_std_2d = self._as_2d(sigma_std)
        mu = self._unstandardize(mu_std_2d)
        _gp_y_mean, gp_y_std = self._broadcast_gp_mean_std(int(mu_std_2d.shape[1]))
        sigma = gp_y_std.reshape(1, -1) * sigma_std_2d
        return mu, sigma
