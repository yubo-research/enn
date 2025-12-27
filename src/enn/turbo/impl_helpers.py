from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def create_trust_region(
    config: Any,
    num_dim: int,
    num_arms: int,
    rng: Generator,
    num_metrics: int | None = None,
) -> Any:
    if config.tr_type == "none":
        from .no_trust_region import NoTrustRegion

        return NoTrustRegion(num_dim=num_dim, num_arms=num_arms)
    elif config.tr_type == "turbo":
        from .turbo_trust_region import TurboTrustRegion

        return TurboTrustRegion(num_dim=num_dim, num_arms=num_arms)
    elif config.tr_type == "morbo":
        from .morbo_trust_region import MorboTrustRegion

        effective_num_metrics = num_metrics or config.num_metrics
        if effective_num_metrics is None:
            raise ValueError("num_metrics required for tr_type='morbo'")
        return MorboTrustRegion(
            num_dim=num_dim,
            num_arms=num_arms,
            num_metrics=effective_num_metrics,
            rng=rng,
        )
    else:
        raise ValueError(f"Unknown tr_type: {config.tr_type!r}")


def get_x_center_fallback(
    config: Any,
    x_obs_list: list,
    y_obs_list: list,
    rng: Generator,
    tr_state: Any = None,
) -> np.ndarray | None:
    import numpy as np

    from .turbo_utils import argmax_random_tie

    y_array = np.asarray(y_obs_list, dtype=float)
    if y_array.size == 0:
        return None
    x_array = np.asarray(x_obs_list, dtype=float)

    if config.tr_type == "morbo" and tr_state is not None:
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, tr_state.num_metrics)
        scalarized = tr_state.scalarize(y_array, clip=True)
        idx = argmax_random_tie(scalarized, rng=rng)
    else:
        idx = argmax_random_tie(y_array, rng=rng)

    return x_array[idx]


def handle_restart_clear_always(
    x_obs_list: list,
    y_obs_list: list,
    yvar_obs_list: list,
) -> tuple[bool, int]:
    """Clear all observation lists and return (True, 0) - used by turbo_one and turbo_enn."""
    x_obs_list.clear()
    y_obs_list.clear()
    yvar_obs_list.clear()
    return True, 0


def handle_restart_check_morbo(
    config: Any,
    x_obs_list: list,
    y_obs_list: list,
    yvar_obs_list: list,
    init_idx: int,
) -> tuple[bool, int]:
    """Clear only for morbo, else preserve init_idx - used by turbo_zero and lhd_only."""
    if config.tr_type == "morbo":
        x_obs_list.clear()
        y_obs_list.clear()
        yvar_obs_list.clear()
        return True, 0
    return False, init_idx


def estimate_y_passthrough(y_observed: np.ndarray) -> np.ndarray:
    import numpy as np

    y = np.asarray(y_observed, dtype=float)
    if y.ndim == 1:
        return y.reshape(-1, 1)
    return y
