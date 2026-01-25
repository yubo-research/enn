from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.random import Generator


def get_x_center_fallback(
    config: Any,
    x_obs_list: list,
    y_obs_list: list,
    rng: Generator,
    tr_state: Any = None,
) -> np.ndarray | None:
    import numpy as np
    from .components.incumbent_selector import ScalarIncumbentSelector

    y_array = np.asarray(y_obs_list, dtype=float)
    if y_array.size == 0:
        return None
    x_array = np.asarray(x_obs_list, dtype=float)
    if tr_state is not None and hasattr(tr_state, "incumbent_selector"):
        selector = tr_state.incumbent_selector
    else:
        selector = ScalarIncumbentSelector(noise_aware=False)
    idx = selector.select(y_array, None, rng)
    return x_array[idx]


def handle_restart_clear_always(
    x_obs_list: list,
    y_obs_list: list,
    yvar_obs_list: list,
) -> tuple[bool, int]:
    x_obs_list.clear()
    y_obs_list.clear()
    yvar_obs_list.clear()
    return True, 0


def handle_restart_check_multi_objective(
    tr_state: Any,
    x_obs_list: list,
    y_obs_list: list,
    yvar_obs_list: list,
    init_idx: int,
) -> tuple[bool, int]:
    is_multi = (
        tr_state is not None
        and hasattr(tr_state, "num_metrics")
        and tr_state.num_metrics > 1
    )
    if is_multi:
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
