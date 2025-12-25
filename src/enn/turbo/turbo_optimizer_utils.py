from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def sobol_seed_for_state(seed_base: int, *, n_obs: int, num_arms: int) -> int:
    mask64 = (1 << 64) - 1
    x = int(seed_base) & mask64
    x ^= (int(n_obs) + 1) * 0x9E3779B97F4A7C15 & mask64
    x ^= (int(num_arms) + 1) * 0xBF58476D1CE4E5B9 & mask64
    x = (x + 0x9E3779B97F4A7C15) & mask64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & mask64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & mask64
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFF)


def validate_tell_inputs(
    x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None, num_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    import numpy as np

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape[1] != num_dim:
        raise ValueError(x.shape)

    if y.ndim == 2:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = y.shape[1]
    elif y.ndim == 1:
        if y.shape[0] != x.shape[0]:
            raise ValueError((x.shape, y.shape))
        num_metrics = 1
    else:
        raise ValueError(y.shape)

    if y_var is not None:
        y_var = np.asarray(y_var, dtype=float)
        if y_var.shape != y.shape:
            raise ValueError((y.shape, y_var.shape))

    return x, y, y_var, num_metrics


def trim_trailing_observations(
    x_obs_list: list,
    y_obs_list: list,
    y_tr_list: list,
    yvar_obs_list: list,
    *,
    trailing_obs: int,
    incumbent_indices: np.ndarray,
) -> tuple[list, list, list, list]:
    import numpy as np

    num_total = len(x_obs_list)
    if num_total <= trailing_obs:
        return x_obs_list, y_obs_list, y_tr_list, yvar_obs_list

    start_idx = max(0, num_total - trailing_obs)
    recent_indices = set(range(start_idx, num_total))
    keep_indices = set(incumbent_indices.tolist()) | recent_indices

    if len(keep_indices) > trailing_obs:
        keep_indices = set(incumbent_indices.tolist())
        remaining_slots = trailing_obs - len(keep_indices)
        if remaining_slots > 0:
            recent_non_incumbent = [
                i for i in range(num_total - 1, -1, -1) if i not in keep_indices
            ][:remaining_slots]
            keep_indices.update(recent_non_incumbent)

    indices = np.array(sorted(keep_indices), dtype=int)

    x_array = np.asarray(x_obs_list, dtype=float)
    y_obs_array = np.asarray(y_obs_list, dtype=float)
    y_tr_array = np.asarray(y_tr_list, dtype=float)

    new_x = x_array[indices].tolist()
    new_y_obs = y_obs_array[indices].tolist()
    new_y_tr = y_tr_array[indices].tolist()
    new_yvar = yvar_obs_list
    if len(yvar_obs_list) == len(y_obs_array):
        yvar_array = np.asarray(yvar_obs_list, dtype=float)
        new_yvar = yvar_array[indices].tolist()

    return new_x, new_y_obs, new_y_tr, new_yvar
