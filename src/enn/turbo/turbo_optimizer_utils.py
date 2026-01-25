from __future__ import annotations
from typing import TYPE_CHECKING
from .types import ObsLists, TellInputs

if TYPE_CHECKING:
    import numpy as np


def sobol_seed_for_state(
    seed_base: int, *, restart_generation: int, n_obs: int, num_arms: int
) -> int:
    mask64 = (1 << 64) - 1
    x = int(seed_base) & mask64
    x ^= (int(restart_generation) + 1) * 0xD1342543DE82EF95 & mask64
    x ^= (int(n_obs) + 1) * 0x9E3779B97F4A7C15 & mask64
    x ^= (int(num_arms) + 1) * 0xBF58476D1CE4E5B9 & mask64
    x = (x + 0x9E3779B97F4A7C15) & mask64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & mask64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & mask64
    z = z ^ (z >> 31)
    return int(z & 0xFFFFFFFF)


def reset_timing(opt: object) -> None:
    setattr(opt, "_dt_fit", 0.0)
    setattr(opt, "_dt_gen", 0.0)
    setattr(opt, "_dt_sel", 0.0)


def validate_tell_inputs(
    x: np.ndarray, y: np.ndarray, y_var: np.ndarray | None, num_dim: int
) -> TellInputs:
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
    return TellInputs(x=x, y=y, y_var=y_var, num_metrics=num_metrics)


def trim_trailing_observations(
    x_obs_list: list,
    y_obs_list: list,
    y_tr_list: list,
    yvar_obs_list: list,
    *,
    trailing_obs: int,
    incumbent_indices: np.ndarray,
) -> ObsLists:
    import numpy as np

    num_total = len(x_obs_list)
    if num_total <= trailing_obs:
        return ObsLists(
            x_obs=x_obs_list,
            y_obs=y_obs_list,
            y_tr=y_tr_list,
            yvar_obs=yvar_obs_list,
        )
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
    new_y_tr = y_tr_array[indices].tolist() if y_tr_array.size > 0 else []
    new_yvar = yvar_obs_list
    if len(yvar_obs_list) == len(y_obs_array):
        yvar_array = np.asarray(yvar_obs_list, dtype=float)
        new_yvar = yvar_array[indices].tolist()
    return ObsLists(x_obs=new_x, y_obs=new_y_obs, y_tr=new_y_tr, yvar_obs=new_yvar)
