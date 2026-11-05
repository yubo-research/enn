from __future__ import annotations

from typing import Any

import numpy as np

from .config.acquisition import acquisition_kind
from .config.candidate_rv import CandidateRV
from .config.init_strategies import LHDOnlyInit
from .config.morbo_tr_config import MorboTRConfig
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .config.trust_region import NoTRConfig
from .config.turbo_tr_config import TurboTRConfig


def _acquisition_to_override(config: OptimizerConfig) -> dict[str, Any]:
    return {"acquisition": acquisition_kind(config.acquisition)}


def _candidate_rv_override(config: OptimizerConfig) -> dict[str, Any]:
    rv = config.candidate_rv
    if rv is CandidateRV.SOBOL:
        return {"candidate_rv": "sobol"}
    if rv is CandidateRV.UNIFORM:
        return {"candidate_rv": "uniform"}
    if rv is CandidateRV.RAASP:
        return {"candidate_rv": "raasp"}
    return {}


def _candidate_count_override(config: OptimizerConfig) -> dict[str, Any]:
    candidates = config.candidates
    return {
        "min_candidates": int(candidates.min_candidates),
        "max_candidates": int(candidates.max_candidates),
        "num_candidates_per_dim": int(candidates.num_candidates_per_dim),
        "num_candidates_per_arm": int(candidates.num_candidates_per_arm),
    }


def _candidates_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(_candidate_rv_override(config))
    out.update(_candidate_count_override(config))
    if config.candidates.raasp_driver.name == "FAST":
        out["raasp_fast"] = True
    return out


def _put_set_lengths(out: dict[str, Any], tr: TurboTRConfig | MorboTRConfig) -> None:
    lengths = {
        "length_init": tr.length_init,
        "length_min": tr.length_min,
        "length_max": tr.length_max,
    }
    for name, value in lengths.items():
        if value is not None:
            out[name] = float(value)


def _trust_region_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tr = config.trust_region
    if isinstance(tr, MorboTRConfig):
        out["trust_region"] = "morbo"
        out["num_metrics"] = int(tr.num_metrics)
        out["alpha"] = float(tr.alpha)
        _put_set_lengths(out, tr)
        out["rescalarize"] = tr.rescalarize.value
        if tr.noise_aware:
            out["noise_aware"] = True
        return out
    if isinstance(tr, NoTRConfig):
        out["trust_region"] = "global"
    else:
        _put_set_lengths(out, tr)
    if tr.noise_aware:
        out["noise_aware"] = True
    return out


def _metric_overrides(surrogate: ENNSurrogateConfig, overrides: dict[str, Any]) -> None:
    if surrogate.tied_dims:
        overrides["tied_dims"] = [list(g) for g in surrogate.tied_dims]
    if surrogate.fit.affine_calibrate:
        overrides["affine_calibrate"] = True


def _config_to_rust_overrides(config: OptimizerConfig) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    overrides.update(_acquisition_to_override(config))
    overrides.update(_candidates_to_override(config))
    overrides.update(_trust_region_to_override(config))
    surrogate = config.surrogate
    if isinstance(surrogate, ENNSurrogateConfig):
        if surrogate.num_fit_samples is None:
            overrides["freeze_params"] = True
        else:
            overrides["num_fit_samples"] = int(surrogate.num_fit_samples)
        if surrogate.num_fit_candidates is not None:
            overrides["num_fit_candidates"] = int(surrogate.num_fit_candidates)
        if surrogate.fit.infer_aleatoric_variance_scale is not None:
            overrides["infer_aleatoric_variance"] = bool(
                surrogate.fit.infer_aleatoric_variance_scale
            )
        overrides["weighting"] = surrogate.weighting
        _metric_overrides(surrogate, overrides)
        if surrogate.y_bounds is not None:
            overrides["y_bounds"] = np.asarray(surrogate.y_bounds, dtype=float)
    return overrides if overrides else None


def is_rust_supported_config(config: OptimizerConfig) -> bool:
    if isinstance(config.surrogate, ENNSurrogateConfig):
        return True
    return bool(isinstance(config.surrogate, NoSurrogateConfig))


def _is_lhd_only_config(config: OptimizerConfig) -> bool:
    return isinstance(config.init.init_strategy, LHDOnlyInit)
