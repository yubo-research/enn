from __future__ import annotations

import os
from typing import Any

import numpy as np

from .config.acquisition import UCBAcquisitionConfig, acquisition_kind
from .config.candidate_gen_config import CandidateGenConfig
from .config.candidate_rv import CandidateRV
from .config.enn_x_scaling import ENNMetricLearning, ENNScaleX
from .config.init_strategies import LHDOnlyInit
from .config.morbo_tr_config import MorboTRConfig
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .config.trust_region import NoTRConfig, TurboTRConfig

def _acquisition_to_override(config: OptimizerConfig) -> dict[str, Any]:
    acq = getattr(config, "acquisition", None)
    if acq is None:
        return {}
    kind = acquisition_kind(acq)
    if isinstance(acq, UCBAcquisitionConfig):
        return {
            "acquisition": kind,
            "acquisition_beta": float(getattr(acq, "beta", 2.0)),
        }
    return {"acquisition": kind}


def _candidate_rv_override(config: OptimizerConfig) -> dict[str, Any]:
    rv = getattr(config, "candidate_rv", None)
    if rv is CandidateRV.SOBOL:
        return {"candidate_rv": "sobol"}
    if rv is CandidateRV.UNIFORM:
        return {"candidate_rv": "uniform"}
    if rv is CandidateRV.RAASP:
        return {"candidate_rv": "raasp"}
    return {}


def _candidate_count_override(config: OptimizerConfig) -> dict[str, Any]:
    candidates = getattr(config, "candidates", None)
    if not isinstance(candidates, CandidateGenConfig):
        return {}
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


def _length_or_none(tr: TurboTRConfig, name: str) -> float | None:
    value = getattr(tr, name) if hasattr(tr, name) else getattr(tr.length, name, None)
    if value is None:
        return None
    return float(value)


def _put_set_lengths(out: dict[str, Any], tr: TurboTRConfig) -> None:
    for name in ("length_init", "length_min", "length_max"):
        value = _length_or_none(tr, name)
        if value is not None:
            out[name] = value


def _trust_region_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tr = getattr(config, "trust_region", None)
    if isinstance(tr, MorboTRConfig):
        out["trust_region"] = "morbo"
        out["num_metrics"] = int(tr.num_metrics)
        out["alpha"] = float(tr.alpha)
        _put_set_lengths(out, tr)
        out["rescalarize"] = tr.rescalarize.value
        if tr.noise_aware:
            out["noise_aware"] = True
        return out
    if not isinstance(tr, TurboTRConfig):
        return out
    _put_set_lengths(out, tr)
    if tr.noise_aware:
        out["noise_aware"] = True
    return out


def _metric_overrides(surrogate: ENNSurrogateConfig, overrides: dict[str, Any]) -> None:
    if surrogate.metric_learning == ENNMetricLearning.AUTO:
        overrides["metric_learning"] = "auto"
    if surrogate.tied_dims:
        overrides["tied_dims"] = [list(g) for g in surrogate.tied_dims]
    if surrogate.fit.affine_calibrate:
        overrides["affine_calibrate"] = True


def _config_to_rust_overrides(config: OptimizerConfig) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    overrides.update(_acquisition_to_override(config))
    overrides.update(_candidates_to_override(config))
    overrides.update(_trust_region_to_override(config))
    surrogate = getattr(config, "surrogate", None)
    if isinstance(surrogate, ENNSurrogateConfig):
        from .config.enn_index_driver import ENN_INDEX_DRIVER_TO_RUST

        if surrogate.index_driver in ENN_INDEX_DRIVER_TO_RUST:
            overrides["index_driver"] = ENN_INDEX_DRIVER_TO_RUST[surrogate.index_driver]
        if surrogate.num_fit_samples is not None:
            overrides["num_fit_samples"] = int(surrogate.num_fit_samples)
        if surrogate.num_fit_candidates is not None:
            overrides["num_fit_candidates"] = int(surrogate.num_fit_candidates)

        overrides["infer_aleatoric_variance"] = bool(
            surrogate.fit.infer_aleatoric_variance_scale
        )
        if surrogate.scale_x == ENNScaleX.ON:
            overrides["scale_x"] = True
        _metric_overrides(surrogate, overrides)
        if surrogate.y_bounds is not None:
            overrides["y_bounds"] = np.asarray(surrogate.y_bounds, dtype=float)
        if surrogate.enn_storage is not None:
            overrides["enn_storage"] = surrogate.enn_storage.name
        if surrogate.work_dir is not None:
            overrides["work_dir"] = os.fspath(surrogate.work_dir)
    return overrides if overrides else None


def is_rust_supported_config(config: OptimizerConfig) -> bool:
    if isinstance(config.surrogate, ENNSurrogateConfig):
        return True
    if isinstance(config.surrogate, NoSurrogateConfig):
        return True
    return False


def _is_lhd_only_config(config: OptimizerConfig) -> bool:
    return (
        isinstance(config.trust_region, NoTRConfig)
        and isinstance(config.init.init_strategy, LHDOnlyInit)
        and isinstance(config.surrogate, NoSurrogateConfig)
    )
