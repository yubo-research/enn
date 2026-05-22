from __future__ import annotations

from typing import Any

from .config.acquisition import (
    DrawAcquisitionConfig,
    ParetoAcquisitionConfig,
    RandomAcquisitionConfig,
    UCBAcquisitionConfig,
)
from .config.candidate_gen_config import CandidateGenConfig
from .config.candidate_rv import CandidateRV
from .config.init_strategies import LHDOnlyInit
from .config.morbo_tr_config import MorboTRConfig
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .config.trust_region import NoTRConfig, TurboTRConfig
from .fallback_registry import requires_python_optimizer_fallback

DEFAULT_ENN_K = 10
_DEFAULT_NUM_CANDIDATES_FACTOR = 100.0
_DEFAULT_MAX_CANDIDATES = 5000


def resolve_enn_k(config: OptimizerConfig) -> int:
    surrogate = config.surrogate
    if not isinstance(surrogate, ENNSurrogateConfig):
        raise TypeError(f"expected ENNSurrogateConfig, got {type(surrogate)!r}")
    return DEFAULT_ENN_K if surrogate.k is None else int(surrogate.k)


def _acquisition_to_override(config: OptimizerConfig) -> dict[str, Any]:
    acq = getattr(config, "acquisition", None)
    if acq is None:
        return {}
    if isinstance(acq, UCBAcquisitionConfig):
        return {
            "acquisition": "ucb",
            "acquisition_beta": float(getattr(acq, "beta", 2.0)),
        }
    if isinstance(acq, DrawAcquisitionConfig):
        return {"acquisition": "thompson"}
    if isinstance(acq, RandomAcquisitionConfig):
        return {"acquisition": "random"}
    if isinstance(acq, ParetoAcquisitionConfig):
        return {"acquisition": "pareto"}
    return {}


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
    out: dict[str, Any] = {}
    if candidates.num_candidates is None and candidates.num_candidates_per_arm is None:
        out["num_candidates_factor"] = _DEFAULT_NUM_CANDIDATES_FACTOR
        out["max_candidates"] = _DEFAULT_MAX_CANDIDATES
    elif candidates.num_candidates is not None:
        n = int(candidates.num_candidates)
        out["num_candidates_factor"] = 1.0
        out["min_candidates"] = n
        if candidates.num_candidates_per_arm is None:
            out["max_candidates"] = n
    elif candidates.num_candidates_per_arm is not None:
        out["num_candidates_factor"] = _DEFAULT_NUM_CANDIDATES_FACTOR
        out["max_candidates"] = _DEFAULT_MAX_CANDIDATES
    if candidates.num_candidates_per_arm is not None:
        out["num_candidates_per_arm"] = int(candidates.num_candidates_per_arm)
    return out


def _candidates_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out.update(_candidate_rv_override(config))
    out.update(_candidate_count_override(config))
    return out


def _get_tr_params(tr: TurboTRConfig) -> tuple[float, float, float]:
    li = (
        tr.length_init
        if hasattr(tr, "length_init")
        else getattr(tr.length, "length_init", 0.8)
    )
    lm = (
        tr.length_min
        if hasattr(tr, "length_min")
        else getattr(tr.length, "length_min", 0.5**7)
    )
    lx = (
        tr.length_max
        if hasattr(tr, "length_max")
        else getattr(tr.length, "length_max", 1.6)
    )
    return float(li), float(lm), float(lx)


def _trust_region_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tr = getattr(config, "trust_region", None)
    if isinstance(tr, MorboTRConfig):
        out["trust_region"] = "morbo"
        out["num_metrics"] = int(tr.num_metrics)
        out["alpha"] = float(tr.alpha)
        li, lm, lx = _get_tr_params(tr)
        out["length_init"] = li
        out["length_min"] = lm
        out["length_max"] = lx
        out["rescalarize"] = tr.rescalarize.value
        if tr.noise_aware:
            out["noise_aware"] = True
        return out
    if not isinstance(tr, TurboTRConfig):
        return out
    li, lm, lx = _get_tr_params(tr)
    if li != 0.8:
        out["length_init"] = li
    if abs(lm - 0.5**7) > 1e-12:
        out["length_min"] = lm
    if lx != 1.6:
        out["length_max"] = lx
    if tr.noise_aware:
        out["noise_aware"] = True
    return out


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
        if surrogate.scale_x:
            overrides["scale_x"] = True
    return overrides if overrides else None


def is_rust_supported_config(config: OptimizerConfig) -> bool:
    if requires_python_optimizer_fallback(config):
        return False
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
