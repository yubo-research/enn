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
from .config.num_candidates_fn import default_num_candidates
from .config.optimizer_config import OptimizerConfig
from .config.surrogate import ENNSurrogateConfig, NoSurrogateConfig
from .config.trust_region import NoTRConfig, TurboTRConfig

_DEFAULT_NUM_CANDIDATES_FACTOR = 100.0
_DEFAULT_MAX_CANDIDATES = 5000


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


def _can_use_rust_num_candidates(config: OptimizerConfig) -> bool:
    candidates = getattr(config, "candidates", None)
    if not isinstance(candidates, CandidateGenConfig):
        return True
    fn = getattr(candidates, "num_candidates", None)
    if not callable(fn):
        return True
    if fn is default_num_candidates:
        return True
    if fn(num_dim=2, num_arms=1) == fn(num_dim=10, num_arms=1):
        return False
    return False


def _candidates_to_override(config: OptimizerConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rv = getattr(config, "candidate_rv", None)
    if rv is CandidateRV.SOBOL:
        out["candidate_rv"] = "sobol"
    elif rv is CandidateRV.UNIFORM:
        out["candidate_rv"] = "uniform"
    elif rv is CandidateRV.RAASP:
        out["candidate_rv"] = "raasp"
    candidates = getattr(config, "candidates", None)
    if isinstance(candidates, CandidateGenConfig):
        fn = getattr(candidates, "num_candidates", None)
        if fn is default_num_candidates:
            out["num_candidates_factor"] = _DEFAULT_NUM_CANDIDATES_FACTOR
            out["max_candidates"] = _DEFAULT_MAX_CANDIDATES
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
    if not isinstance(tr, TurboTRConfig):
        return out
    li, lm, lx = _get_tr_params(tr)
    if li != 0.8:
        out["length_init"] = li
    if abs(lm - 0.5**7) > 1e-12:
        out["length_min"] = lm
    if lx != 1.6:
        out["length_max"] = lx
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
    trailing_obs = getattr(config, "trailing_obs", None)
    if trailing_obs is not None:
        overrides["trailing_obs"] = int(trailing_obs)
    return overrides if overrides else None


def is_rust_supported_config(config: OptimizerConfig) -> bool:
    if not _can_use_rust_num_candidates(config):
        return False
    if isinstance(config.surrogate, ENNSurrogateConfig):
        return config.surrogate.k is not None
    if isinstance(config.surrogate, NoSurrogateConfig):
        return True
    return False


def _is_lhd_only_config(config: OptimizerConfig) -> bool:
    return (
        isinstance(config.trust_region, NoTRConfig)
        and isinstance(config.init.init_strategy, LHDOnlyInit)
        and isinstance(config.surrogate, NoSurrogateConfig)
    )
