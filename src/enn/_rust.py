from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch  # noqa: F401
except ImportError:
    pass

try:
    from . import enn_rust as _ext
except ImportError as exc:  # pragma: no cover - exercised when extension unavailable
    raise ImportError(
        "Rust extension submodule `enn.enn_rust` is not available"
    ) from exc


hypervolume_2d_max = _ext.hypervolume.hypervolume_2d_max
normal_hash_batch_multi_seed_fast = _ext.hash.normal_hash_batch_multi_seed_fast
standardize_y = _ext.util.standardize_y
pareto_front_2d_maximize = _ext.util.pareto_front_2d_maximize
calculate_sobol_indices = _ext.util.calculate_sobol_indices
sobol_sequence = _ext.util.sobol_sequence
arms_from_pareto_fronts = _ext.util.arms_from_pareto_fronts
EpistemicNearestNeighbors = _ext.model.EpistemicNearestNeighbors
ENNParams = _ext.model.ENNParams
enn_fit = _ext.fit.enn_fit
subsample_loglik = _ext.fit.subsample_loglik
Optimizer = _ext.optimizer.Optimizer
create_optimizer_enn = _ext.optimizer.create_optimizer_enn
create_optimizer_zero = _ext.optimizer.create_optimizer_zero
create_optimizer_lhd = _ext.optimizer.create_optimizer_lhd


__all__ = [
    "hypervolume_2d_max",
    "normal_hash_batch_multi_seed_fast",
    "standardize_y",
    "pareto_front_2d_maximize",
    "calculate_sobol_indices",
    "sobol_sequence",
    "arms_from_pareto_fronts",
    "EpistemicNearestNeighbors",
    "ENNParams",
    "enn_fit",
    "subsample_loglik",
    "Optimizer",
    "create_optimizer_enn",
    "create_optimizer_zero",
    "create_optimizer_lhd",
]
