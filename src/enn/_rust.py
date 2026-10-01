from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from . import enn_rust as _ext
except ImportError as exc:
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
set_config_path = _ext.util.set_config_path
ensure_config_file = _ext.util.ensure_config_file
EpistemicNearestNeighbors = _ext.model.EpistemicNearestNeighbors
PyReservoir = _ext.model.PyReservoir
metric_snapshot = _ext.model.metric_snapshot
metric_set_weights = _ext.model.metric_set_weights
metric_configure = _ext.model.metric_configure
metric_tied = _ext.model.metric_tied
weight_drift = _ext.model.weight_drift
auto_uses_learned_metric = _ext.model.auto_uses_learned_metric
dependence_weights = _ext.model.dependence_weights
auto_weights = _ext.model.auto_weights
sobol_index = _ext.model.sobol_index
null_sd = _ext.model.null_sd
group_sobol_index = _ext.model.group_sobol_index
loo_loglik = _ext.model.loo_loglik
validate_tied_dims = _ext.model.validate_tied_dims
DEFAULT_REBUILD_DRIFT = _ext.model.DEFAULT_REBUILD_DRIFT
DRIFT_WEIGHT_FLOOR = _ext.model.DRIFT_WEIGHT_FLOOR
AUTO_MIN_HELDOUT_GAIN = _ext.model.AUTO_MIN_HELDOUT_GAIN
AUTO_RESERVOIR_CAPACITY = _ext.model.AUTO_RESERVOIR_CAPACITY
AUTO_K = _ext.model.AUTO_K
AUTO_REFIT_GROWTH = _ext.model.AUTO_REFIT_GROWTH
AUTO_RESCALE_TOL = _ext.model.AUTO_RESCALE_TOL
DEPENDENCE_FLOOR = _ext.model.DEPENDENCE_FLOOR
fit_model_affine = _ext.fit.fit_model_affine
normal_hash_batch_multi_seed = _ext.hash.normal_hash_batch_multi_seed
fit_affine = _ext.util.fit_affine
affine_map_mu = _ext.util.affine_map_mu
affine_map_draws = _ext.util.affine_map_draws
affine_apply = _ext.util.affine_apply
sample_normal = _ext.util.sample_normal
confidence_interval = _ext.util.confidence_interval
z_crit = _ext.util.z_crit
ackley_core = _ext.util.ackley_core
separable_unimodal = _ext.util.separable_unimodal
choose_indices = _ext.util.choose_indices
NumpyNormal = _ext.util.NumpyNormal
ENNNormal = _ext.util.ENNNormal
Ackley = _ext.util.Ackley
DoubleAckley = _ext.util.DoubleAckley
ENNParams = _ext.model.ENNParams
set_unscaled_dims = _ext.model.set_unscaled_dims
ENNStatefulFitter = _ext.fit.ENNStatefulFitter
subsample_loglik = _ext.fit.subsample_loglik
Optimizer = _ext.optimizer.Optimizer
create_optimizer_enn = _ext.optimizer.create_optimizer_enn
create_optimizer_zero = _ext.optimizer.create_optimizer_zero
create_optimizer_lhd = _ext.optimizer.create_optimizer_lhd
require_num_fit_samples = _ext.optimizer.require_num_fit_samples
validate_optimizer_rules = _ext.optimizer.validate_optimizer_rules


__all__ = [
    "hypervolume_2d_max",
    "normal_hash_batch_multi_seed_fast",
    "standardize_y",
    "pareto_front_2d_maximize",
    "calculate_sobol_indices",
    "sobol_sequence",
    "arms_from_pareto_fronts",
    "set_config_path",
    "ensure_config_file",
    "EpistemicNearestNeighbors",
    "ENNParams",
    "set_unscaled_dims",
    "ENNStatefulFitter",
    "subsample_loglik",
    "Optimizer",
    "create_optimizer_enn",
    "create_optimizer_zero",
    "create_optimizer_lhd",
]
