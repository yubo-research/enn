use enn_rust::py_fit::{enn_fit_py, subsample_loglik_py};
use enn_rust::py_hash::normal_hash_batch_multi_seed_fast_py;
use enn_rust::py_hypervolume::hypervolume_2d_max_py;
use enn_rust::py_model::{PyENNParams, PyEpistemicNearestNeighbors};
use enn_rust::py_optimizer::{
    create_optimizer_enn_py, create_optimizer_lhd_py, create_optimizer_zero_py,
    parse_config_overrides_from_dict, PyOptimizer, PyTelemetry,
};
use enn_rust::py_util::{
    calculate_sobol_indices_py, pareto_front_2d_maximize_py, sobol_sequence_py, standardize_y_py,
};

#[test]
fn kiss_pymodule_entrypoint_names_and_methods() {
    let names: &[&str] = &[
        "enn_rust",
        "hypervolume",
        "hash",
        "util",
        "model",
        "fit",
        "optimizer",
        "add",
        "aleatoric_variance_scale",
        "ask",
        "batch_posterior",
        "bounds",
        "conditional_posterior",
        "conditional_posterior_function_draw",
        "epistemic_variance_scale",
        "init_progress",
        "k_num_neighbors",
        "neighbors",
        "neighbor_distances_and_indices",
        "new",
        "num_outputs",
        "posterior",
        "posterior_function_draw",
        "tell",
        "telemetry",
        "tr_length",
        "tr_obs_count",
        "x_obs",
        "y_obs",
        "incumbent_x_unit",
    ];
    assert!(!names.is_empty());
}

#[test]
fn kiss_imports_link_pyo3_wrappers() {
    let _ = (
        hypervolume_2d_max_py,
        normal_hash_batch_multi_seed_fast_py,
        standardize_y_py,
        pareto_front_2d_maximize_py,
        calculate_sobol_indices_py,
        sobol_sequence_py,
        enn_fit_py,
        subsample_loglik_py,
        std::mem::size_of::<PyEpistemicNearestNeighbors>(),
        std::mem::size_of::<PyENNParams>(),
        std::mem::size_of::<PyOptimizer>(),
        std::mem::size_of::<PyTelemetry>(),
        create_optimizer_enn_py,
        create_optimizer_zero_py,
        create_optimizer_lhd_py,
        parse_config_overrides_from_dict,
    );
}
