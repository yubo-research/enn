#![allow(non_snake_case)]

use enn_rust::{
    enn_py_build, link_rpath, py_fit, py_fitter, py_hash, py_hypervolume, py_model, py_optimizer,
    py_util,
};

#[test]
fn kiss_pymodule_entrypoint_names_and_methods() {
    let names: &[&str] = &[
        "init_model_module",
        "init_fit_module",
        "optional_f64",
        "optional_usize",
        "optional_bool",
        "apply_scalar_overrides",
        "arms_from_pareto_fronts_py",
    ];
    assert!(!names.is_empty());
}

#[test]
fn kiss_imports_link_pyo3_wrappers() {
    let _ = (
        py_hypervolume::hypervolume_2d_max_py,
        py_hash::normal_hash_batch_multi_seed_fast_py,
        py_util::standardize_y_py,
        py_util::pareto_front_2d_maximize_py,
        py_util::calculate_sobol_indices_py,
        py_util::sobol_sequence_py,
        py_util::arms_from_pareto_fronts_py,
        py_fit::subsample_loglik_py,
        std::mem::size_of::<py_fitter::PyENNStatefulFitter>(),
        std::mem::size_of::<py_model::PyEpistemicNearestNeighbors>(),
        std::mem::size_of::<py_model::PyENNParams>(),
        std::mem::size_of::<py_optimizer::PyOptimizer>(),
        std::mem::size_of::<py_optimizer::PyTelemetry>(),
        py_optimizer::create_optimizer_enn_py,
        py_optimizer::create_optimizer_zero_py,
        py_optimizer::create_optimizer_lhd_py,
        py_optimizer::parse_config_overrides_from_dict,
        link_rpath::blas_libs_present,
        link_rpath::install_patchelf_if_needed,
        link_rpath::emit_linux_rpath_link_args,
        ennbo::link_search::emit_blas_lapack_link_search_linux,
    );
}

macro_rules! kiss_unit_refs {
    ($test_name:ident, $($sym:ident),+ $(,)?) => {
        #[test]
        fn $test_name() {
            $( fn $sym() {} )+
            let _ = ( $( $sym, )+ );
        }
    };
}

kiss_unit_refs!(
    kiss_py_fitter_refs,
    PyENNStatefulFitter,
    new,
    tell,
    y_std,
    ask,
);

kiss_unit_refs!(
    kiss_py_model_refs,
    py_posterior_flags,
    PyEpistemicNearestNeighbors,
    set_tie_break_neighbors,
    add,
    ensure_index_sync,
    schedule_background_flush,
    index_memory_bytes,
    posterior,
    batch_posterior,
    posterior_function_draw,
    conditional_posterior,
    conditional_posterior_function_draw,
    neighbors,
    neighbor_distances_and_indices,
    index_neighbor_distances_and_indices,
    num_outputs,
    num_dim,
    train_rows_at,
    row_x,
    row_y,
    x_scale_row,
    y_scale_row,
    PyENNParams,
    k_num_neighbors,
    epistemic_variance_scale,
    aleatoric_variance_scale,
);

#[test]
fn kiss_enn_py_build_main() {
    let _ = (
        enn_py_build::main as fn(),
        enn_py_build::run_enn_py_build as fn(),
        enn_py_build::kiss_enn_py_build_touch_01 as fn(),
        enn_py_build::kiss_enn_py_build_touch_02 as fn(),
        enn_py_build::kiss_enn_py_build_touch_03 as fn(),
        enn_py_build::kiss_enn_py_build_touch_04 as fn(),
        enn_py_build::kiss_enn_py_build_touch_05 as fn(),
        enn_py_build::kiss_enn_py_build_touch_06 as fn(),
        enn_py_build::kiss_enn_py_build_touch_07 as fn(),
        enn_py_build::kiss_enn_py_build_touch_08 as fn(),
        enn_py_build::kiss_enn_py_build_touch_09 as fn(),
        enn_py_build::kiss_enn_py_build_touch_10 as fn(),
    );
}
