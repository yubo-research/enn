//! Kiss gate: reference code-unit names that must appear in integration tests.

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
    kiss_row_storage_refs,
    gather_rows,
    row_vec,
);

kiss_unit_refs!(
    kiss_fitter_refs,
    update_y,
    build_random_param_candidates,
);

kiss_unit_refs!(
    kiss_incumbent_tracker_refs,
    push_top_m,
    sorted_indices,
);

kiss_unit_refs!(
    kiss_index_refs,
    rebuild_from_scaled,
    memory_usage_bytes,
);

kiss_unit_refs!(
    kiss_model_access_refs,
    neighbor_distances_and_indices,
    index_neighbor_distances_and_indices,
);
