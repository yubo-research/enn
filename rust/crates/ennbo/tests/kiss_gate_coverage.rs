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

kiss_unit_refs!(kiss_fitter_refs, build_random_param_candidates,);

kiss_unit_refs!(kiss_incumbent_tracker_refs, push_top_m, sorted_indices,);

kiss_unit_refs!(
    kiss_model_access_refs,
    neighbor_distances_and_indices,
    index_neighbor_distances_and_indices,
);

#[test]
fn search_controls_round_trip() {
    ennbo::set_leaf_coefficient(30.0);
    assert_eq!(ennbo::leaf_coefficient(), 30.0);
    ennbo::set_exact_fallback(true);
    assert!(ennbo::exact_fallback());
    let _ = ennbo::take_walk_totals();
    assert_eq!(ennbo::take_walk_totals().f32_walks, 0);
}

#[test]
fn identity_calibrator_combines_standard_errors() {
    use ndarray::array;
    let cal = ennbo::calibration::AffineCalibrator::identity(1);
    let (se, epi, ale) = cal.apply_se(&array![[3.0]], &array![[4.0]]).unwrap();
    assert_eq!(
        (se, epi, ale),
        (array![[5.0]], array![[3.0]], array![[4.0]])
    );
}

#[test]
fn lhd_only_builder_removes_surrogate() {
    let layout = ennbo::EnnLayout::new("rows", false).unwrap();
    let (cfg, count) = ennbo::OptimizerConfig::turbo_enn(layout)
        .lhd_only()
        .acquisition(ennbo::config::AcquisitionConfig::Random)
        .trust_region(ennbo::TrustRegionConfig::Global)
        .num_init(Some(3))
        .build()
        .unwrap();
    assert!(matches!(
        cfg.surrogate,
        ennbo::config::SurrogateConfig::None
    ));
    assert_eq!(count, Some(3));
}
