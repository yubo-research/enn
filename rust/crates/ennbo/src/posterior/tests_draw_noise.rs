use super::*;
use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
use ndarray::array;

#[test]
fn test_draw_from_internals_observation_noise_adds_independent_aleatoric() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5], [0.25, 0.75]];
    let seeds = vec![7i64];

    let flags_off = PosteriorFlags::new();
    let flags_on = PosteriorFlags {
        exclude_nearest: false,
        observation_noise: true,
    };

    let off = compute_posterior_internals(&model, &query.view(), &params, &flags_off).unwrap();
    let on = compute_posterior_internals(&model, &query.view(), &params, &flags_on).unwrap();
    assert!(on.se_ale[[0, 0]] > 0.0);

    let draw_off = draw_from_internals(&model, &off, &seeds).unwrap();
    let draw_on = draw_from_internals(&model, &on, &seeds).unwrap();
    assert!((draw_on[[0, 0, 0]] - draw_off[[0, 0, 0]]).abs() > 1e-12);

    // Epistemic field matches observation_noise=false when aleatoric is stripped.
    let mut on_epi_only = on.clone();
    on_epi_only.se_ale.fill(0.0);
    let draw_epi = draw_from_internals(&model, &on_epi_only, &seeds).unwrap();
    assert!((draw_epi[[0, 0, 0]] - draw_off[[0, 0, 0]]).abs() < 1e-12);
}
