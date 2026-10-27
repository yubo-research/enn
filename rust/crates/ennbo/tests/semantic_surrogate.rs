//! Semantic features affect only the surrogate, preserving observation coordinates.
use ennbo::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate};
use ennbo::{FitSamples, EnnLayout};
use ndarray::array;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn semantic_fit_append_prediction_and_samples_match_explicit_features() {
    check_feature_mode(false);
    check_feature_mode(true);
}

fn check_feature_mode(semantic_only: bool) {
    let projection = array![[0.6], [-0.6]];
    let x = array![[0.0, 0.1], [0.2, 0.8], [0.7, 0.3], [1.0, 0.9]];
    let y = array![[0.3], [0.5], [-0.2], [0.7]];
    let features = ennbo::semantic::features(&x.view(), Some(&projection), semantic_only).unwrap();
    let config = ENNSurrogateConfig {
        k: ennbo::NeighborCount::new(2).unwrap(),
        fit_samples: FitSamples::Frozen,
        semantic_projection: Some(Box::new(projection.clone())),
        semantic_only,
        ..Default::default()
    };
    config.validate(2).unwrap();
    assert!(config.validate(3).is_err());
    let mut actual = ENNSurrogate::new(config);
    let mut manual = ENNSurrogate::new(ENNSurrogateConfig {
        k: ennbo::NeighborCount::new(2).unwrap(),
        fit_samples: FitSamples::Frozen,
        ..Default::default()
    });
    let mut rng = StdRng::seed_from_u64(4);
    actual.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
    manual.fit(&features.view(), &y.view(), None, &mut rng).unwrap();
    let more = array![[0.25, 0.6]];
    let more_y = array![[0.4]];
    let more_z = ennbo::semantic::features(&more.view(), Some(&projection), semantic_only).unwrap();
    actual.fit_append(&more.view(), &more_y.view(), None, &mut rng).unwrap();
    manual.fit_append(&more_z.view(), &more_y.view(), None, &mut rng).unwrap();
    let a = actual.predict(&more.view()).unwrap();
    let b = manual.predict(&more_z.view()).unwrap();
    assert_eq!(a.mu, b.mu);
    assert_eq!(a.se, b.se);
    let a = actual.sample(&more.view(), 2, &mut StdRng::seed_from_u64(8)).unwrap();
    let b = manual.sample(&more_z.view(), 2, &mut StdRng::seed_from_u64(8)).unwrap();
    assert_eq!(a, b);
    assert_eq!(actual.observations_x().unwrap().unwrap().ncols(), 2);
    assert!((actual.observation_row_x(4).unwrap() - more.row(0)).iter().all(|v| v.abs() < 1e-15));
    assert!(actual.lengthscales().is_none());
    actual.clear_observations().unwrap();
    assert!(actual.observations_x().unwrap().is_none());
    actual.fit_append(&more.view(), &more_y.view(), None, &mut rng).unwrap();
    assert!((actual.observations_x().unwrap().unwrap() - more).iter().all(|v| v.abs() < 1e-15));
}

#[test]
fn semantic_config_accepts_adaptive_scaling_and_rejects_nonfinite_projection() {
    let mut config = ENNSurrogateConfig {
        semantic_projection: Some(Box::new(array![[1.0], [-1.0]])),
        layout: EnnLayout::flat(true),
        ..Default::default()
    };
    assert!(config.validate(2).is_ok());
    config.layout = EnnLayout::flat(false);
    config.semantic_projection = Some(Box::new(array![[f64::NAN], [0.0]]));
    assert!(config.validate(2).is_err());
}
