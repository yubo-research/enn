use ennbo::morbo_trust_region::{MorboTRSettings, MorboTrustRegion, Rescalarize};
use ennbo::trust_region::TRLengthConfig;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::str::FromStr;

#[test]
fn morbo_update_paths_and_incumbent_only() {
    assert_eq!(
        Rescalarize::from_str("on_restart").unwrap(),
        Rescalarize::OnRestart
    );
    assert_eq!(
        Rescalarize::from_str("on_propose").unwrap(),
        Rescalarize::OnPropose
    );
    assert!(Rescalarize::from_str("nope").is_err());

    let settings = MorboTRSettings {
        num_metrics: 2,
        alpha: 0.1,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: true,
    };
    let mut rng = StdRng::seed_from_u64(3);
    let mut tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
    assert_eq!(tr.num_metrics(), 2);
    assert!(tr.noise_aware());
    assert_eq!(tr.rescalarize(), Rescalarize::OnRestart);
    assert!(tr.y_min().is_none());
    assert!(tr.y_max().is_none());
    tr.set_num_arms(2);

    let empty = ndarray::Array2::<f64>::zeros((0, 2));
    let inc0 = array![0.0, 0.0];
    tr.update(&empty.view(), &inc0.view()).unwrap();

    let y1 = array![[0.0, 1.0], [1.0, 0.0]];
    let inc1 = array![0.0, 1.0];
    tr.update(&y1.view(), &inc1.view()).unwrap();
    let _ = tr.scalarize(&y1.view(), true).unwrap();
    let _ = tr.scalarize(&y1.view(), false).unwrap();
    assert!(tr.y_min().is_some());
    assert!(tr.y_max().is_some());

    let y2 = array![[0.0, 1.0], [1.0, 0.0], [0.9, 0.9]];
    let inc2 = array![0.9, 0.9];
    tr.update(&y2.view(), &inc2.view()).unwrap();

    tr.update_incumbent_only(&inc2.view(), 3).unwrap();
    tr.rescalarize_incumbent_under_weights(3).unwrap();

    let bad = array![1.0];
    assert!(tr.update_incumbent_only(&bad.view(), 3).is_err());
    assert!(tr.update(&array![[0.0, 1.0]].view(), &inc2.view()).is_err());
    tr.update_incumbent_only(&inc2.view(), 0).unwrap();

    let settings2 = MorboTRSettings {
        num_metrics: 2,
        alpha: 0.1,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnPropose,
        noise_aware: false,
    };
    let mut tr2 = MorboTrustRegion::new(2, settings2, &mut rng).unwrap();
    tr2.set_num_arms(2);
    let y3 = array![[0.2, 0.3], [0.4, 0.5]];
    let inc3 = array![0.4, 0.5];
    tr2.update(&y3.view(), &inc3.view()).unwrap();
    let center = array![0.5, 0.5];
    let ls = array![1.0, 1.0];
    let _ = tr2.compute_bounds_1d(&center.view(), Some(&ls.view()));
    assert!(!tr2.needs_restart());
    tr2.restart(None);
    tr2.restart(Some(&mut rng));
    assert!(tr2.update(&y3.view(), &array![0.0].view()).is_err());
    assert!(tr2.update(&array![[0.0]].view(), &inc3.view()).is_err());
}
