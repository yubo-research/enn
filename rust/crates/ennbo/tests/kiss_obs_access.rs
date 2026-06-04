//! Kiss static coverage for optimizer observation access helpers.

use ennbo::config::turbo_zero_config;
use ennbo::optimizer::obs_access::{build_obs_array2, ObsAccess};
use ennbo::optimizer::Optimizer;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

const OBS_ACCESS_SRC: &str = include_str!("../src/optimizer/obs_access.rs");

#[test]
fn kiss_obs_access_helpers_called() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let mut opt = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();
    let access: ObsAccess<'_> = opt.obs_access();
    assert!(access.observations_empty());
    let x = array![[0.1, 0.2]];
    let y = array![[0.5, 1.5]];
    opt.add_observations(&x.view(), &y.view()).unwrap();
    let access = opt.obs_access();
    assert!(!access.observations_empty());
    let _ = access.obs_row_x(0).unwrap();
    let _ = access.obs_row_y(0).unwrap();
    let arr = build_obs_array2(&[array![1.0, 2.0]]);
    assert_eq!(arr.shape(), &[1, 2]);
}

#[test]
fn kiss_obs_access_names_in_source() {
    for name in [
        "ObsAccess",
        "observations_empty",
        "obs_row_x",
        "obs_row_y",
        "build_obs_array2",
    ] {
        assert!(
            OBS_ACCESS_SRC.contains(name),
            "missing {name} in obs_access.rs"
        );
    }
}
