//! Regression: Morbo + noise_aware must keep the mu row used to pick the incumbent.

use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{OptimizerConfig, SurrogateConfig};
use crate::morbo_trust_region::{MorboTRSettings, Rescalarize};
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::TRLengthConfig;
use crate::trust_region_config::TrustRegionConfig;
use crate::util::argmax_random_tie;

fn argmax_scores(scores: &ndarray::Array1<f64>) -> usize {
    argmax_random_tie(scores.as_slice().unwrap_or(&[]), &mut StdRng::seed_from_u64(0))
}

#[test]
fn morbo_noise_aware_incumbent_y_is_mu_row_used_for_selection() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(9001);
    let mut cfg = OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 3,
            num_fit_samples: 6,
            num_fit_candidates: 4,
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: true,
        }),
        ..OptimizerConfig::default()
    };
    cfg.noise_aware = true;

    let mut opt =
        Optimizer::new_with_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let x = array![
        [0.02, 0.02],
        [0.98, 0.02],
        [0.02, 0.98],
        [0.60, 0.60],
    ];
    let y = array![
        [2.0, 2.0],
        [50.0, 1.0],
        [1.0, 50.0],
        [3.0, 3.0],
    ];
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();

    let sur = opt.surrogate().expect("enn surrogate");
    let y_all = opt.y_obs().expect("y observations");
    let x_all = opt.x_obs().expect("x observations");
    let n = y_all.nrows();
    assert_eq!(n, 4);

    let pred = sur.predict(&x_all.view()).expect("predict");
    let mu = pred.mu;

    let scores_on_y = opt
        .trust_region()
        .morbo_scalarize(&y_all.view(), true)
        .expect("scalarize y");
    let scores_on_mu = opt
        .trust_region()
        .morbo_scalarize(&mu.view(), true)
        .expect("scalarize mu");

    let y_winner = argmax_scores(&scores_on_y);
    let mu_winner = argmax_scores(&scores_on_mu);

    assert_ne!(
        y_winner, mu_winner,
        "test setup must make mu and raw-y Morbo rankings diverge (y_winner={y_winner}, mu_winner={mu_winner})"
    );

    let y_inc = opt.incumbent_y_scalar().expect("incumbent y").to_owned();
    let expected_mu = mu.row(mu_winner).to_owned();
    let raw_y_at_mu_winner = y_all.row(mu_winner).to_owned();

    assert!(
        y_inc.iter()
            .zip(expected_mu.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6),
        "noise_aware Morbo must store posterior mu row {:?} used to select incumbent, got {:?}",
        expected_mu.as_slice().unwrap(),
        y_inc.as_slice().unwrap()
    );
    assert!(
        y_inc.iter()
            .zip(raw_y_at_mu_winner.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "stored incumbent must not be raw observed y {:?} when mu differs {:?}",
        raw_y_at_mu_winner.as_slice().unwrap(),
        expected_mu.as_slice().unwrap()
    );
}
