//! Acceptance #4: TuRBO tell/ask incumbents do not depend on the work_dir location.

use ennbo::candidates::CandidateRV;
use ennbo::config::{turbo_enn_config, CandidateConfig, OptimizerConfig, SurrogateConfig};
use ennbo::optimizer::Optimizer;
use ennbo::strategy::Strategy;
use ennbo::{ENNError, InitStrategy};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use tempfile::TempDir;

type TellSchedule = (Vec<Array2<f64>>, Vec<Array2<f64>>);
type IncumbentReplay = (
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
    usize,
);

fn turbo_test_config(work_dir: &Path) -> OptimizerConfig {
    let mut cfg = turbo_enn_config(ennbo::EnnLayout::new(work_dir, false).unwrap());
    if let SurrogateConfig::ENN(ref mut enn) = cfg.surrogate {
        enn.k = ennbo::NeighborCount::new(3).unwrap();
        enn.fit_samples = ennbo::FitSamples::draw(3, 5).unwrap();
    }
    cfg.candidates = CandidateConfig {
        min_candidates: 20,
        max_candidates: 20,
        num_candidates_per_dim: 0,
        num_candidates_per_arm: 0,
        candidate_rv: CandidateRV::Uniform,
        raasp_fast: false,
    };
    cfg
}

fn synthetic_y(x: &ArrayView2<f64>, rng: &mut StdRng) -> Array2<f64> {
    let n = x.nrows();
    let mut y = Array2::zeros((n, 1));
    for i in 0..n {
        let sq = x.row(i).iter().map(|v| v * v).sum::<f64>();
        y[[i, 0]] = sq + 0.01 * rng.gen::<f64>();
    }
    y
}

fn record_tell_schedule(
    config: OptimizerConfig,
    seed: u64,
    num_init: usize,
    num_arms: usize,
    num_rounds: usize,
) -> Result<TellSchedule, ENNError> {
    let bounds = ndarray::array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(seed);
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    let mut opt = Optimizer::new_with_strategy(bounds, config, strategy, &mut rng)?;

    let mut xs = Vec::new();
    let mut ys = Vec::new();

    while opt.init_progress().is_some() {
        let x = opt.ask(num_arms)?;
        let y = synthetic_y(&x.view(), &mut rng);
        opt.tell(&x.view(), &y.view(), None)?;
        xs.push(x);
        ys.push(y);
    }

    for _ in 0..num_rounds {
        let x = opt.ask(num_arms)?;
        let y = synthetic_y(&x.view(), &mut rng);
        opt.tell(&x.view(), &y.view(), None)?;
        xs.push(x);
        ys.push(y);
    }

    Ok((xs, ys))
}

fn replay_tells(
    config: OptimizerConfig,
    seed: u64,
    num_init: usize,
    xs: &[Array2<f64>],
    ys: &[Array2<f64>],
) -> Result<IncumbentReplay, ENNError> {
    let bounds = ndarray::array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(seed);
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    let mut opt = Optimizer::new_with_strategy(bounds, config, strategy, &mut rng)?;

    for (x, y) in xs.iter().zip(ys.iter()) {
        opt.tell(&x.view(), &y.view(), None)?;
    }

    let obs_count = opt.obs_count();
    let x_inc = opt.incumbent_x();
    let y_inc = opt.incumbent_y_scalar().map(|y| y.to_owned());
    Ok((x_inc, y_inc, obs_count))
}

#[test]
fn turbo_incumbent_does_not_depend_on_work_dir_location() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().join("disk");
    let seed = 42_u64;
    let num_init = 8;
    let num_arms = 2;
    let num_rounds = 20;

    let (xs, ys) = record_tell_schedule(
        turbo_test_config(&dir.path().join("recording")),
        seed,
        num_init,
        num_arms,
        num_rounds,
    )
    .unwrap();

    let (x_other, y_other, n_other) = replay_tells(
        turbo_test_config(&dir.path().join("other")),
        seed,
        num_init,
        &xs,
        &ys,
    )
    .unwrap();
    let (x_disk, y_disk, n_disk) =
        replay_tells(turbo_test_config(&work_dir), seed, num_init, &xs, &ys).unwrap();

    assert_eq!(n_other, n_disk);
    assert!(n_disk >= num_init + num_rounds * num_arms);

    let x_other = x_other.expect("other incumbent x");
    let y_other = y_other.expect("other incumbent y");
    let x_disk = x_disk.expect("disk incumbent x");
    let y_disk = y_disk.expect("disk incumbent y");

    for i in 0..x_other.len() {
        assert!(
            (x_other[i] - x_disk[i]).abs() < 1e-6,
            "incumbent x[{i}] other={} disk={}",
            x_other[i],
            x_disk[i]
        );
    }
    assert!(
        (y_other[0] - y_disk[0]).abs() < 1e-6,
        "incumbent y other={} disk={}",
        y_other[0],
        y_disk[0]
    );

    let train_x = work_dir.join("train_x.bin");
    assert!(train_x.exists(), "train_x.bin should exist under work_dir");
    let expected_bytes = n_disk as u64 * 4 * 8;
    let actual_bytes = train_x.metadata().expect("stat train_x").len();
    assert!(
        actual_bytes >= expected_bytes,
        "train_x.bin size {actual_bytes} < expected {expected_bytes}"
    );
}
