//! Acceptance #4: TuRBO tell/ask with disk-backed ENN matches in-memory incumbent.

use ennbo::backend::EnnStorage;
use ennbo::candidates::CandidateRV;
use ennbo::config::{CandidateConfig, OptimizerConfig, SurrogateConfig, turbo_enn_config};
use ennbo::index::IndexDriver;
use ennbo::optimizer::Optimizer;
use ennbo::strategy::Strategy;
use ennbo::surrogate::ENNSurrogateConfig;
use ennbo::{InitStrategy, ENNError};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use tempfile::TempDir;

type TellSchedule = (Vec<Array2<f64>>, Vec<Array2<f64>>);
type IncumbentReplay = (
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
    usize,
);

fn turbo_test_config(storage: EnnStorage, work_dir: Option<PathBuf>) -> OptimizerConfig {
    let mut cfg = turbo_enn_config();
    if let SurrogateConfig::ENN(ref mut enn) = cfg.surrogate {
        let index_driver = match storage {
            EnnStorage::Disk => IndexDriver::HNSWDisk,
            EnnStorage::InMemory => IndexDriver::Exact,
        };
        *enn = ENNSurrogateConfig {
            k: 3,
            scale_x: false,
            num_fit_candidates: 5,
            num_fit_samples: 3,
            infer_aleatoric_variance: true,
            index_driver,
            storage,
            work_dir,
        };
    }
    cfg.candidates = CandidateConfig {
        num_candidates_factor: 1.0,
        min_candidates: 20,
        max_candidates: Some(20),
        num_candidates_per_arm: None,
        candidate_rv: CandidateRV::Uniform,
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
        let x = opt.ask(num_arms, &mut rng)?;
        let y = synthetic_y(&x.view(), &mut rng);
        opt.tell(&x.view(), &y.view(), &mut rng)?;
        xs.push(x);
        ys.push(y);
    }

    for _ in 0..num_rounds {
        let x = opt.ask(num_arms, &mut rng)?;
        let y = synthetic_y(&x.view(), &mut rng);
        opt.tell(&x.view(), &y.view(), &mut rng)?;
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
        opt.tell(&x.view(), &y.view(), &mut rng)?;
    }

    let obs_count = opt.obs_count();
    let x_inc = opt.incumbent_x_unit().map(|x| x.to_owned());
    let y_inc = opt.incumbent_y_scalar().map(|y| y.to_owned());
    Ok((x_inc, y_inc, obs_count))
}

#[test]
fn turbo_disk_backend_matches_in_memory_incumbent() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().to_path_buf();
    let seed = 42_u64;
    let num_init = 8;
    let num_arms = 2;
    let num_rounds = 20;

    let mem_cfg = turbo_test_config(EnnStorage::InMemory, None);
    let disk_cfg = turbo_test_config(EnnStorage::Disk, Some(work_dir.clone()));

    let (xs, ys) =
        record_tell_schedule(mem_cfg.clone(), seed, num_init, num_arms, num_rounds).unwrap();

    let (x_mem, y_mem, n_mem) =
        replay_tells(mem_cfg, seed, num_init, &xs, &ys).unwrap();
    let (x_disk, y_disk, n_disk) =
        replay_tells(disk_cfg, seed, num_init, &xs, &ys).unwrap();

    assert_eq!(n_mem, n_disk);
    assert!(n_disk >= num_init + num_rounds * num_arms);

    let x_mem = x_mem.expect("memory incumbent x");
    let y_mem = y_mem.expect("memory incumbent y");
    let x_disk = x_disk.expect("disk incumbent x");
    let y_disk = y_disk.expect("disk incumbent y");

    for i in 0..x_mem.len() {
        assert!(
            (x_mem[i] - x_disk[i]).abs() < 1e-6,
            "incumbent x[{i}] mem={} disk={}",
            x_mem[i],
            x_disk[i]
        );
    }
    assert!(
        (y_mem[0] - y_disk[0]).abs() < 1e-6,
        "incumbent y mem={} disk={}",
        y_mem[0],
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
