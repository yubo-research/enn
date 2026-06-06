//! Optimizer ask/tell integration with disk HNSW background flush.

use ennbo::backend::{DiskEnnBackend, EnnStorage};
use ennbo::disk_hnsw::flush::wait_for_background_flush;
use ennbo::candidates::CandidateRV;
use ennbo::config::{CandidateConfig, InitStrategy, OptimizerConfig, SurrogateConfig, turbo_enn_config};
use ennbo::disk_hnsw::DiskHnswEnnBackend;
use ennbo::index::IndexDriver;
use ennbo::optimizer::Optimizer;
use ennbo::strategy::Strategy;
use ennbo::surrogate::ENNSurrogateConfig;
use ndarray::{array, Array1};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

fn disk_optimizer_config(work_dir: PathBuf) -> OptimizerConfig {
    let mut cfg = turbo_enn_config();
    if let SurrogateConfig::ENN(ref mut enn) = cfg.surrogate {
        *enn = ENNSurrogateConfig {
            k: 3,
            scale_x: false,
            num_fit_candidates: 5,
            num_fit_samples: 3,
            infer_aleatoric_variance: true,
            index_driver: IndexDriver::HNSWDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(work_dir),
        };
    }
    cfg.candidates = CandidateConfig {
        num_candidates_factor: 1.0,
        min_candidates: 12,
        max_candidates: Some(12),
        num_candidates_per_arm: None,
        candidate_rv: CandidateRV::Uniform,
    };
    cfg
}

fn synthetic_y(x: &ndarray::ArrayView2<f64>, rng: &mut StdRng) -> ndarray::Array2<f64> {
    let n = x.nrows();
    let mut y = ndarray::Array2::zeros((n, 1));
    for i in 0..n {
        let sq = x.row(i).iter().map(|v| v * v).sum::<f64>();
        y[[i, 0]] = sq + 0.01 * rng.gen::<f64>();
    }
    y
}

fn run_init(opt: &mut Optimizer, rng: &mut StdRng) {
    while opt.init_progress().is_some() {
        let x = opt.ask(1, rng).unwrap();
        let y = synthetic_y(&x.view(), rng);
        opt.tell(&x.view(), &y.view(), rng).unwrap();
    }
}

fn fitted_enn_model(opt: &Optimizer) -> &ennbo::model::EpistemicNearestNeighbors {
    use ennbo::surrogate::{ENNSurrogate, Surrogate};
    let s = opt.surrogate().expect("fitted surrogate");
    let ptr = s as *const dyn Surrogate as *const ENNSurrogate;
    // SAFETY: disk-flush integration tests always use ENNSurrogate as the sole surrogate impl.
    unsafe { (*ptr).model().expect("enn model") }
}

fn disk_hnsw_backend(opt: &Optimizer) -> Arc<Mutex<DiskEnnBackend>> {
    fitted_enn_model(opt)
        .disk_backend_arc()
        .expect("disk backend")
}

fn with_hnsw<T>(arc: &Arc<Mutex<DiskEnnBackend>>, f: impl FnOnce(&DiskHnswEnnBackend) -> T) -> T {
    let guard = arc.lock().expect("disk lock");
    match &*guard {
        DiskEnnBackend::Hnsw(b) => f(b),
    }
}

fn with_hnsw_mut<T>(
    arc: &Arc<Mutex<DiskEnnBackend>>,
    f: impl FnOnce(&mut DiskHnswEnnBackend) -> T,
) -> T {
    let mut guard = arc.lock().expect("disk lock");
    match &mut *guard {
        DiskEnnBackend::Hnsw(b) => f(b),
    }
}

fn prime_pending_flush(opt: &mut Optimizer, rng: &mut StdRng) -> Arc<Mutex<DiskEnnBackend>> {
    let arc = disk_hnsw_backend(opt);
    with_hnsw_mut(&arc, |b| {
        b.set_pending_flush_threshold(3);
        b.set_defer_append_indexing(true);
        b.ensure_index_sync(false, &Array1::ones(b.num_dim())).unwrap();
    });
    for _ in 0..3 {
        let x = opt.ask(1, rng).unwrap();
        let y = synthetic_y(&x.view(), rng);
        opt.tell(&x.view(), &y.view(), rng).unwrap();
    }
    with_hnsw(&arc, |b| assert!(b.pending_rows() >= 3));
    arc
}

#[test]
fn optimizer_ask_schedules_background_flush_disk_hnsw() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = disk_optimizer_config(dir.path().to_path_buf());
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 4);
    let mut opt = Optimizer::new_with_strategy(bounds, cfg, strategy, &mut rng).unwrap();

    run_init(&mut opt, &mut rng);
    let arc = prime_pending_flush(&mut opt, &mut rng);
    let _ = opt.ask(1, &mut rng).unwrap();

    with_hnsw(&arc, |b| {
        let len = b.len();
        let indexed = b.indexed_rows();
        assert!(indexed <= len);
    });
}

fn flush_arc_from(
    arc: &Arc<Mutex<DiskEnnBackend>>,
) -> Arc<Mutex<ennbo::disk_hnsw::flush::BackgroundFlushState>> {
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    b.flush_arc()
}

#[test]
fn optimizer_ask_does_not_block_on_flush() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = disk_optimizer_config(dir.path().to_path_buf());
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 4);
    let mut opt = Optimizer::new_with_strategy(bounds, cfg, strategy, &mut rng).unwrap();

    run_init(&mut opt, &mut rng);
    let arc = disk_hnsw_backend(&opt);
    with_hnsw_mut(&arc, |b| {
        b.set_pending_flush_threshold(3);
        b.set_defer_append_indexing(true);
        b.ensure_index_sync(false, &Array1::ones(b.num_dim())).unwrap();
    });
    let x = opt.ask(3, &mut rng).unwrap();
    let y = synthetic_y(&x.view(), &mut rng);
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();
    with_hnsw(&arc, |b| assert!(b.pending_rows() >= 3));
    with_hnsw_mut(&arc, |b| b.flush_test_barrier_hold(true));
    let indexed_before = with_hnsw(&arc, |b| b.indexed_rows());

    let _ = opt.ask(1, &mut rng).unwrap();

    with_hnsw(&arc, |b| assert_eq!(b.indexed_rows(), indexed_before));
    let flush = flush_arc_from(&arc);
    assert!(flush.lock().expect("flush lock").in_progress);
    flush.lock().expect("flush lock").barrier.set_hold(false);
    wait_for_background_flush(&flush).unwrap();
}

#[test]
fn optimizer_ask_second_call_while_flush_running() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = disk_optimizer_config(dir.path().to_path_buf());
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(17);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 4);
    let mut opt = Optimizer::new_with_strategy(bounds, cfg, strategy, &mut rng).unwrap();

    run_init(&mut opt, &mut rng);
    let arc = prime_pending_flush(&mut opt, &mut rng);

    let _ = opt.ask(1, &mut rng).unwrap();
    let _ = opt.ask(1, &mut rng).unwrap();

    with_hnsw(&arc, |b| b.wait_for_flush().unwrap());
}

#[test]
fn optimizer_tell_waits_for_background_flush() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = disk_optimizer_config(dir.path().to_path_buf());
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(11);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 4);
    let mut opt = Optimizer::new_with_strategy(bounds, cfg, strategy, &mut rng).unwrap();

    run_init(&mut opt, &mut rng);
    let arc = prime_pending_flush(&mut opt, &mut rng);
    let _ = opt.ask(1, &mut rng).unwrap();
    let len_before_tell = with_hnsw(&arc, |b| b.len());

    let x = array![[0.5, 0.5]];
    let y = array![[0.1]];
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();

    with_hnsw(&arc, |b| {
        assert_eq!(b.indexed_rows(), len_before_tell);
        assert_eq!(b.len(), len_before_tell + 1);
    });
}

#[test]
fn optimizer_tell_propagates_flush_error() {
    let dir = TempDir::new().expect("tempdir");
    let cfg = disk_optimizer_config(dir.path().to_path_buf());
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(19);
    let strategy = Strategy::hybrid(InitStrategy::LHD, 4);
    let mut opt = Optimizer::new_with_strategy(bounds, cfg, strategy, &mut rng).unwrap();

    run_init(&mut opt, &mut rng);
    let arc = prime_pending_flush(&mut opt, &mut rng);
    with_hnsw_mut(&arc, |b| b.inject_next_flush_failure());
    let _ = opt.ask(1, &mut rng).unwrap();

    let x = array![[0.5, 0.5]];
    let y = array![[0.1]];
    let err = opt.tell(&x.view(), &y.view(), &mut rng).unwrap_err();
    assert!(err.to_string().contains("injected background flush failure"));
}
