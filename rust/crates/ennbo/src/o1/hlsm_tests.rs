use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use super::lsm_tests::exact;
use super::O1Store;

fn append(store: &mut O1Store, x: &Array2<f64>, step: usize) {
    for s in (0..x.nrows()).step_by(step) {
        let e = (s + step).min(x.nrows());
        let y = Array2::from_shape_fn((e - s, 1), |(i, _)| x[[s + i, 0]]);
        store.append_rows(&x.slice(ndarray::s![s..e, ..]), &y.view(), None).unwrap();
    }
}

fn build(n: usize, d: usize, seed: u64, l0: usize) -> (TempDir, O1Store, Array2<f64>) {
    std::env::set_var("ENN_O1_KIND", "hlsm");
    std::env::set_var("ENN_E_L0", l0.to_string());
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    append(&mut store, &x, 250);
    (dir, store, x)
}

fn weighted_exact(x: &Array2<f64>, q: &[f64], w: &[f64], k: usize) -> Vec<i64> {
    let mut dd: Vec<(f64, i64)> = x
        .rows()
        .into_iter()
        .enumerate()
        .map(|(j, r)| (r.iter().zip(q).zip(w).map(|((a, b), c)| c * (a - b) * (a - b)).sum(), j as i64))
        .collect();
    dd.sort_by(|a, b| a.0.total_cmp(&b.0));
    dd.into_iter().take(k).map(|p| p.1).collect()
}

fn meta(dir: &TempDir) -> super::hlsm::EMeta {
    ennbo_bpann::bpann_read_json(&dir.path().join(super::hlsm::E_META)).unwrap().unwrap()
}

#[test]
fn hlsm_with_unbounded_window_is_exact_through_reopen_and_metric_change() {
    std::env::set_var("ENN_E_WIN", "1e12");
    let (n, d) = (5000, 6);
    let (dir, mut store, x) = build(n, d, 11, 300);
    store.set_probe_budget(1, usize::MAX).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let q = Array2::from_shape_fn((20, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), exact(&x, &q.row(i).to_vec(), 10), "query {i}");
    }
    let scale = [1.0, 2.0, 0.5, 1.0, 3.0, 1.0];
    store.set_metric_scale(&ndarray::Array1::from(scale.to_vec()), true).unwrap();
    let more = Array2::from_shape_fn((3000, d), |_| rng.gen::<f64>());
    append(&mut store, &more, 250);
    drop(store);
    let m = meta(&dir);
    assert!(m.stats.rebuilds >= 8, "every table re-keyed: {:?}", m.stats);
    for f in &m.frames {
        assert_eq!(f.as_ref().unwrap().scale, scale.to_vec());
    }
    let shape = |t: usize| {
        let mut v: Vec<(u64, u64, u32)> = m.runs.iter().filter(|r| r.table == t).map(|r| (r.lo, r.hi, r.level)).collect();
        v.sort_unstable();
        v
    };
    assert!((1..m.frames.len()).all(|t| shape(t) == shape(0)), "tables keep one run structure after re-keying");
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    store.set_probe_budget(1, usize::MAX).unwrap();
    let all = ndarray::concatenate![ndarray::Axis(0), x, more];
    let w: Vec<f64> = scale.iter().map(|s| 1.0 / (s * s)).collect();
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), weighted_exact(&all, &q.row(i).to_vec(), &w, 10), "query {i} after metric change");
    }
    assert!(store.occupancy().unwrap().splits > 0);
}

#[test]
fn hlsm_default_window_finds_most_neighbours() {
    // One merged run of 12000 entries, about 5x the default 32 KiB window (2340 entries at d=8), so the window is bounded.
    let (n, d) = (12000, 8);
    let (_dir, mut store, x) = build(n, d, 5, 750);
    store.set_probe_budget(1, 100).unwrap();
    let mut rng = StdRng::seed_from_u64(9);
    let q = Array2::from_shape_fn((30, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    let mut hit = 0;
    for i in 0..q.nrows() {
        let want = exact(&x, &q.row(i).to_vec(), 10);
        hit += idx.row(i).iter().filter(|j| want.contains(j)).count();
    }
    let recall = hit as f64 / (10 * q.nrows()) as f64;
    assert!(recall > 0.9, "recall@10 = {recall}");
}


#[test]
fn hlsm_per_run_frames_resolve_a_converging_trajectory() {
    std::env::set_var("ENN_O1_KIND", "hlsm");
    std::env::set_var("ENN_E_L0", "300");
    let (n, d) = (12000, 6);
    let mut rng = StdRng::seed_from_u64(21);
    let x = Array2::from_shape_fn((n, d), |(i, j)| 0.5 + (0.3 + 0.03 * rng.gen::<f64>() + 0.01 * j as f64) * (-(i as f64) / 400.0).exp());
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    append(&mut store, &x, 250);
    store.set_probe_budget(1, 100).unwrap();
    let picks: Vec<usize> = (0..30).map(|i| 3000 + i * 290).collect();
    let q = Array2::from_shape_fn((picks.len(), d), |(i, j)| x[[picks[i], j]]);
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    let mut hit = 0;
    for i in 0..q.nrows() {
        let want = exact(&x, &q.row(i).to_vec(), 10);
        hit += idx.row(i).iter().filter(|j| want.contains(j)).count();
    }
    let recall = hit as f64 / (10 * q.nrows()) as f64;
    assert!(recall > 0.8, "recall@10 on a converging trajectory = {recall}");
}
