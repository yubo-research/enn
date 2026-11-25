use super::lsh::target_bits;
use super::search::O1_EXACT_ROWS;
use super::*;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn random_rows(n: usize, d: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let y = Array2::from_shape_fn((n, 1), |(i, _)| x[[i, 0]]);
    (x, y)
}

fn filled_store(dir: &TempDir, n: usize, d: usize) -> O1Store {
    let mut store = O1Store::open_or_create(dir.path().to_path_buf(), d, 1).unwrap();
    let (x, y) = random_rows(n, d, 7);
    for start in (0..n).step_by(500) {
        let end = (start + 500).min(n);
        let xs = x.slice(ndarray::s![start..end, ..]);
        let ys = y.slice(ndarray::s![start..end, ..]);
        store.append_rows(&xs, &ys, None).unwrap();
    }
    store
}

fn exact_top(store: &O1Store, q: &[f64], k: usize) -> Vec<i64> {
    let mut all: Vec<(f64, i64)> = (0..store.len())
        .map(|i| {
            let r = store.row_x(i).unwrap();
            let d: f64 = r.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum();
            (d, i as i64)
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all.into_iter().take(k).map(|p| p.1).collect()
}

#[test]
fn target_bits_keeps_mean_bucket_near_target() {
    let t = O1Params::default().bucket_target;
    assert_eq!(target_bits(0, t), 0);
    assert_eq!(target_bits(t, t), 0);
    assert_eq!(target_bits(t + 1, t), 1);
    assert_eq!(target_bits(t * 1024, t), 10);
}

#[test]
fn small_store_search_is_exact() {
    let dir = TempDir::new().unwrap();
    let store = filled_store(&dir, 300, 3);
    let q = [0.5, 0.5, 0.5];
    let (dist, idx) = store
        .search(&ndarray::arr2(&[q]).view(), 5, false)
        .unwrap();
    assert_eq!(idx.row(0).to_vec(), exact_top(&store, &q, 5));
    assert!(dist[[0, 0]] <= dist[[0, 4]]);
}

#[test]
fn large_store_recall_and_bucket_bounds() {
    let dir = TempDir::new().unwrap();
    let store = filled_store(&dir, 4 * O1_EXACT_ROWS, 4);
    let p = O1Params::default();
    assert_eq!(store.hash_bits(), target_bits(store.len(), p.bucket_target));
    let mut hits = 0;
    for seed in 0..20u64 {
        let (q, _) = random_rows(1, 4, 100 + seed);
        let q = q.row(0).to_vec();
        assert!(store.index.candidate_ids(&q, &Default::default(), &store.x).unwrap().len() <= p.num_tables * (1 + p.num_probes) * p.bucket_cap);
        let (_, idx) = store.search(&ndarray::arr2(&[[q[0], q[1], q[2], q[3]]]).view(), 1, false).unwrap();
        hits += usize::from(idx[[0, 0]] == exact_top(&store, &q, 1)[0]);
    }
    assert!(hits >= 14, "recall@1 too low: {hits}/20");
}

#[test]
fn exclude_nearest_drops_the_query_row() {
    let dir = TempDir::new().unwrap();
    let store = filled_store(&dir, 2000, 3);
    let q = store.row_x(17).unwrap();
    let qv = q.view().insert_axis(ndarray::Axis(0));
    let (_, with_self) = store.search(&qv, 3, false).unwrap();
    let (_, without) = store.search(&qv, 3, true).unwrap();
    assert_eq!(with_self[[0, 0]], 17);
    assert!(without.row(0).iter().all(|&i| i != 17));
    assert_eq!(without.ncols(), 3);
}

#[test]
fn metric_scale_rebuild_and_reopen_round_trip() {
    let dir = TempDir::new().unwrap();
    let mut store = filled_store(&dir, O1_EXACT_ROWS, 3);
    let scale = Array1::from(vec![1.0, 10.0, 10.0]);
    store.set_metric_scale(&scale, true).unwrap();
    assert!(store.set_metric_scale(&Array1::from(vec![1.0]), true).is_err());
    let q = [0.2, 0.9, 0.1];
    let (_, before) = store.search(&ndarray::arr2(&[q]).view(), 4, false).unwrap();
    drop(store);
    let reopened = O1Store::open_or_create(dir.path().to_path_buf(), 3, 1).unwrap();
    assert_eq!(reopened.len(), O1_EXACT_ROWS);
    assert_eq!(reopened.metric_scale(), Some(scale));
    let (_, after) = reopened.search(&ndarray::arr2(&[q]).view(), 4, false).unwrap();
    assert_eq!(before, after);
    assert!(reopened.index_memory_bytes() > 0);
    assert_eq!((reopened.num_dim(), reopened.num_metrics()), (3, 1));
}

#[test]
fn yvar_rows_round_trip_and_mismatch_is_rejected() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_or_create(dir.path().to_path_buf(), 2, 1).unwrap();
    assert!(store.is_empty());
    let x = ndarray::arr2(&[[0.0, 1.0], [1.0, 0.0]]);
    let y = ndarray::arr2(&[[1.0], [2.0]]);
    let v = ndarray::arr2(&[[0.1], [0.2]]);
    store.append_rows(&x.view(), &y.view(), Some(&v.view())).unwrap();
    assert_eq!(store.row_yvar(1).unwrap().unwrap()[0], 0.2);
    assert_eq!(store.row_y(0).unwrap()[0], 1.0);
    assert!(store.append_rows(&x.view(), &y.view(), None).is_err());
    let (tx, ty, tv) = store.train_rows_at(&[1]).unwrap();
    assert_eq!((tx[[0, 0]], ty[[0, 0]], tv.unwrap()[[0, 0]]), (1.0, 2.0, 0.2));
}

#[test]
fn o1_model_learns_metric_and_reopens() {
    use crate::layout::{EnnLayout, StoreType};
    use crate::model::EpistemicNearestNeighbors;
    for store in [StoreType::O1, StoreType::O1Forest] {
        let dir = TempDir::new().unwrap();
        let layout = EnnLayout::new(dir.path(), false).unwrap().with_store(store);
        let (x, y) = random_rows(400, 3, 11);
        let mut model = EpistemicNearestNeighbors::new(x.clone(), y.clone(), None, layout).unwrap();
        assert!(model.auto_metric.is_some());
        model.add(&x.view(), &y.view(), None).unwrap();
        model.add(&x.view(), &y.view(), None).unwrap();
        let idx = model.neighbors(&x.slice(ndarray::s![0..2, ..]), 3, false).unwrap();
        assert_eq!(idx.shape(), &[2, 3]);
        drop(model);
        let reservoir = dir.path().join("metric_reservoir.json");
        std::fs::remove_file(&reservoir).ok();
        let reopened = EpistemicNearestNeighbors::open(dir.path()).unwrap();
        assert!(reservoir.exists());
        assert_eq!(reopened.len(), 1200);
        assert_eq!(reopened.layout.store(), store);
    }
}

#[test]
#[ignore]
fn o1_recall_and_candidates_versus_n() {
    let base = O1Params::default();
    let settings = [
        base,
        O1Params { num_tables: 16, ..base },
        O1Params { num_probes: 12, ..base },
        O1Params { bucket_target: 32, bucket_cap: 128, ..base },
    ];
    for d in [10usize, 100] {
        for params in settings {
            recall_diagnostic(d, params);
        }
    }
}

fn recall_diagnostic(d: usize, params: O1Params) {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_with_params(dir.path().to_path_buf(), (d, 1), params).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let queries = Array2::from_shape_fn((200, d), |_| rng.gen::<f64>());
    for target in [1000usize, 10000, 100000] {
        while store.len() < target {
            let (x, y) = random_rows(1000, d, store.len() as u64);
            store.append_rows(&x.view(), &y.view(), None).unwrap();
        }
        let mut cand = 0usize;
        let mut hits = 0usize;
        for q in queries.rows() {
            let qv = q.to_vec();
            cand += store.index.candidate_ids(&qv, &Default::default(), &store.x).unwrap().len();
            let truth = exact_top(&store, &qv, 10);
            let (_, idx) = store.search(&q.insert_axis(ndarray::Axis(0)), 10, false).unwrap();
            hits += idx.row(0).iter().filter(|i| truth.contains(i)).count();
        }
        println!(
            "O1DIAG d={d} L={} T={} B={} P={} n={target} bits={} mean_candidates={:.1} recall10={:.3}",
            params.num_tables,
            params.bucket_target,
            params.bucket_cap,
            params.num_probes,
            store.hash_bits(),
            cand as f64 / 200.0,
            hits as f64 / 2000.0
        );
    }
}

#[test]
fn occupancy_counts_reachable_rows_within_cap() {
    let dir = TempDir::new().unwrap();
    let store = filled_store(&dir, 3000, 3);
    let occ = store.occupancy().unwrap();
    assert_eq!(occ.num_rows, 3000);
    assert_eq!(occ.hash_bits, store.hash_bits());
    assert_eq!(occ.table_rows.len(), O1Params::default().num_tables);
    assert!(occ.reachable_rows <= occ.num_rows);
    for (t, &rows) in occ.table_rows.iter().enumerate() {
        assert!(rows <= occ.reachable_rows);
        assert!(rows <= occ.keys_used[t] * occ.bucket_cap);
        assert!(occ.overflow_keys[t] <= occ.keys_used[t]);
    }
}

#[test]
fn occupancy_shows_overflow_loss_for_duplicate_rows() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_or_create(dir.path().to_path_buf(), 2, 1).unwrap();
    let x = Array2::from_elem((2000, 2), 0.25);
    let y = Array2::zeros((2000, 1));
    store.append_rows(&x.view(), &y.view(), None).unwrap();
    let occ = store.occupancy().unwrap();
    let cap = occ.bucket_cap;
    assert_eq!(occ.max_arrivals, 2000);
    assert!(occ.overflow_keys.iter().all(|&k| k == 1));
    assert!(occ.reachable_rows <= cap * occ.table_rows.len());
}

#[test]
fn candidates_cover_search_results() {
    let dir = TempDir::new().unwrap();
    let store = filled_store(&dir, 3000, 4);
    let q = ndarray::arr2(&[[0.3, 0.6, 0.1, 0.9], [0.5, 0.5, 0.5, 0.5]]);
    let cands = store.candidates(&q.view(), 10).unwrap();
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    assert_eq!(cands.len(), 2);
    for (qi, c) in cands.iter().enumerate() {
        assert!(c.len() >= 10);
        for &id in idx.row(qi) {
            assert!(c.contains(&(id as u32)));
        }
    }
}
