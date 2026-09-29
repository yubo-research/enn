//! Query cost and recall after a large metric change, while rows keep arriving.
//!
//! Usage (temporarily copied to `rust/crates/bpann/examples/`):
//! `cargo run --release -p ennbo-bpann --example morph_bench -- N0 EXTRA {uniform|corner} SHRINK [nomorph]`
//! 12-d uniform rows; the first N0 are bulk-built under the identity metric, then dims
//! 2..11 are scaled by SHRINK and EXTRA more rows are added in batches of 1000.
//! `nomorph` rescales the coordinates but does not start the re-partitioning sweep.

use std::time::Instant;

use ennbo_bpann::index::search::MmapSearchStore;
use ennbo_bpann::mmap_store::MmapColumnStore;
use ennbo_bpann::IncrementalIndex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const D: usize = 12;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n0: usize = args.get(1).map_or(100_000, |s| s.parse().unwrap());
    let extra: usize = args.get(2).map_or(60_000, |s| s.parse().unwrap());
    let corner = args.get(3).is_some_and(|s| s == "corner");
    let dir = tempfile::TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let x = ndarray::Array2::from_shape_fn((n0 + extra, D), |(i, j)| {
        let u = rng.gen::<f64>();
        if corner && i >= n0 && j < 2 { 0.2 * u } else { u }
    });
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), D, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    let shrink: f64 = args.get(4).map_or(0.001, |s| s.parse().unwrap());
    let ratio: Vec<f64> = (0..D).map(|j| if j < 2 { 1.0 } else { shrink }).collect();
    let x_scale: Vec<f64> = ratio.iter().map(|r| 1.0 / r).collect();
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    idx.ensure_sync_for_backend(&store, D, false, &[1.0; D], n0).unwrap();
    idx.rescale_coords(&ratio);
    if args.get(5).map_or(true, |s| s != "nomorph") {
        idx.start_morph();
    }
    let s = MmapSearchStore { train_x: &store, scale_x: true, x_scale: &x_scale };
    let mut qrng = ChaCha8Rng::seed_from_u64(7);
    let queries: Vec<Vec<f32>> =
        (0..300).map(|_| (0..D).map(|j| (qrng.gen::<f64>() * ratio[j]) as f32).collect()).collect();
    let scaled = |r: usize| -> Vec<f32> { (0..D).map(|j| (x[[r, j]] * ratio[j]) as f32).collect() };
    let report = |idx: &IncrementalIndex, end: usize, label: &str, add_s: f64, worst: f64| {
        let t = Instant::now();
        let got: Vec<Vec<u32>> =
            queries.iter().map(|q| idx.search_candidates(q, 10, Some(&s)).unwrap().iter().map(|g| g.0).collect()).collect();
        let q_s = t.elapsed().as_secs_f64();
        let rows: Vec<Vec<f32>> = (0..end).map(scaled).collect();
        let mut hit = 0usize;
        for (q, g) in queries.iter().zip(&got).take(100) {
            let mut all: Vec<(f32, u32)> =
                rows.iter().enumerate().map(|(r, v)| (ennbo_bpann::distance::l2_sq_f32(q, v), r as u32)).collect();
            all.select_nth_unstable_by(10, |a, b| a.0.total_cmp(&b.0));
            hit += all[..10].iter().filter(|a| g.contains(&a.1)).count();
        }
        let pages = idx.indices[0].pages.len();
        println!(
            "{label:>8} end={end:>7} pages={pages:>6} q300={q_s:.4}s recall={:.4} add_us/row={:.2} worst_batch={worst:.4}s morphing={}",
            hit as f64 / 1000.0,
            add_s * 1e6,
            idx.morphing()
        );
    };
    report(&idx, n0, "start", 0.0, 0.0);
    let (mut end, batch) = (n0, 1000usize);
    let (mut acc, mut worst, mut since) = (0.0f64, 0.0f64, 0usize);
    while end < n0 + extra {
        let t = Instant::now();
        idx.ensure_sync_for_backend(&store, D, true, &x_scale, end + batch).unwrap();
        let dt = t.elapsed().as_secs_f64();
        acc += dt;
        worst = worst.max(dt);
        since += batch;
        end += batch;
        let k = end - n0;
        if [1000, 2000, 5000, 10_000, 20_000, 40_000, 60_000, 100_000, 200_000].contains(&k) || end == n0 + extra {
            report(&idx, end, "morph", acc / since as f64, worst);
            (acc, worst, since) = (0.0, 0.0, 0);
        }
    }
    let fresh_dir = tempfile::TempDir::new().unwrap();
    let mut fresh = IncrementalIndex::new(fresh_dir.path().join("index"));
    fresh.ensure_sync_for_backend(&store, D, true, &x_scale, end).unwrap();
    report(&fresh, end, "fresh", 0.0, 0.0);
}
