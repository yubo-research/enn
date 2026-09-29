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
use ndarray::Array2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const D: usize = 12;
const BATCH: usize = 1000;
const CHECKPOINTS: [usize; 9] = [1000, 2000, 5000, 10_000, 20_000, 40_000, 60_000, 100_000, 200_000];

struct Bench {
    x: Array2<f64>,
    ratio: Vec<f64>,
    x_scale: Vec<f64>,
    queries: Vec<Vec<f32>>,
}

impl Bench {
    fn new(n0: usize, extra: usize, corner: bool, shrink: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let x = Array2::from_shape_fn((n0 + extra, D), |(i, j)| {
            let u = rng.gen::<f64>();
            if corner && i >= n0 && j < 2 { 0.2 * u } else { u }
        });
        let ratio: Vec<f64> = (0..D).map(|j| if j < 2 { 1.0 } else { shrink }).collect();
        let x_scale = ratio.iter().map(|r| 1.0 / r).collect();
        let mut qrng = ChaCha8Rng::seed_from_u64(7);
        let queries = (0..300).map(|_| (0..D).map(|j| (qrng.gen::<f64>() * ratio[j]) as f32).collect()).collect();
        Self { x, ratio, x_scale, queries }
    }

    fn scaled(&self, r: usize) -> Vec<f32> {
        (0..D).map(|j| (self.x[[r, j]] * self.ratio[j]) as f32).collect()
    }

    /// Fraction of the true 10 nearest rows (among the first `end`) found, over the first 100 queries.
    fn recall(&self, got: &[Vec<u32>], end: usize) -> f64 {
        let rows: Vec<Vec<f32>> = (0..end).map(|r| self.scaled(r)).collect();
        let mut hit = 0usize;
        for (q, g) in self.queries.iter().zip(got).take(100) {
            let mut all: Vec<(f32, u32)> =
                rows.iter().enumerate().map(|(r, v)| (ennbo_bpann::distance::l2_sq_f32(q, v), r as u32)).collect();
            all.select_nth_unstable_by(10, |a, b| a.0.total_cmp(&b.0));
            hit += all[..10].iter().filter(|a| g.contains(&a.1)).count();
        }
        hit as f64 / 1000.0
    }

    fn report(&self, store: &MmapColumnStore, idx: &IncrementalIndex, end: usize, label: &str, add: (f64, f64)) {
        let s = MmapSearchStore { train_x: store, scale_x: true, x_scale: &self.x_scale };
        let t = Instant::now();
        let got: Vec<Vec<u32>> = self
            .queries
            .iter()
            .map(|q| idx.search_candidates(q, 10, Some(&s)).unwrap().iter().map(|g| g.0).collect())
            .collect();
        let q_s = t.elapsed().as_secs_f64();
        println!(
            "{label:>8} end={end:>7} pages={:>6} q300={q_s:.4}s recall={:.4} add_us/row={:.2} worst_batch={:.4}s morphing={}",
            idx.indices[0].pages.len(),
            self.recall(&got, end),
            add.0 * 1e6,
            add.1,
            idx.morphing()
        );
    }
}

/// Add rows `n0..` in batches, reporting at each checkpoint the mean add time per row and the worst batch since the last one.
fn stream(bench: &Bench, store: &MmapColumnStore, idx: &mut IncrementalIndex, n0: usize) {
    let total = bench.x.nrows();
    let (mut end, mut acc, mut worst, mut since) = (n0, 0.0f64, 0.0f64, 0usize);
    while end < total {
        let t = Instant::now();
        idx.ensure_sync_for_backend(store, D, true, &bench.x_scale, end + BATCH).unwrap();
        let dt = t.elapsed().as_secs_f64();
        (acc, worst, since, end) = (acc + dt, worst.max(dt), since + BATCH, end + BATCH);
        if CHECKPOINTS.contains(&(end - n0)) || end == total {
            bench.report(store, idx, end, "morph", (acc / since as f64, worst));
            (acc, worst, since) = (0.0, 0.0, 0);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n0: usize = args.get(1).map_or(100_000, |s| s.parse().unwrap());
    let extra: usize = args.get(2).map_or(60_000, |s| s.parse().unwrap());
    let corner = args.get(3).is_some_and(|s| s == "corner");
    let shrink: f64 = args.get(4).map_or(0.001, |s| s.parse().unwrap());
    let bench = Bench::new(n0, extra, corner, shrink);
    let dir = tempfile::TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), D, None).unwrap();
    store.mmap_append(&bench.x.view()).unwrap();
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    idx.ensure_sync_for_backend(&store, D, false, &[1.0; D], n0).unwrap();
    idx.rescale_coords(&bench.ratio);
    if args.get(5).map_or(true, |s| s != "nomorph") {
        idx.start_morph();
    }
    bench.report(&store, &idx, n0, "start", (0.0, 0.0));
    stream(&bench, &store, &mut idx, n0);
    let fresh_dir = tempfile::TempDir::new().unwrap();
    let mut fresh = IncrementalIndex::new(fresh_dir.path().join("index"));
    fresh.ensure_sync_for_backend(&store, D, true, &bench.x_scale, n0 + extra).unwrap();
    bench.report(&store, &fresh, n0 + extra, "fresh", (0.0, 0.0));
}
