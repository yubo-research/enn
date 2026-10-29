//! Compare numerical refinement pools with unchanged growing trees and float64 ranking.
use std::io::Write;
use std::time::Instant;
use ennbo_bpann::index::search::MmapSearchStore;
use ennbo_bpann::index::tree_search::search_tree_refined_with_extra as search_tree_refined;
use ennbo_bpann::mmap_store::MmapColumnStore;
use ennbo_bpann::BpannBackend;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let output = std::env::args().nth(1).expect("output JSONL path");
    let mut output = std::fs::File::create(output).unwrap();
    for d in [3, 16] {
        let mut rng = ChaCha8Rng::seed_from_u64(d as u64);
        let x = Array2::from_shape_fn((1_000_000, d), |_| rng.gen::<f64>());
        let q = Array2::from_shape_fn((64, d), |_| rng.gen::<f64>());
        let y = Array2::zeros((1000, 1));
        let dir = tempfile::TempDir::new().unwrap();
        let mut raw = MmapColumnStore::mmap_open_or_create(dir.path().join("raw.bin"), d, None).unwrap();
        raw.mmap_append(&x.view()).unwrap();
        let mut model = BpannBackend::new_empty(dir.path().join("model"), d, 1).unwrap();
        let mut scale = Array1::ones(d);
        let mut n = 0;
        for end in [300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000] {
            while n < end {
                let hi = (n+1000).min(end);
                model.append_rows(&x.slice(ndarray::s![n..hi, ..]), &y.slice(ndarray::s![..hi-n, ..]), None).unwrap();
                n = hi;
            }
            if end == 10000 {
                scale = Array1::from_shape_fn(d, |j| if j < d.min(4) { 0.3 } else { 30.0 });
                model.rebuild_metric(&scale).unwrap();
            }
            let store = MmapSearchStore { train_x: &raw, scale_x: true, x_scale: scale.as_slice().unwrap() };
            let tree = model.index_snapshot().unwrap();
            let reference: Vec<_> = q.outer_iter().map(|row| search_tree_refined(tree, row.as_slice().unwrap(), 42, &store, 32).unwrap()).collect();
            for repetition in 0..3 {
                let mut extras = [1, 3, 10, 30, 32];
                extras.rotate_left(repetition);
                for extra in extras {
                    let start = Instant::now();
                    let actual: Vec<_> = q.outer_iter().map(|row| search_tree_refined(tree, row.as_slice().unwrap(), 42, &store, extra).unwrap()).collect();
                    let elapsed = start.elapsed().as_secs_f64();
                    let mismatches = actual.iter().zip(&reference).filter(|(a,b)| a[..10] != b[..10]).count();
                    writeln!(output,"{}",serde_json::json!({"d":d,"n":end,"extra":extra,"repetition":repetition,"query_us":elapsed*1e6/64.0,"first10_mismatches":mismatches})).unwrap();
                }
            }
            output.flush().unwrap();
            println!("d={d} n={end} done");
        }
    }
}
