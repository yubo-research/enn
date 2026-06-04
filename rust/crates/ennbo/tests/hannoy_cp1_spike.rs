//! CP-1 spike: hannoy open/build/search/incremental from Rust (tempdir).
#![cfg(feature = "hannoy")]

use hannoy::{distances::Euclidean, Database, Reader, Writer};
use heed::EnvOpenOptions;
use rand::Rng;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;
use tempfile::TempDir;

fn open_env(path: &std::path::Path) -> heed::Env {
    unsafe {
        EnvOpenOptions::new()
            .map_size(1 << 30)
            .max_dbs(1)
            .open(path)
            .unwrap()
    }
}

fn brute_top_k(query: &[f32], all: &[f32], dim: usize, n: u32, k: usize) -> HashSet<u32> {
    let mut dists: Vec<(u32, f32)> = (0..n)
        .map(|id| {
            let start = id as usize * dim;
            let sq: f32 = query
                .iter()
                .zip(all[start..start + dim].iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            (id, sq)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

fn hannoy_build_batch<'a>(
    env: &heed::Env,
    db: Database<Euclidean>,
    dim: usize,
    ef: usize,
    ids_data: impl Iterator<Item = (u32, &'a [f32])>,
    build_seed: u64,
) {
    let mut wtxn = env.write_txn().unwrap();
    let writer = Writer::new(db, 0, dim);
    for (id, vec) in ids_data {
        writer.add_item(&mut wtxn, id, vec).unwrap();
    }
    let mut build_rng = StdRng::seed_from_u64(build_seed);
    writer
        .builder(&mut build_rng)
        .ef_construction(ef)
        .build::<16, 32>(&mut wtxn)
        .unwrap();
    wtxn.commit().unwrap();
}

#[test]
fn cp1_hannoy_build_search_incremental() {
    let dim = 8;
    let n0 = 200;
    let n_add = 50;
    let k = 5;
    let ef = 64;
    let dir = TempDir::new().unwrap();
    let hannoy_dir = dir.path().join("hannoy");
    std::fs::create_dir_all(&hannoy_dir).unwrap();

    let mut rng = StdRng::seed_from_u64(42);
    let data0: Vec<f32> = (0..n0 * dim).map(|_| rng.gen::<f32>()).collect();
    let data_add: Vec<f32> = (0..n_add * dim).map(|_| rng.gen::<f32>()).collect();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

    let env = open_env(&hannoy_dir);
    let mut wtxn = env.write_txn().unwrap();
    let db: Database<Euclidean> = env.create_database(&mut wtxn, None).unwrap();
    wtxn.commit().unwrap();

    let batch0 = (0..n0 as u32).map(|id| {
        let start = id as usize * dim;
        (id, &data0[start..start + dim])
    });
    hannoy_build_batch(&env, db, dim, ef, batch0, 0);

    let batch1 = (0..n_add as u32).map(|j| {
        let id = n0 as u32 + j;
        let start = j as usize * dim;
        (id, &data_add[start..start + dim])
    });
    hannoy_build_batch(&env, db, dim, ef, batch1, 1);

    let rtxn = env.read_txn().unwrap();
    let reader = Reader::<Euclidean>::open(&rtxn, 0, db).unwrap();
    let got: Vec<u32> = reader
        .nns(k)
        .ef_search(ef)
        .by_vector(&rtxn, &query)
        .unwrap()
        .into_nns()
        .into_iter()
        .map(|(id, _dist)| id)
        .collect();

    let mut all = data0.clone();
    all.extend_from_slice(&data_add);
    let gt = brute_top_k(&query, &all, dim, (n0 + n_add) as u32, k);
    let recall = got.iter().filter(|id| gt.contains(id)).count() as f64 / k as f64;
    assert!(
        recall >= 0.6,
        "recall@{k}={recall:.2} got={got:?} gt={gt:?}"
    );
}
