//! Benchmarks for x_obs/y_obs cache, exact_search memory, and ENN incremental fit.

#![allow(clippy::pedantic, clippy::nursery)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use ennbo::{
    config::ConfigOverrides,
    index::{ENNIndex, IndexDriver},
    optimizer_factory::create_optimizer_enn_with_overrides,
};

fn bench_x_obs_y_obs_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("x_obs_y_obs");
    for n in [10, 50, 200, 500] {
        group.bench_with_input(BenchmarkId::new("repeated_calls", n), &n, |b, &n| {
            let bounds = Array2::from_shape_fn((2, 2), |(_, j)| if j == 0 { 0.0 } else { 1.0 });
            let mut rng = StdRng::seed_from_u64(42);
            let mut opt = create_optimizer_enn_with_overrides(
                bounds,
                10,
                4,
                &mut rng,
                Some(&ConfigOverrides::default()),
            )
            .unwrap();

            for _ in 0..n {
                let x = opt.ask(2, &mut rng).unwrap();
                let y = Array2::from_shape_fn((2, 1), |(i, _)| {
                    -(x.row(i).mapv(|v| (v - 0.5).powi(2)).sum())
                });
                opt.tell(&x.view(), &y.view(), &mut rng).unwrap();
            }

            b.iter(|| {
                for _ in 0..5 {
                    black_box(opt.x_obs());
                    black_box(opt.y_obs());
                }
            });
        });
    }
    group.finish();
}

fn bench_exact_search_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("exact_search");
    for (n_train, n_query, k) in [
        (100, 50, 10_i32),
        (500, 100, 20_i32),
        (2000, 200, 30_i32),
        (5000, 500, 50_i32),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n_train={}_q={}_k={}", n_train, n_query, k)),
            &(n_train, n_query, k),
            |b, &(n_train, n_query, k)| {
                let mut rng = StdRng::seed_from_u64(123);
                let train = Array2::from_shape_fn((n_train, 5), |_| rng.gen::<f64>());
                let query = Array2::from_shape_fn((n_query, 5), |_| rng.gen::<f64>());
                let scale = Array1::from_elem(5, 1.0);
                let index =
                    ENNIndex::new(train.clone(), 5, scale, false, IndexDriver::Exact);

                b.iter(|| {
                    let (d, idx) = index.search(&query.view(), k, false).unwrap();
                    black_box((d, idx));
                });
            },
        );
    }
    group.finish();
}

fn bench_enn_fit_incremental(c: &mut Criterion) {
    let mut group = c.benchmark_group("enn_fit");
    for n_obs in [50, 100, 200, 400] {
        group.bench_with_input(
            BenchmarkId::new("append_only_tell_cycles", n_obs),
            &n_obs,
            |b, &n_obs| {
                let bounds = Array2::from_shape_fn((5, 2), |(_, j)| if j == 0 { 0.0 } else { 1.0 });
                let overrides = ConfigOverrides::default();
                let mut rng = StdRng::seed_from_u64(99);

                b.iter(|| {
                    let mut opt = create_optimizer_enn_with_overrides(
                        bounds.clone(),
                        10,
                        n_obs / 4,
                        &mut rng,
                        Some(&overrides),
                    )
                    .unwrap();
                    for _ in 0..(n_obs / 4) {
                        let x = opt.ask(4, &mut rng).unwrap();
                        let y = Array2::from_shape_fn((4, 1), |(i, _)| {
                            -(x.row(i).mapv(|v| (v - 0.5).powi(2)).sum())
                        });
                        opt.tell(&x.view(), &y.view(), &mut rng).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_x_obs_y_obs_cache,
    bench_exact_search_memory,
    bench_enn_fit_incremental,
);
criterion_main!(benches);
