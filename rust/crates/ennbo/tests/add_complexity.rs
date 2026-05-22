use ennbo::{EpistemicNearestNeighbors, IndexDriver};
use ndarray::Array2;
use std::hint::black_box;
use std::time::{Duration, Instant};

const ROW_COUNTS: [usize; 5] = [1_000, 3_000, 10_000, 30_000, 100_000];
const BASELINE_ROWS: usize = ROW_COUNTS[0];
const NUM_ADDS: usize = 128;
const CONFIGS: [(bool, bool); 4] = [(false, false), (true, false), (false, true), (true, true)];
const MAX_SECS_PER_BASELINE_SEC: f64 = 2.5;
const TIMING_REPEATS: usize = 5;

fn row_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).unwrap_or(u32::MAX))
}

fn deterministic_x(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, j)| {
        let base = row_f64(i);
        if j == 0 {
            base * 0.001
        } else {
            (base * 0.017).sin()
        }
    })
}

fn deterministic_y(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 1), |(i, _)| (row_f64(i) * 0.013).cos())
}

fn deterministic_yvar(n: usize) -> Array2<f64> {
    Array2::from_elem((n, 1), 1e-6)
}

fn model_with_rows(n: usize, scale_x: bool) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new(
        deterministic_x(n),
        deterministic_y(n),
        None,
        scale_x,
        IndexDriver::Exact,
    )
    .unwrap()
}

fn model_with_rows_and_yvar(n: usize, scale_x: bool) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new(
        deterministic_x(n),
        deterministic_y(n),
        Some(deterministic_yvar(n)),
        scale_x,
        IndexDriver::Exact,
    )
    .unwrap()
}

fn timed_single_row_adds_with_yvar(
    starting_rows: usize,
    num_adds: usize,
    scale_x: bool,
    with_yvar: bool,
) -> Duration {
    let mut model = if with_yvar {
        model_with_rows_and_yvar(starting_rows, scale_x)
    } else {
        model_with_rows(starting_rows, scale_x)
    };

    let warm_x = deterministic_x(1);
    let warm_y = deterministic_y(1);
    let warm_yvar = if with_yvar {
        Some(deterministic_yvar(1))
    } else {
        None
    };
    match warm_yvar.as_ref() {
        Some(yv) => model
            .add(&warm_x.view(), &warm_y.view(), Some(&yv.view()))
            .unwrap(),
        None => model.add(&warm_x.view(), &warm_y.view(), None).unwrap(),
    }

    let start = Instant::now();
    for i in 0..num_adds {
        let fi = row_f64(i);
        let x = Array2::from_shape_vec(
            (1, 2),
            vec![
                fi.mul_add(0.001, 10_000.0),
                fi.mul_add(0.017, 10_000.0).sin(),
            ],
        )
        .unwrap();
        let y = Array2::from_shape_vec((1, 1), vec![fi.mul_add(0.013, 10_000.0).cos()])
            .unwrap();
        let yvar = if with_yvar {
            Some(deterministic_yvar(1))
        } else {
            None
        };
        match yvar.as_ref() {
            Some(yv) => model.add(&x.view(), &y.view(), Some(&yv.view())).unwrap(),
            None => model.add(&x.view(), &y.view(), None).unwrap(),
        }
    }
    let elapsed = start.elapsed();

    black_box(model.len());
    elapsed
}

fn best_of_repeats_secs(
    starting_rows: usize,
    num_adds: usize,
    scale_x: bool,
    with_yvar: bool,
) -> f64 {
    (0..TIMING_REPEATS)
        .map(|_| {
            timed_single_row_adds_with_yvar(starting_rows, num_adds, scale_x, with_yvar).as_secs_f64()
        })
        .min_by(|a, b| a.partial_cmp(b).expect("timing samples must be ordered"))
        .expect("at least one timing repeat")
}

#[test]
fn single_row_add_has_flat_growth_across_configs() {
    let mut observed_peak_ratio = 1.0_f64;
    for &(scale_x, with_yvar) in &CONFIGS {
        let measurements: Vec<(usize, f64)> = ROW_COUNTS
            .into_iter()
            .map(|n| (n, best_of_repeats_secs(n, NUM_ADDS, scale_x, with_yvar)))
            .collect();

        let baseline_secs = measurements
            .iter()
            .find(|(n, _)| *n == BASELINE_ROWS)
            .map(|(_, t)| *t)
            .expect("baseline row count must be measured");
        assert!(baseline_secs > 0.0, "baseline timing must be positive: {measurements:?}");

        let budget_secs = baseline_secs * MAX_SECS_PER_BASELINE_SEC;
        for &(n, secs) in &measurements {
            let ratio = secs / baseline_secs;
            observed_peak_ratio = observed_peak_ratio.max(ratio);
            assert!(
                secs <= budget_secs,
                "single-row add at n={n} took {secs}s, budget {budget_secs}s \
                 ({MAX_SECS_PER_BASELINE_SEC}x baseline {BASELINE_ROWS}={baseline_secs}s) \
                 scale_x={scale_x} with_yvar={with_yvar}: {measurements:?}",
            );
        }
    }
    assert!(
        observed_peak_ratio <= MAX_SECS_PER_BASELINE_SEC,
        "peak ratio {observed_peak_ratio} exceeds calibrated cap {MAX_SECS_PER_BASELINE_SEC}",
    );
}
