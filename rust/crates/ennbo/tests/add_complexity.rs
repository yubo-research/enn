use ennbo::EpistemicNearestNeighbors;
use ndarray::Array2;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const ROW_COUNTS: [usize; 5] = [1_000, 3_000, 10_000, 30_000, 100_000];
const BASELINE_ROWS: usize = ROW_COUNTS[0];
const NUM_ADDS: usize = 32;
const MAX_SECS_PER_BASELINE_SEC: f64 = 2.5;
const TIMING_SAMPLES: usize = 10;
const MAX_ATTEMPTS: usize = 3;

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

fn layout(dir: &TempDir, weighting: bool) -> ennbo::EnnLayout {
    ennbo::EnnLayout::new(dir.path(), weighting).unwrap()
}

fn model_with_rows(dir: &TempDir, n: usize, scale_x: bool) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new(
        deterministic_x(n),
        deterministic_y(n),
        None,
        layout(dir, scale_x),
    )
    .unwrap()
}

fn model_with_rows_and_yvar(dir: &TempDir, n: usize, scale_x: bool) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new(
        deterministic_x(n),
        deterministic_y(n),
        Some(deterministic_yvar(n)),
        layout(dir, scale_x),
    )
    .unwrap()
}

fn prepared_model(
    starting_rows: usize,
    scale_x: bool,
    with_yvar: bool,
) -> (EpistemicNearestNeighbors, TempDir) {
    let dir = TempDir::new().unwrap();
    let mut model = if with_yvar {
        model_with_rows_and_yvar(&dir, starting_rows, scale_x)
    } else {
        model_with_rows(&dir, starting_rows, scale_x)
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

    (model, dir)
}

type PreparedModels = [(EpistemicNearestNeighbors, TempDir); ROW_COUNTS.len()];

fn measure_flat_growth(
    models: &mut PreparedModels,
    attempt: usize,
    with_yvar: bool,
) -> Vec<(usize, f64)> {
    let yvar = with_yvar.then(|| deterministic_yvar(1));
    let mut samples = [[Duration::ZERO; ROW_COUNTS.len()]; TIMING_SAMPLES];
    // Compare the same input under nearby system load, rotating which size goes first.
    for (sample, batch) in samples.iter_mut().enumerate() {
        for i in 0..NUM_ADDS {
            let row = (attempt * TIMING_SAMPLES + sample) * NUM_ADDS + i;
            let fi = row_f64(row);
            let x = ndarray::arr2(&[[
                fi.mul_add(0.001, 10_000.0),
                fi.mul_add(0.017, 10_000.0).sin(),
            ]]);
            let y = ndarray::arr2(&[[fi.mul_add(0.013, 10_000.0).cos()]]);
            let yvar_view = yvar.as_ref().map(|a| a.view());
            for offset in 0..models.len() {
                let index = (row + offset) % models.len();
                let start = Instant::now();
                models[index]
                    .0
                    .add(&x.view(), &y.view(), yvar_view.as_ref())
                    .unwrap();
                batch[index] += start.elapsed();
            }
        }
    }
    ROW_COUNTS
        .into_iter()
        .enumerate()
        .map(|(index, n)| {
            // Contention only adds time, so the fastest batch best estimates the add cost.
            let fastest = samples
                .iter()
                .map(|batch| batch[index])
                .min()
                .expect("at least one timing sample");
            (n, fastest.as_secs_f64())
        })
        .collect()
}

fn flat_growth_violation(
    measurements: &[(usize, f64)],
    scale_x: bool,
    with_yvar: bool,
) -> Option<String> {
    let baseline_secs = measurements
        .iter()
        .find(|(n, _)| *n == BASELINE_ROWS)
        .map(|(_, t)| *t)
        .expect("baseline row count must be measured");
    if baseline_secs <= 0.0 {
        return Some(format!(
            "baseline timing must be positive: {measurements:?}"
        ));
    }

    let budget_secs = baseline_secs * MAX_SECS_PER_BASELINE_SEC;
    let mut observed_peak_ratio = 1.0_f64;
    for &(n, secs) in measurements {
        let ratio = secs / baseline_secs;
        observed_peak_ratio = observed_peak_ratio.max(ratio);
        if secs > budget_secs {
            return Some(format!(
                "single-row add at n={n} took {secs}s, budget {budget_secs}s \
                 ({MAX_SECS_PER_BASELINE_SEC}x baseline {BASELINE_ROWS}={baseline_secs}s) \
                 scale_x={scale_x} with_yvar={with_yvar}: {measurements:?}",
            ));
        }
    }
    if observed_peak_ratio > MAX_SECS_PER_BASELINE_SEC {
        return Some(format!(
            "peak ratio {observed_peak_ratio} exceeds calibrated cap {MAX_SECS_PER_BASELINE_SEC} \
             scale_x={scale_x} with_yvar={with_yvar}: {measurements:?}",
        ));
    }
    None
}

fn assert_flat_growth_for_config(scale_x: bool, with_yvar: bool) {
    let mut models = ROW_COUNTS.map(|n| prepared_model(n, scale_x, with_yvar));
    let mut failures = Vec::with_capacity(MAX_ATTEMPTS);
    // Real growth fails every attempt; a burst of load from other tests rarely does.
    for attempt in 0..MAX_ATTEMPTS {
        let measurements = measure_flat_growth(&mut models, attempt, with_yvar);
        match flat_growth_violation(&measurements, scale_x, with_yvar) {
            None => return,
            Some(message) => failures.push(message),
        }
    }
    panic!(
        "flat growth failed in all {MAX_ATTEMPTS} attempts:\n{}",
        failures.join("\n")
    );
}

#[test]
fn single_row_add_flat_growth_scale_x_false_yvar_false() {
    assert_flat_growth_for_config(false, false);
}

#[test]
fn single_row_add_flat_growth_scale_x_false_yvar_true() {
    assert_flat_growth_for_config(false, true);
}

#[test]
fn scale_x_true_incremental_add_succeeds() {
    let dir = TempDir::new().unwrap();
    let mut model = model_with_rows(&dir, 100, true);
    let x = deterministic_x(1);
    let y = deterministic_y(1);
    model
        .add(&x.view(), &y.view(), None)
        .expect("scale_x=true incremental add must succeed on nonempty model");
    assert_eq!(model.len(), 101);
}
