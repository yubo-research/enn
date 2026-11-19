use ennbo::EpistemicNearestNeighbors;
use ndarray::{Array2, ArrayView2};
use std::time::Duration;
use tempfile::TempDir;

const ROW_COUNTS: [usize; 5] = [1_000, 3_000, 10_000, 30_000, 100_000];
const NUM_ADDS: usize = 32;
const NUM_SAMPLES: usize = 11;
const MAX_BATCH_TIME_RATIO: f64 = 2.5;

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

fn model_with_rows(
    dir: &TempDir,
    n: usize,
    weighting: bool,
    with_yvar: bool,
) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new(
        deterministic_x(n),
        deterministic_y(n),
        with_yvar.then(|| deterministic_yvar(n)),
        ennbo::EnnLayout::new(dir.path(), weighting).unwrap(),
    )
    .unwrap()
}

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn thread_cpu_time() -> Duration {
    use std::ffi::{c_int, c_long};

    const CLOCK_THREAD_CPUTIME_ID: c_int = 3;
    #[repr(C)]
    struct Timespec {
        seconds: c_long,
        nanoseconds: c_long,
    }
    unsafe extern "C" {
        fn clock_gettime(clock_id: c_int, time: *mut Timespec) -> c_int;
    }

    let mut time = Timespec {
        seconds: 0,
        nanoseconds: 0,
    };
    // The 64-bit Linux timespec has two long fields and time is writable.
    let result = unsafe { clock_gettime(CLOCK_THREAD_CPUTIME_ID, &mut time) };
    assert_eq!(result, 0, "clock_gettime failed");
    Duration::new(
        time.seconds.try_into().unwrap(),
        time.nanoseconds.try_into().unwrap(),
    )
}

#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
fn batch_timer() -> impl FnOnce() -> Duration {
    // The append work is synchronous. Thread CPU time excludes descheduling
    // while the full suite runs other tests in parallel.
    let start = thread_cpu_time();
    move || thread_cpu_time() - start
}

#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
fn batch_timer() -> impl FnOnce() -> Duration {
    let start = std::time::Instant::now();
    move || start.elapsed()
}

fn time_add_batch(
    model: &mut EpistemicNearestNeighbors,
    batch: usize,
    yvar: Option<&ArrayView2<'_, f64>>,
) -> Duration {
    let elapsed = batch_timer();
    let first = batch * NUM_ADDS;
    for i in first..first + NUM_ADDS {
        let fi = row_f64(i);
        let x = Array2::from_shape_vec(
            (1, 2),
            vec![
                fi.mul_add(0.001, 10_000.0),
                fi.mul_add(0.017, 10_000.0).sin(),
            ],
        )
        .unwrap();
        let y = Array2::from_shape_vec((1, 1), vec![fi.mul_add(0.013, 10_000.0).cos()]).unwrap();
        model.add(&x.view(), &y.view(), yvar).unwrap();
    }
    elapsed()
}

fn median_growth_ratios(with_yvar: bool) -> [f64; ROW_COUNTS.len()] {
    let dirs = ROW_COUNTS.map(|_| TempDir::new().unwrap());
    let mut models: [_; ROW_COUNTS.len()] =
        std::array::from_fn(|i| model_with_rows(&dirs[i], ROW_COUNTS[i], false, with_yvar));
    let yvar = with_yvar.then(|| deterministic_yvar(1));
    let yvar = yvar.as_ref().map(Array2::view);
    let warm_x = deterministic_x(1);
    let warm_y = deterministic_y(1);
    for model in &mut models {
        model
            .add(&warm_x.view(), &warm_y.view(), yvar.as_ref())
            .unwrap();
    }

    // Compare each batch with a nearby baseline and rotate the first model each round.
    let mut ratios = [[0.0; NUM_SAMPLES]; ROW_COUNTS.len()];
    for batch in 0..NUM_SAMPLES {
        let mut elapsed = [Duration::ZERO; ROW_COUNTS.len()];
        for offset in 0..models.len() {
            let i = (batch + offset) % models.len();
            elapsed[i] = time_add_batch(&mut models[i], batch, yvar.as_ref());
        }
        let baseline = elapsed[0].as_secs_f64();
        assert!(baseline > 0.0);
        for (samples, time) in ratios.iter_mut().zip(elapsed) {
            samples[batch] = time.as_secs_f64() / baseline;
        }
    }

    for (model, starting_rows) in models.iter().zip(ROW_COUNTS) {
        assert_eq!(model.len(), starting_rows + 1 + NUM_SAMPLES * NUM_ADDS);
    }
    ratios.map(|mut samples| {
        samples.sort_unstable_by(f64::total_cmp);
        samples[NUM_SAMPLES / 2]
    })
}

fn assert_flat_growth_for_config(with_yvar: bool) {
    let ratios = median_growth_ratios(with_yvar);
    for (n, ratio) in ROW_COUNTS.into_iter().zip(ratios) {
        assert!(
            ratio <= MAX_BATCH_TIME_RATIO,
            "median single-row add batch time at n={n} was {ratio:.2}x the baseline, \
             above the {MAX_BATCH_TIME_RATIO}x limit with_yvar={with_yvar}: \
             starting rows {ROW_COUNTS:?}, ratios {ratios:?}",
        );
    }
}

#[test]
fn single_row_add_flat_growth_scale_x_false_yvar_false() {
    assert_flat_growth_for_config(false);
}

#[test]
fn single_row_add_flat_growth_scale_x_false_yvar_true() {
    assert_flat_growth_for_config(true);
}

#[test]
fn scale_x_true_incremental_add_succeeds() {
    let dir = TempDir::new().unwrap();
    let mut model = model_with_rows(&dir, 100, true, false);
    let x = deterministic_x(1);
    let y = deterministic_y(1);
    model
        .add(&x.view(), &y.view(), None)
        .expect("scale_x=true incremental add must succeed on nonempty model");
    assert_eq!(model.len(), 101);
}
