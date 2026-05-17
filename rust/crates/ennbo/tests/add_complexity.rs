use ennbo::{EpistemicNearestNeighbors, IndexDriver};
use ndarray::Array2;
use std::hint::black_box;
use std::time::{Duration, Instant};

fn deterministic_x(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 2), |(i, j)| {
        let base = i as f64;
        if j == 0 {
            base * 0.001
        } else {
            (base * 0.017).sin()
        }
    })
}

fn deterministic_y(n: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, 1), |(i, _)| (i as f64 * 0.013).cos())
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
        let x = Array2::from_shape_vec(
            (1, 2),
            vec![
                10_000.0 + i as f64 * 0.001,
                (10_000.0 + i as f64 * 0.017).sin(),
            ],
        )
        .unwrap();
        let y = Array2::from_shape_vec((1, 1), vec![(10_000.0 + i as f64 * 0.013).cos()])
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

#[derive(Debug)]
struct QuadraticFit {
    t_b: f64,
    t_c: f64,
}

fn invert_3x3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    assert!(det.abs() > f64::EPSILON, "quadratic design matrix is singular");

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ]
}

fn quadratic_fit_t_stats(measurements: &[(usize, f64)]) -> QuadraticFit {
    assert!(measurements.len() > 3, "need residual degrees of freedom");
    let n_scale = measurements
        .iter()
        .map(|(n, _)| *n as f64)
        .fold(0.0, f64::max);

    let mut xtx = [[0.0; 3]; 3];
    let mut xty = [0.0; 3];
    for &(n, t) in measurements {
        let x = n as f64 / n_scale;
        let row = [1.0, x, x * x];
        for i in 0..3 {
            xty[i] += row[i] * t;
            for j in 0..3 {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }

    let xtx_inv = invert_3x3(xtx);
    let mut beta = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            beta[i] += xtx_inv[i][j] * xty[j];
        }
    }

    let mut rss = 0.0;
    for &(n, t) in measurements {
        let x = n as f64 / n_scale;
        let predicted = beta[0] + beta[1] * x + beta[2] * x * x;
        let residual = t - predicted;
        rss += residual * residual;
    }
    let sigma2 = rss / (measurements.len() as f64 - 3.0);
    let se_b = (sigma2 * xtx_inv[1][1]).sqrt();
    let se_c = (sigma2 * xtx_inv[2][2]).sqrt();

    QuadraticFit {
        t_b: beta[1] / se_b,
        t_c: beta[2] / se_c,
    }
}

fn assert_single_row_add_has_flat_growth(scale_x: bool, with_yvar: bool) {
    let num_adds = 128;
    let mut last: Option<(QuadraticFit, Vec<(usize, f64)>)> = None;
    for _ in 0..25 {
        let measurements: Vec<(usize, f64)> = [1_000, 3_000, 10_000, 30_000, 100_000]
            .into_iter()
            .map(|n| {
                let elapsed = timed_single_row_adds_with_yvar(n, num_adds, scale_x, with_yvar);
                (n, elapsed.as_secs_f64())
            })
            .collect();

        let fit = quadratic_fit_t_stats(&measurements);
        if fit.t_b.abs() < 1.0 && fit.t_c.abs() < 1.0 {
            return;
        }
        last = Some((fit, measurements));
    }

    let (fit, measurements) = last.expect("expected at least one attempt");
    panic!(
        "single-row add should be effectively independent of existing row count \
         when scale_x={scale_x} with_yvar={with_yvar}: measurements={measurements:?}, \
         t_b={}, t_c={}",
        fit.t_b,
        fit.t_c
    );
}

#[test]
fn single_row_add_scale_x_false_has_flat_growth_with_existing_n() {
    assert_single_row_add_has_flat_growth(false, false);
}

#[test]
fn single_row_add_scale_x_true_has_flat_growth_with_existing_n() {
    assert_single_row_add_has_flat_growth(true, false);
}

#[test]
fn single_row_add_with_yvar_scale_x_false_has_flat_growth_with_existing_n() {
    assert_single_row_add_has_flat_growth(false, true);
}

#[test]
fn single_row_add_with_yvar_scale_x_true_has_flat_growth_with_existing_n() {
    assert_single_row_add_has_flat_growth(true, true);
}
