use super::*;
use crate::layout::EnnLayout;
use ndarray::{array, s, Axis};
use tempfile::TempDir;

struct Built {
    model: EpistemicNearestNeighbors,
    dir: TempDir,
}

fn build(aggregate: bool, x: Array2<f64>, y: Array2<f64>, yvar: Option<Array2<f64>>) -> Built {
    let dir = TempDir::new().unwrap();
    let layout = EnnLayout::new(dir.path(), false)
        .unwrap()
        .with_aggregate_duplicates(aggregate);
    let model = EpistemicNearestNeighbors::new(x, y, yvar, layout).unwrap();
    Built { model, dir }
}

fn distinct_points(n: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((n, 3), |(i, j)| ((i * 7 + j * 3) as f64 * 0.37).sin());
    let y = Array2::from_shape_fn((n, 2), |(i, j)| ((i + 2 * j) as f64 * 0.21).cos());
    let yvar = Array2::from_shape_fn((n, 2), |(i, j)| 0.05 + 0.01 * ((i + j) % 4) as f64);
    (x, y, yvar)
}

fn all_outputs(m: &EpistemicNearestNeighbors) -> Vec<ndarray::ArrayD<f64>> {
    let q = array![[0.1, -0.2, 0.3], [0.5, 0.5, -0.5], [-0.9, 0.0, 0.9]];
    let params = ENNParams::new(5, 2.5, 0.07).unwrap();
    let mut out = Vec::new();
    for exclude_nearest in [false, true] {
        for observation_noise in [false, true] {
            let flags = PosteriorFlags::new()
                .with_exclude_nearest(exclude_nearest)
                .with_observation_noise(observation_noise);
            let p = m.posterior(&q.view(), &params, &flags).unwrap();
            out.extend([p.mu, p.se, p.se_epi, p.se_ale]);
            let b = m
                .batch_posterior(&q.view(), &[params.clone(), params.clone()], &flags)
                .unwrap();
            out.extend([b.mu, b.se]);
            let xw = array![[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]];
            let yw = array![[0.4, 0.4], [1.0, 2.0]];
            let c = m
                .conditional_posterior(&xw.view(), &yw.view(), &q.view(), &params, &flags)
                .unwrap();
            out.extend([c.mu, c.se, c.se_epi, c.se_ale]);
            let (draws, idx) = m
                .posterior_function_draw(&q.view(), &params, &[3, 4], &flags)
                .unwrap();
            out.push(draws.into_dyn());
            out.push(Array2::from_shape_fn((idx.len(), idx[0].len()), |(i, j)| idx[i][j] as f64).into_dyn());
        }
    }
    let nb = m.neighbors(&q.view(), 4, false).unwrap();
    out.push(nb.mapv(|v| v as f64).into_dyn());
    out
}

#[test]
fn without_duplicates_aggregation_changes_nothing() {
    for with_yvar in [false, true] {
        let (x, y, yvar) = distinct_points(40);
        let yvar = with_yvar.then_some(yvar);
        let off = build(false, x.clone(), y.clone(), yvar.clone());
        let on = build(true, x, y, yvar);
        assert!(on.model.aggregates_duplicates() && !off.model.aggregates_duplicates());
        assert_eq!(all_outputs(&on.model), all_outputs(&off.model));
        assert_eq!(on.model.num_rows(), 40);
    }
}

#[test]
fn without_duplicates_adding_rows_keeps_the_two_settings_identical() {
    let (x, y, _) = distinct_points(30);
    let mut off = build(false, x.slice(s![..20, ..]).to_owned(), y.slice(s![..20, ..]).to_owned(), None);
    let mut on = build(true, x.slice(s![..20, ..]).to_owned(), y.slice(s![..20, ..]).to_owned(), None);
    for i in 20..30 {
        let (xi, yi) = (x.slice(s![i..i + 1, ..]), y.slice(s![i..i + 1, ..]));
        off.model.add(&xi, &yi, None).unwrap();
        on.model.add(&xi, &yi, None).unwrap();
    }
    assert_eq!(all_outputs(&on.model), all_outputs(&off.model));
}

fn site_model(aggregate: bool, copies: usize) -> Built {
    let mut x = Array2::zeros((copies + 4, 1));
    let mut y = Array2::zeros((copies + 4, 1));
    for i in 0..copies {
        y[[i, 0]] = 1.0 + if i % 2 == 0 { 0.5 } else { -0.5 };
    }
    for (j, (xv, yv)) in [(1.0, 0.0), (2.0, 3.0), (3.0, -1.0), (4.0, 2.0)].iter().enumerate() {
        x[[copies + j, 0]] = *xv;
        y[[copies + j, 0]] = *yv;
    }
    build(aggregate, x, y, None)
}

#[test]
fn se_keeps_shrinking_with_repeats_and_matches_the_row_formula() {
    let params = ENNParams::new(3, 1.0, 0.5).unwrap();
    let flags = PosteriorFlags::new();
    let q = array![[0.0]];
    let (mut on_se, mut off_se) = (Vec::new(), Vec::new());
    for copies in [2usize, 6, 10, 40] {
        let on = site_model(true, copies);
        let p = on.model.posterior(&q.view(), &params, &flags).unwrap();
        let ys = on.model.y_scale()[0];
        let w = |d: f64, n: f64| 1.0 / (EPS_VAR + d + 0.5 / n);
        let norm = w(0.0, copies as f64) + w(1.0, 1.0) + w(4.0, 1.0);
        let expected_mu = (w(0.0, copies as f64) * 1.0 + w(1.0, 1.0) * 0.0 + w(4.0, 1.0) * 3.0) / norm;
        assert!((p.mu[[0, 0]] - expected_mu).abs() < 1e-12, "copies={copies}");
        assert!((p.se_epi[[0, 0]] - (1.0 / norm).sqrt() * ys).abs() < 1e-12);
        on_se.push(p.se_epi[[0, 0]] / ys);
        let off = site_model(false, copies);
        let p = off.model.posterior(&q.view(), &params, &flags).unwrap();
        off_se.push(p.se_epi[[0, 0]] / off.model.y_scale()[0]);
    }
    assert!(on_se.windows(2).all(|w| w[1] < w[0]), "{on_se:?}");
    assert_eq!(off_se[2], off_se[3], "k copies fill the neighbors: {off_se:?}");
}

#[test]
fn neighbors_are_distinct_x_values_even_with_many_copies() {
    let copies = 50;
    let mut x = Array2::zeros((copies + 12, 2));
    for j in 0..12 {
        x[[copies + j, 0]] = (j + 1) as f64 * 0.1;
        x[[copies + j, 1]] = 0.5;
    }
    let built = build(true, x, Array2::zeros((copies + 12, 1)), None);
    let nb = built.model.neighbors(&array![[0.0, 0.0]].view(), 10, false).unwrap();
    assert_eq!(nb.ncols(), 10);
    let mut xs: Vec<Vec<f64>> = nb
        .row(0)
        .iter()
        .map(|&i| built.model.rows().row_x(i).unwrap().to_vec())
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup();
    assert_eq!(xs.len(), 10);
    assert_eq!(nb[[0, 0]], 0, "the representative is the first observation at x");
    assert_eq!(built.model.len(), copies + 12);
    assert_eq!(built.model.num_rows(), 13);
}

#[test]
fn mean_does_not_depend_on_the_order_of_interleaved_adds() {
    let sites = [(0.0, 1.0), (1.0, 4.0), (0.0, 3.0), (2.0, -1.0), (1.0, 2.0), (0.0, 5.0)];
    let grouped = [0usize, 2, 5, 1, 4, 3];
    let make = |order: &[usize]| {
        let mut b = build(true, array![[9.0]], array![[0.0]], None);
        for &i in order {
            b.model
                .add(&array![[sites[i].0]].view(), &array![[sites[i].1]].view(), None)
                .unwrap();
        }
        b
    };
    let a = make(&[0, 1, 2, 3, 4, 5]);
    let b = make(&grouped);
    let q = array![[0.2], [1.1], [1.9]];
    let params = ENNParams::new(3, 1.0, 0.3).unwrap();
    let pa = a.model.posterior(&q.view(), &params, &PosteriorFlags::new()).unwrap();
    let pb = b.model.posterior(&q.view(), &params, &PosteriorFlags::new()).unwrap();
    for (u, v) in pa.mu.iter().zip(pb.mu.iter()) {
        assert!((u - v).abs() < 1e-12);
    }
    for (u, v) in pa.se.iter().zip(pb.se.iter()) {
        assert!((u - v).abs() < 1e-12);
    }
}

#[test]
fn exclude_nearest_prediction_equals_the_model_built_without_that_x() {
    let (x, y, _) = distinct_points(25);
    let mut cx = x.clone();
    let mut cy = y.clone();
    for _ in 0..4 {
        cx = ndarray::concatenate![Axis(0), cx, x.slice(s![3..4, ..])];
        cy = ndarray::concatenate![Axis(0), cy, array![[5.0, -5.0]]];
    }
    let on = build(true, cx, cy, None);
    let kept: Vec<usize> = (0..25).filter(|&i| i != 3).collect();
    let without = build(true, x.select(Axis(0), &kept), y.select(Axis(0), &kept), None);
    let q = x.slice(s![3..4, ..]).to_owned();
    let params = ENNParams::new(6, 1.5, 0.2).unwrap();
    let got = on
        .model
        .posterior(&q.view(), &params, &PosteriorFlags::new().with_exclude_nearest(true))
        .unwrap();
    let want = without
        .model
        .posterior(&q.view(), &params, &PosteriorFlags::new())
        .unwrap();
    let (ys_on, ys_without) = (on.model.y_scale().to_owned(), without.model.y_scale().to_owned());
    for m in 0..2 {
        assert!((got.mu[[0, m]] - want.mu[[0, m]]).abs() < 1e-12);
        let epi = |s: f64, ys: f64| s / ys;
        assert!((epi(got.se_epi[[0, m]], ys_on[m]) - epi(want.se_epi[[0, m]], ys_without[m])).abs() < 1e-12);
    }
}

#[test]
fn equal_yvar_gives_the_precision_weighted_row_noise() {
    let n = 4usize;
    let v = 0.3;
    let on = build(
        true,
        Array2::from_elem((n, 1), 2.0),
        array![[1.0], [2.0], [3.0], [4.0]],
        Some(Array2::from_elem((n, 1), v)),
    );
    let params = ENNParams::new(1, 1.0, 0.2).unwrap();
    let p = on
        .model
        .posterior(&array![[2.0]].view(), &params, &PosteriorFlags::new())
        .unwrap();
    let ys = on.model.y_scale()[0];
    let per_copy = 0.2 + v / (ys * ys);
    let optimal_var = 1.0 / (n as f64 / per_copy);
    assert!((p.se_epi[[0, 0]] / ys - (EPS_VAR + optimal_var).sqrt()).abs() < 1e-12);
    assert!((p.mu[[0, 0]] - 2.5).abs() < 1e-12);
}

#[test]
fn unequal_yvar_reports_the_exact_variance_of_the_plain_mean() {
    let yv = [0.1, 0.2, 0.6];
    let on = build(
        true,
        Array2::from_elem((3, 1), 1.0),
        array![[1.0], [2.0], [6.0]],
        Some(array![[yv[0]], [yv[1]], [yv[2]]]),
    );
    let params = ENNParams::new(1, 1.0, 0.2).unwrap();
    let flags = PosteriorFlags::new().with_observation_noise(true);
    let p = on.model.posterior(&array![[1.0]].view(), &params, &flags).unwrap();
    let ys = on.model.y_scale()[0];
    let var_mean = 0.2 / 3.0 + (yv.iter().sum::<f64>() / (ys * ys)) / 9.0;
    assert!((p.se_epi[[0, 0]] / ys - (EPS_VAR + var_mean).sqrt()).abs() < 1e-12);
    assert!((p.mu[[0, 0]] - 3.0).abs() < 1e-12);
    let predictive = 0.2 + (yv.iter().sum::<f64>() / 3.0) / (ys * ys);
    assert!((p.se_ale[[0, 0]] / ys - predictive.sqrt()).abs() < 1e-12);
}

#[test]
fn predictive_noise_is_not_divided_by_the_count() {
    let on = build(true, Array2::from_elem((5, 1), 0.5), array![[1.0], [2.0], [3.0], [4.0], [5.0]], None);
    let params = ENNParams::new(1, 1.0, 0.4).unwrap();
    let flags = PosteriorFlags::new().with_observation_noise(true);
    let p = on.model.posterior(&array![[0.5]].view(), &params, &flags).unwrap();
    let ys = on.model.y_scale()[0];
    assert!((p.se_ale[[0, 0]] / ys - 0.4f64.sqrt()).abs() < 1e-12);
}

#[test]
fn reopening_rebuilds_the_rows_and_the_posterior() {
    let (x, y, _) = distinct_points(30);
    let mut b = build(true, x.clone(), y.clone(), None);
    b.model.add(&x.slice(s![..12, ..]), &y.slice(s![..12, ..]).mapv(|v| v + 1.0).view(), None).unwrap();
    b.model.add(&x.slice(s![..4, ..]), &y.slice(s![..4, ..]).mapv(|v| v - 2.0).view(), None).unwrap();
    let before = all_outputs(&b.model);
    let rows = b.model.num_rows();
    let Built { model, dir } = b;
    drop(model);
    let reopened = EpistemicNearestNeighbors::open(dir.path()).unwrap();
    assert!(reopened.aggregates_duplicates());
    assert_eq!((reopened.len(), reopened.num_rows()), (46, rows));
    assert_eq!(rows, 30);
    assert_eq!(all_outputs(&reopened), before);
}

#[test]
fn a_store_made_without_aggregation_reopens_without_it() {
    let (x, y, _) = distinct_points(10);
    let Built { model, dir } = build(false, x, y, None);
    drop(model);
    assert!(!EpistemicNearestNeighbors::open(dir.path()).unwrap().aggregates_duplicates());
}

#[test]
fn zero_and_negative_zero_are_the_same_x() {
    let b = build(true, array![[0.0, 1.0], [-0.0, 1.0], [2.0, 2.0]], array![[1.0], [3.0], [5.0]], None);
    assert_eq!(b.model.num_rows(), 2);
    assert_eq!(b.model.neighbor_count(0), 2.0);
    assert_eq!(b.model.neighbor_y(0).unwrap()[0], 2.0);
}

#[test]
fn means_are_taken_in_warped_space_per_metric() {
    let dir = TempDir::new().unwrap();
    let layout = EnnLayout::new(dir.path(), false).unwrap().with_aggregate_duplicates(true);
    let bounds = array![[0.0, 10.0], [f64::NEG_INFINITY, f64::INFINITY]];
    let model = EpistemicNearestNeighbors::new_with_storage(
        array![[1.0], [1.0], [1.0]],
        array![[1.0, 4.0], [5.0, 6.0], [9.0, 8.0]],
        None,
        layout,
        Some(bounds),
    )
    .unwrap();
    let stored: Vec<_> = (0..3).map(|i| model.rows().row_y(i).unwrap()).collect();
    let mean = model.neighbor_y(0).unwrap();
    for m in 0..2 {
        let expected = stored.iter().map(|r| r[m]).sum::<f64>() / 3.0;
        assert!((mean[m] - expected).abs() < 1e-12);
    }
    assert_ne!(mean[0], 5.0, "the mean is of warped values");
}

#[test]
fn many_duplicate_adds_leave_one_row() {
    let mut b = build(true, array![[0.0, 0.0]], array![[0.0]], None);
    let x = Array2::zeros((100_000, 2));
    let y = Array2::from_elem((100_000, 1), 2.0);
    b.model.add(&x.view(), &y.view(), None).unwrap();
    assert_eq!((b.model.len(), b.model.num_rows()), (100_001, 1));
    assert_eq!(b.model.neighbor_count(0), 100_001.0);
}
