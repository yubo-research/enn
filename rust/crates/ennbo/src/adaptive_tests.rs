//! Behavioral contracts for native adaptation and learned output mixing.
use crate::adaptive::AdaptiveSemanticENN;
use crate::adaptive_basis::{Basis, Metadata};
use crate::adaptive_geometry::{fixed_rank, quadratic, ridge};
use crate::adaptive_predictor::{Choice, Predictor};
use crate::adaptive_scales::{bake, centered, distance_scales};
use crate::adaptive_selection::{fit_epoch, mixing};
use crate::posterior::neighbor_dist::posterior_row_sq_l2;
use ndarray::{array, s, Array2, Axis};

fn observations() -> (Array2<f64>, Array2<f64>) {
    let x = Array2::from_shape_fn((60, 3), |(i, j)| ((i * 7 + j * 11) as f64 * 0.1).sin());
    let y = Array2::from_shape_fn((60, 3), |(i, j)| {
        if j == 2 {
            3.0
        } else {
            x[[i, 0]] * if j == 0 { 1.0 } else { -1.0 }
        }
    });
    (x, y)
}
fn metadata() -> Metadata {
    Metadata {
        output: Some(array![[0.5_f64.sqrt()], [0.5_f64.sqrt()], [0.0]]),
        ..Default::default()
    }
}

#[test]
fn calibration_labels_cannot_change_choice_or_fitted_mean() {
    let (x, y) = observations();
    let mut changed = y.clone();
    changed.slice_mut(s![48.., ..]).mapv_inplace(|v| v + 100.0);
    let model = fit_epoch(&x, &y, &metadata(), false, "all", None, None).unwrap();
    let other = fit_epoch(&x, &changed, &metadata(), false, "all", None, None).unwrap();
    assert_eq!(model.predictor.choice, other.predictor.choice);
    assert_eq!(
        model.predictor.posterior(&x).unwrap().0,
        other.predictor.posterior(&x).unwrap().0
    );
    assert!(other.calibration.sum() > model.calibration.sum() * 10.0);
}

#[test]
fn continuous_mixing_recovers_interior_and_boundary_optima() {
    let mean = array![[1., 2.], [3., 4.]];
    let projector = array![[0.5, 0.5], [0.5, 0.5]];
    for alpha in [0.0, 0.1, 0.3, 1.0] {
        let target = &mean + &(mean.dot(&projector) - &mean) * alpha;
        assert!((mixing(&mean, &target, &projector) - alpha).abs() < 1e-12);
    }
    assert_eq!(mixing(&mean, &(-&mean), &Array2::zeros((2, 2))), 1.0);
    assert_eq!(mixing(&mean, &(&mean * 2.0), &Array2::zeros((2, 2))), 0.0);
    assert_eq!(mixing(&mean, &mean, &Array2::eye(2)), 0.0);
}

#[test]
fn geometry_keeps_metadata_width_and_pairwise_distances() {
    let (x, y) = observations();
    for rank in [0, 1, 2] {
        let input = Array2::from_shape_fn((3, rank), |(j, k)| if j == k { 1.0 } else { 0.0 });
        let metadata = Metadata {
            bounds: Some(array![[0., 1.], [0., 1.], [0., 1.]]),
            input: Some(input.clone()),
            output: None,
        };
        let basis = Basis::new(&x, &y, &metadata).unwrap();
        assert_eq!(basis.linear.dim(), (3, rank));
        assert_eq!(basis.quadratic.dim(), (3, rank));
        for geometry in ["fixed", "linear", "quadratic"] {
            let features = basis.features(&x, geometry, 3.0).unwrap();
            let (z, direction, weight) = match geometry {
                "fixed" => (x.clone(), &input, 1.0),
                "linear" => ((&x - &basis.x_mean) / &basis.x_scale, &basis.linear, 3.0),
                _ => ((&x - &basis.x_mean) / &basis.x_scale, &basis.quadratic, 3.0),
            };
            let delta = &z.row(1) - &z.row(0);
            let semantic = delta.dot(direction);
            let expected = delta.dot(&delta) + weight * semantic.dot(&semantic);
            let actual = &features.row(1) - &features.row(0);
            assert!((actual.dot(&actual) - expected).abs() < 1e-10);
        }
    }
    assert_eq!(fixed_rank(Array2::eye(3), Some(0)).unwrap().dim(), (3, 0));
}

#[test]
fn quadratic_map_preserves_gradient_moment() {
    let (z, target) = observations();
    let design =
        ndarray::concatenate(Axis(1), &[z.view(), z.mapv(|v| v * v - 1.0).view()]).unwrap();
    let coefficients = ridge(&design, &target).unwrap();
    let mut moment = Array2::<f64>::zeros((3, 3));
    for row in z.rows() {
        let gradient = Array2::from_shape_fn((3, 3), |(j, m)| {
            coefficients[[j, m]] + 2.0 * row[j] * coefficients[[3 + j, m]]
        });
        moment += &gradient.dot(&gradient.t());
    }
    moment /= z.nrows() as f64;
    let direction = quadratic(&z, &target).unwrap();
    assert!(direction
        .dot(&direction.t())
        .iter()
        .zip(moment.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10));
}

#[test]
fn failed_updates_leave_replay_predictions_and_random_state_intact() {
    let (x, y) = observations();
    let mut left = AdaptiveSemanticENN::new(metadata(), 30, 3, false, "all".into()).unwrap();
    let mut right = AdaptiveSemanticENN::new(metadata(), 30, 3, false, "all".into()).unwrap();
    for model in [&mut left, &mut right] {
        model.add(x.clone(), y.clone(), None).unwrap();
    }
    for weight in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(left.add(x.clone(), y.clone(), Some(weight)).is_err());
    }
    assert!(left.add(x.clone(), Array2::zeros((60, 2)), None).is_err());
    for model in [&mut left, &mut right] {
        model.add(x.clone(), y.clone(), Some(30.0)).unwrap();
    }
    assert_eq!(left.num_seen, 120);
    assert_eq!(left.num_retained(), 30);
    assert_eq!(left.choice(), right.choice());
    assert_eq!(
        left.posterior(&x).unwrap().0,
        right.posterior(&x).unwrap().0
    );
}

#[test]
fn column_scales_store_factors_and_reject_failed_batches() {
    let x = Array2::from_shape_fn((16, 2), |(i, j)| (i + j) as f64 / 20.0);
    let y = x.column(0).insert_axis(Axis(1)).mapv(|v| (v - 0.5).sin());
    let projection = array![[1.0], [-1.0]];
    let metadata = Metadata {
        bounds: Some(array![[0.0, 1.0], [0.0, 1.0]]),
        input: Some(projection.clone()),
        output: None,
    };
    let z = centered(&x, metadata.bounds.as_ref().unwrap());
    let factors = z.dot(&projection);
    let mut basis = Basis::new(&x, &y, &metadata).unwrap();
    basis.column_scale = array![2.0];
    let stored = basis.features(&x, "scales", 1.0).unwrap();
    let joined = ndarray::concatenate(Axis(1), &[z.view(), factors.view()]).unwrap();
    assert!(stored
        .iter()
        .zip(joined.iter())
        .all(|(left, right)| (left - right).abs() < 1e-12));
    let mut baked = stored.clone();
    bake(&mut baked, &basis.column_scale);
    let delta = &baked.row(0) - &baked.row(1);
    let dz = &z.row(0) - &z.row(1);
    let ds = (&factors.row(0) - &factors.row(1)) * 2.0;
    let expected = dz.dot(&dz) + ds.dot(&ds);
    assert!((delta.dot(&delta) - expected).abs() < 1e-10);
    let predictor = Predictor::new(
        &x,
        &y,
        &metadata,
        Choice {
            geometry: "scales".into(),
            weight: 1.0,
            output: "independent".into(),
            neighbors: 3,
            alpha: 0.0,
        },
        false,
        Some(&basis.column_scale),
    )
    .unwrap();
    let rows = predictor.model.train_x_view_opt().unwrap().to_owned();
    assert!(rows
        .iter()
        .zip(stored.iter())
        .all(|(left, right)| (left - right).abs() < 1e-12));
    let scale = distance_scales(&basis.column_scale, 2);
    assert!((predictor.model.x_scale[2] - scale[2]).abs() < 1e-12);
    let distance = posterior_row_sq_l2(
        stored.row(0),
        stored.row(1),
        true,
        predictor.model.x_scale.view(),
    );
    assert!((distance - expected).abs() < 1e-8);
    let mut changed = predictor;
    crate::adaptive_predictor::install_column_scale(&mut changed.model, &array![0.25]).unwrap();
    assert_eq!(changed.model.train_x_view_opt().unwrap(), rows.view());
    assert!((changed.model.x_scale[2] - 4.0).abs() < 1e-12);

    let mut model = AdaptiveSemanticENN::new(metadata, 30, 1, false, "inputs".into()).unwrap();
    model.add(x.clone(), y.clone(), None).unwrap();
    let before = model.column_scale().unwrap();
    assert!(model.add(x.clone(), Array2::zeros((16, 2)), None).is_err());
    assert_eq!(model.column_scale().unwrap(), before);
}

#[test]
fn constant_input_uses_unit_scale_at_query() {
    let x = Array2::from_shape_fn((30, 2), |(i, j)| if j == 0 { i as f64 / 30.0 } else { 0.0 });
    let y = x.slice(s![.., ..1]).mapv(f64::sin);
    let basis = Basis::new(&x, &y, &Metadata::default()).unwrap();
    assert_eq!(basis.x_scale[1], 1.0);
    for geometry in ["linear", "quadratic"] {
        let features = basis
            .features(&array![[0.5, 0.], [0.5, 1.]], geometry, 3.0)
            .unwrap();
        let delta = &features.row(1) - &features.row(0);
        assert!((delta.dot(&delta) - 1.0).abs() < 1e-10);
    }
}
