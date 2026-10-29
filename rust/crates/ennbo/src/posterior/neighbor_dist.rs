use ndarray::ArrayView1;

#[cfg(test)]
use crate::model::EpistemicNearestNeighbors;

pub fn posterior_row_sq_l2(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    scale_x: bool,
    x_scale: ArrayView1<f64>,
) -> f64 {
    if let (Some(x), Some(y), Some(scale)) = (x.as_slice(), y.as_slice(), x_scale.as_slice()) {
        return posterior_slice_sq_l2(x, y, scale_x.then_some(scale));
    }
    let mut acc = 0.0;
    if scale_x {
        for i in 0..x.len() {
            let sc = x_scale[i];
            let d = x[i] / sc - y[i] / sc;
            acc += d * d;
        }
    } else {
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let d = xi - yi;
            acc += d * d;
        }
    }
    acc.max(0.0)
}

#[inline]
pub(super) fn posterior_slice_sq_l2(x: &[f64], y: &[f64], scale: Option<&[f64]>) -> f64 {
    let mut distance = 0.0;
    if let Some(scale) = scale {
        for ((&x, &y), &s) in x.iter().zip(y).zip(scale) {
            let delta = x / s - y / s;
            distance += delta * delta;
        }
    } else {
        for (&x, &y) in x.iter().zip(y) {
            let delta = x - y;
            distance += delta * delta;
        }
    }
    distance.max(0.0)
}

#[cfg(test)]
pub(crate) fn row_dist2s_for_query(
    model: &EpistemicNearestNeighbors,
    x_row: ArrayView1<f64>,
) -> Vec<f64> {
    let n_train = model.num_obs();
    (0..n_train)
        .map(|j| {
            let t = model.rows().row_x(j).expect("row_x");
            posterior_row_sq_l2(x_row, t.view(), model.scale_x, model.x_scale.view())
        })
        .collect()
}

#[cfg(test)]
mod kiss_coverage_tests {
    use crate::model::EpistemicNearestNeighbors;
    use crate::IndexDriver;
    use ndarray::array;

    #[test]
    fn neighbor_dist_units_are_linked() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
                .unwrap();
        let dists = crate::posterior::neighbor_dist::row_dist2s_for_query(
            &model,
            array![0.5, 0.0].view(),
        );
        assert_eq!(dists.len(), 2);
        let d = crate::posterior::neighbor_dist::posterior_row_sq_l2(
            array![0.0, 0.0].view(),
            array![1.0, 0.0].view(),
            false,
            array![1.0, 1.0].view(),
        );
        assert!((d - 1.0).abs() < 1e-12);
    }
}
