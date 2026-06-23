use ndarray::ArrayView1;

use crate::model::EpistemicNearestNeighbors;

pub fn posterior_row_sq_l2(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    scale_x: bool,
    x_scale: ArrayView1<f64>,
) -> f64 {
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
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
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
