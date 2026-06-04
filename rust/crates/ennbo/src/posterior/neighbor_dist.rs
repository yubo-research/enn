use ndarray::ArrayView1;

use crate::model::EpistemicNearestNeighbors;

pub(crate) fn row_sq_l2(
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
            row_sq_l2(x_row, t.view(), model.scale_x, model.x_scale.view())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::row_sq_l2;
    use ndarray::array;

    #[test]
    fn row_sq_l2_scaled_matches_pairwise() {
        let x = array![2.0, 0.0];
        let y = array![0.0, 0.0];
        let scale = array![2.0, 1.0];
        let d = row_sq_l2(x.view(), y.view(), true, scale.view());
        assert!((d - 1.0).abs() < 1e-12);
    }
}
