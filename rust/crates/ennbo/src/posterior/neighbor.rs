//! Neighbor lookup helpers for posterior computation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::draw::NeighborData;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

fn pairwise_sq_l2(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    scale: bool,
    x_scale: &ArrayView1<f64>,
) -> Array2<f64> {
    let (x_s, y_s) = if scale {
        (
            x / &x_scale.view().insert_axis(Axis(0)),
            y / &x_scale.view().insert_axis(Axis(0)),
        )
    } else {
        (x.to_owned(), y.to_owned())
    };
    let xx: Array1<f64> = x_s.map_axis(Axis(1), |r| r.dot(&r));
    let yy: Array1<f64> = y_s.map_axis(Axis(1), |r| r.dot(&r));
    let xy = x_s.dot(&y_s.t());
    let d2 = xx.view().insert_axis(Axis(1)).to_owned()
        + yy.view().insert_axis(Axis(0)).to_owned()
        - 2.0 * xy;
    d2.mapv(|v| v.max(0.0))
}

pub(crate) fn get_neighbor_data(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    exclude_nearest: bool,
) -> Result<Option<NeighborData>, ENNError> {
    if exclude_nearest && model.num_obs() <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {}",
            model.num_obs()
        )));
    }
    let search_k = if exclude_nearest {
        (params.k_num_neighbors as usize + 1).min(model.num_obs())
    } else {
        (params.k_num_neighbors as usize).min(model.num_obs())
    };

    if search_k == 0 {
        return Ok(None);
    }

    let (dist2s_full, idx_full) = model.index().search(x, search_k as i32, exclude_nearest)?;

    let available_k = if exclude_nearest {
        search_k.saturating_sub(1)
    } else {
        search_k
    };
    let k = (params.k_num_neighbors as usize).min(available_k);

    if k == 0 {
        return Ok(None);
    }

    let dist2s = dist2s_full.slice_axis(Axis(1), ndarray::Slice::from(..k)).to_owned();
    let idx: Vec<Vec<usize>> = idx_full
        .slice_axis(Axis(1), ndarray::Slice::from(..k))
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&i| i as usize).collect())
        .collect();

    let n_query = x.nrows();
    let num_metrics = model.num_metrics();
    let mut y_neighbors = Array2::zeros((n_query * k, num_metrics));

    let train_y = model.train_y();
    for (i, idx_row) in idx.iter().enumerate().take(n_query) {
        let base_idx = i * k;
        for (j, &neighbor_idx) in idx_row.iter().enumerate() {
            let mut target_row = y_neighbors.row_mut(base_idx + j);
            let source_row = train_y.row(neighbor_idx);
            target_row.assign(&source_row);
        }
    }
    Ok(Some(NeighborData::new(dist2s, idx, y_neighbors, k)))
}

pub(crate) fn get_conditional_neighbor_data(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<Option<NeighborData>, ENNError> {
    let n_whatif = x_whatif.nrows();
    let total_n = model.num_obs() + n_whatif;

    if flags.exclude_nearest && total_n <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {}",
            total_n
        )));
    }

    let search_k = (params.k_num_neighbors as usize
        + if flags.exclude_nearest { 1 } else { 0 })
        .min(total_n);

    if search_k == 0 {
        return Ok(None);
    }

    let batch_size = x.nrows();
    let n_train = model.num_obs();
    let k = params.k_num_neighbors as usize;

    let train_search_k = search_k.min(n_train);
    let (dist2_train, idx_train) = if train_search_k > 0 {
        model.index().search(x, train_search_k as i32, false)?
    } else {
        (
            Array2::zeros((batch_size, 0)),
            ndarray::Array2::from_elem((batch_size, 0), -1i64),
        )
    };

    let dist2_whatif = if n_whatif > 0 {
        pairwise_sq_l2(
            x,
            x_whatif,
            model.scale_x,
            &model.x_scale.view(),
        )
    } else {
        Array2::zeros((batch_size, 0))
    };

    let n_candidates = dist2_train.ncols() + dist2_whatif.ncols();
    if n_candidates == 0 {
        return Ok(None);
    }

    if dist2_train.iter().any(|v| !v.is_finite())
        || dist2_whatif.iter().any(|v| !v.is_finite())
    {
        return Err(ENNError::InvalidParameter(
            "NaN or Inf in distance computation (check x, x_whatif for invalid values)".to_string(),
        ));
    }

    if batch_size == 0 {
        return Ok(None);
    }

    let mut dist2_all = Array2::zeros((batch_size, n_candidates));
    let mut ids_all: Vec<Vec<usize>> = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_candidates);
        for j in 0..dist2_train.ncols() {
            combined.push((dist2_train[[i, j]], idx_train[[i, j]] as usize));
        }
        for j in 0..dist2_whatif.ncols() {
            combined.push((dist2_whatif[[i, j]], n_train + j));
        }

        let sel_end = search_k.min(combined.len());
        if sel_end > 0 && sel_end < combined.len() {
            combined.select_nth_unstable_by(sel_end - 1, |a, b| a.0.total_cmp(&b.0));
            combined[..sel_end].sort_by(|a, b| a.0.total_cmp(&b.0));
        } else {
            combined.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        let sel: Vec<(f64, usize)> = if flags.exclude_nearest && sel_end > 0 {
            combined[1..sel_end].to_vec()
        } else {
            combined[..sel_end].to_vec()
        };
        let k_actual = k.min(sel.len());
        let sel = &sel[..k_actual];

        let mut row_ids = Vec::with_capacity(k_actual);
        for (col, &(d, idx)) in sel.iter().enumerate() {
            dist2_all[[i, col]] = d;
            row_ids.push(idx);
        }
        ids_all.push(row_ids);
    }

    let k_out = ids_all[0].len();
    if k_out == 0 {
        return Ok(None);
    }

    let num_metrics = model.num_metrics();
    let mut y_neighbors = Array2::zeros((batch_size * k_out, num_metrics));

    let train_y = model.train_y();
    for (i, ids_row) in ids_all.iter().enumerate().take(batch_size) {
        let base_idx = i * k_out;
        for (j, &neighbor_idx) in ids_row.iter().enumerate() {
            let mut target_row = y_neighbors.row_mut(base_idx + j);
            let source_row = if neighbor_idx < n_train {
                train_y.row(neighbor_idx)
            } else {
                y_whatif.row(neighbor_idx - n_train)
            };
            target_row.assign(&source_row);
        }
    }

    let dist2s = dist2_all.slice_axis(Axis(1), ndarray::Slice::from(..k_out)).to_owned();
    Ok(Some(NeighborData::new(dist2s, ids_all, y_neighbors, k_out)))
}

#[cfg(test)]
mod pairwise_tests {
    use ndarray::{array, Array2};
    use super::pairwise_sq_l2;

    #[test]
    fn pairwise_sq_l2_scaled_and_unscaled() {
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0, 1.0], [1.0, 1.0]];
        let scale = array![1.0, 2.0];
        let d_unscaled = pairwise_sq_l2(&x.view(), &y.view(), false, &scale.view());
        assert_eq!(d_unscaled.nrows(), 2);
        assert_eq!(d_unscaled.ncols(), 2);
        let d_scaled = pairwise_sq_l2(&x.view(), &y.view(), true, &scale.view());
        assert_eq!(d_scaled.shape(), d_unscaled.shape());
        assert!(d_scaled.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn pairwise_sq_l2_single_row_each() {
        let x = array![[3.0, 4.0]];
        let y = array![[0.0, 0.0]];
        let scale = array![1.0, 1.0];
        let d = pairwise_sq_l2(&x.view(), &y.view(), false, &scale.view());
        assert_eq!(d.nrows(), 1);
        assert_eq!(d.ncols(), 1);
        assert!((d[[0, 0]] - 25.0).abs() < 1e-9);
    }

    #[test]
    fn pairwise_sq_l2_empty_y_cols() {
        let x = array![[1.0, 2.0]];
        let y = Array2::<f64>::zeros((1, 2));
        let scale = array![1.0, 1.0];
        let d = pairwise_sq_l2(&x.view(), &y.view(), false, &scale.view());
        assert_eq!(d[[0, 0]], 5.0);
    }
}
