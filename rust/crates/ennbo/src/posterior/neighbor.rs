//! Neighbor lookup helpers for posterior computation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::draw::NeighborData;
use crate::error::ENNError;
use crate::index::IndexDriver;
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
    let d2 = xx.view().insert_axis(Axis(1)).to_owned() + yy.view().insert_axis(Axis(0)).to_owned()
        - 2.0 * xy;
    d2.mapv(|v| v.max(0.0))
}

fn row_sq_l2(
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

fn dist2s_for_neighbor_indices(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    idx: &Array2<i64>,
) -> Array2<f64> {
    let train_x = model.train_x();
    let n_query = idx.nrows();
    let k = idx.ncols();
    let mut out = Array2::zeros((n_query, k));
    for i in 0..n_query {
        let x_row = x.row(i);
        for j in 0..k {
            let ni = idx[[i, j]];
            if ni < 0 {
                out[[i, j]] = f64::INFINITY;
            } else {
                let t_row = train_x.row(ni as usize);
                out[[i, j]] = row_sq_l2(x_row, t_row, model.scale_x, model.x_scale.view());
            }
        }
    }
    out
}

fn index_search(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    search_k: i32,
    exclude_nearest: bool,
) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
    model.ensure_index_sync()?;
    if model.index().driver() == IndexDriver::Exact {
        let train_x = model.train_x();
        let n_query = x.nrows();
        let n_train = train_x.nrows();
        let search_k = search_k as usize;
        let k = search_k.min(n_train);
        let mut dist2s = Array2::from_elem((n_query, search_k), f64::INFINITY);
        let mut idx = Array2::from_elem((n_query, search_k), -1i64);
        for i in 0..n_query {
            let x_row = x.row(i);
            let mut pairs: Vec<(f64, i64)> = (0..n_train)
                .map(|j| {
                    (
                        row_sq_l2(x_row, train_x.row(j), model.scale_x, model.x_scale.view()),
                        j as i64,
                    )
                })
                .collect();
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for j in 0..k {
                dist2s[[i, j]] = pairs[j].0;
                idx[[i, j]] = pairs[j].1;
            }
        }
        if exclude_nearest {
            let nc = dist2s.ncols();
            if nc <= 1 {
                dist2s = Array2::zeros((n_query, nc.saturating_sub(1)));
                idx = Array2::zeros((n_query, nc.saturating_sub(1)));
            } else {
                dist2s = dist2s
                    .slice_axis(Axis(1), ndarray::Slice::from(1..))
                    .to_owned();
                idx = idx
                    .slice_axis(Axis(1), ndarray::Slice::from(1..))
                    .to_owned();
            }
        }
        Ok((dist2s, idx))
    } else {
        let (_, idx) = model.index().search(x, search_k, exclude_nearest)?;
        let dist2s = dist2s_for_neighbor_indices(model, x, &idx);
        Ok((dist2s, idx))
    }
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

    let (dist2s_full, idx_full) = index_search(model, x, search_k as i32, exclude_nearest)?;

    let available_k = if exclude_nearest {
        search_k.saturating_sub(1)
    } else {
        search_k
    };
    let k = (params.k_num_neighbors as usize).min(available_k);

    if k == 0 {
        return Ok(None);
    }

    let dist2s = dist2s_full
        .slice_axis(Axis(1), ndarray::Slice::from(..k))
        .to_owned();
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

    let search_k =
        (params.k_num_neighbors as usize + if flags.exclude_nearest { 1 } else { 0 }).min(total_n);

    if search_k == 0 {
        return Ok(None);
    }

    let batch_size = x.nrows();
    let n_train = model.num_obs();
    let k = params.k_num_neighbors as usize;

    let train_search_k = search_k.min(n_train);
    let (dist2_train, idx_train) = if train_search_k > 0 {
        index_search(model, x, train_search_k as i32, false)?
    } else {
        (
            Array2::zeros((batch_size, 0)),
            ndarray::Array2::from_elem((batch_size, 0), -1i64),
        )
    };

    let dist2_whatif = if n_whatif > 0 {
        pairwise_sq_l2(x, x_whatif, model.scale_x, &model.x_scale.view())
    } else {
        Array2::zeros((batch_size, 0))
    };

    let n_candidates = dist2_train.ncols() + dist2_whatif.ncols();
    if n_candidates == 0 {
        return Ok(None);
    }

    if dist2_train.iter().any(|v| !v.is_finite()) || dist2_whatif.iter().any(|v| !v.is_finite()) {
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

    let dist2s = dist2_all
        .slice_axis(Axis(1), ndarray::Slice::from(..k_out))
        .to_owned();
    Ok(Some(NeighborData::new(dist2s, ids_all, y_neighbors, k_out)))
}

#[cfg(test)]
mod pairwise_tests {
    use super::pairwise_sq_l2;
    use ndarray::{array, Array2};

    #[test]
    fn index_search_returns_neighbors() {
        use super::index_search;
        use crate::index::IndexDriver;
        use crate::model::EpistemicNearestNeighbors;
        use ndarray::array;

        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.0, 0.0]];
        let (dist2s, idx) = index_search(&model, &query.view(), 2, false).unwrap();
        assert_eq!(idx[[0, 0]], 0);
        assert!(dist2s[[0, 0]] < 1e-6);
    }

    #[test]
    fn index_search_exclude_nearest_drops_self() {
        use super::index_search;
        use crate::index::IndexDriver;
        use crate::model::EpistemicNearestNeighbors;
        use ndarray::array;

        let train_x = array![[0.0], [1.0], [2.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[1.0]];
        let (dist2s, idx) = index_search(&model, &query.view(), 2, true).unwrap();
        assert_eq!(idx.ncols(), 1);
        assert_ne!(idx[[0, 0]], 1);
        assert!(dist2s[[0, 0]] > 0.0);
    }

    #[test]
    fn index_search_batch_matches_single_on_train_ties() {
        use super::index_search;
        use crate::index::IndexDriver;
        use crate::model::EpistemicNearestNeighbors;
        use ndarray::Array2;

        let train_x = Array2::from_shape_fn((20, 1), |(i, _)| {
            (i as f64 - 9.5) / 3.0 + 0.01 * (i as f64)
        });
        let train_y = Array2::from_shape_fn((20, 1), |(i, _)| {
            ((i as f64 + 1.0) * 0.37 - 2.1) * 100.0
        });
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let (dist2_batch, idx_batch) = index_search(&model, &train_x.view(), 10, false).unwrap();
        for i in 0..train_x.nrows() {
            let row = train_x.slice(ndarray::s![i..i + 1, ..]);
            let (dist2_one, idx_one) = index_search(&model, &row, 10, false).unwrap();
            assert_eq!(idx_batch.row(i).to_vec(), idx_one.row(0).to_vec());
            assert_eq!(dist2_batch.row(i).to_vec(), dist2_one.row(0).to_vec());
        }
    }

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

    #[test]
    fn dist2s_for_neighbor_indices_matches_pairwise_block() {
        use super::dist2s_for_neighbor_indices;
        use crate::index::IndexDriver;
        use crate::model::EpistemicNearestNeighbors;

        let train_x = array![[0.0], [1.0], [2.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.5]];
        let idx = array![[0i64, 1]];
        let dist2s = dist2s_for_neighbor_indices(&model, &query.view(), &idx);
        let block = pairwise_sq_l2(&query.view(), &train_x.view(), false, &array![1.0].view());
        assert!((dist2s[[0, 0]] - block[[0, 0]]).abs() < 1e-12);
        assert!((dist2s[[0, 1]] - block[[0, 1]]).abs() < 1e-12);
    }

    #[test]
    fn row_sq_l2_scaled_matches_pairwise() {
        use super::row_sq_l2;

        let x = array![2.0, 0.0];
        let y = array![0.0, 0.0];
        let scale = array![2.0, 1.0];
        let d = row_sq_l2(x.view(), y.view(), true, scale.view());
        assert!((d - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_sq_l2_scaled_differs_from_unscaled() {
        let x = array![[0.0, 0.0]];
        let y = array![[2.0, 0.0]];
        let scale = array![2.0, 1.0];
        let d0 = pairwise_sq_l2(&x.view(), &y.view(), false, &scale.view());
        let d1 = pairwise_sq_l2(&x.view(), &y.view(), true, &scale.view());
        assert!((d0[[0, 0]] - 4.0).abs() < 1e-12);
        assert!((d1[[0, 0]] - 1.0).abs() < 1e-12);
    }
}
