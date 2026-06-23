//! Neighbor lookup helpers for posterior computation.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use crate::draw::NeighborData;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

use super::neighbor_dist::{posterior_row_sq_l2, row_dist2s_for_query};
use super::tie_break::{
    finalize_faiss_pool_topk, topk_indices_from_row_dists, FaissPoolFinalizeCtx, PoolTieScratch,
    topk_indices_from_row_dists_with_buffers,
};

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

pub(crate) fn dist2s_for_neighbor_indices(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    idx: &Array2<i64>,
) -> Array2<f64> {
    let n_query = idx.nrows();
    let k = idx.ncols();
    let mut out = Array2::zeros((n_query, k));
    if let Some(train_x) = model.train_x_view_opt() {
        for i in 0..n_query {
            let x_row = x.row(i);
            for j in 0..k {
                let ni = idx[[i, j]];
                if ni < 0 {
                    out[[i, j]] = f64::INFINITY;
                } else {
                    let t_row = train_x.row(ni as usize);
                    out[[i, j]] = posterior_row_sq_l2(x_row, t_row, model.scale_x, model.x_scale.view());
                }
            }
        }
        return out;
    }
    let mut unique: Vec<usize> = Vec::new();
    for i in 0..n_query {
        for j in 0..k {
            let ni = idx[[i, j]];
            if ni >= 0 {
                let u = ni as usize;
                if !unique.contains(&u) {
                    unique.push(u);
                }
            }
        }
    }
    unique.sort_unstable();
    let mut row_map: std::collections::HashMap<usize, Array1<f64>> =
        std::collections::HashMap::with_capacity(unique.len());
    if !unique.is_empty() {
        let (train_x, _, _) = model
            .rows().train_rows_at(&unique)
            .expect("train_rows_at for neighbors");
        for (pos, &u) in unique.iter().enumerate() {
            row_map.insert(u, train_x.row(pos).to_owned());
        }
    }
    for i in 0..n_query {
        let x_row = x.row(i);
        for j in 0..k {
            let ni = idx[[i, j]];
            if ni < 0 {
                out[[i, j]] = f64::INFINITY;
            } else {
                let t_row = row_map
                    .get(&(ni as usize))
                    .expect("neighbor row")
                    .view();
                out[[i, j]] = posterior_row_sq_l2(x_row, t_row, model.scale_x, model.x_scale.view());
            }
        }
    }
    out
}

fn faiss_pairs_from_row(
    dist2s_faiss: &Array2<f64>,
    idx_faiss: &Array2<i64>,
    row: usize,
) -> Vec<(f64, i64)> {
    let mut pairs: Vec<(f64, i64)> = dist2s_faiss
        .row(row)
        .iter()
        .zip(idx_faiss.row(row).iter())
        .filter_map(|(&d, &t)| (t >= 0).then_some((d, t)))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    pairs
}

#[allow(dead_code)]
fn batched_row_dists_threshold(n_query: usize) -> usize {
    (n_query / 4).max(2)
}

#[allow(dead_code)]
pub(crate) fn use_matrix_topk_batch(n_query: usize, escalate_count: usize, tie_break_neighbors: bool) -> bool {
    // Full n×n distance matrix is slower than FAISS batch + in-pool tie resolution
    // on tie-heavy self-search (see KPop exp log 20260529).
    let _ = (n_query, escalate_count, tie_break_neighbors);
    false
}

pub(crate) fn exact_f64_batch_topk(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    search_k: i32,
    exclude_nearest: bool,
    tie_break_neighbors: bool,
) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
    let n_query = x.nrows();
    let n_train = model.num_obs();
    let search_k = search_k as usize;
    let k = search_k.min(n_train);
    if k == 0 {
        return Ok(apply_exclude_nearest(
            Array2::from_elem((n_query, search_k), f64::INFINITY),
            Array2::from_elem((n_query, search_k), -1i64),
            exclude_nearest,
        ));
    }
    let mut dist2s = Array2::from_elem((n_query, search_k), f64::INFINITY);
    let mut idx = Array2::from_elem((n_query, search_k), -1i64);
    let faiss_search_k = if tie_break_neighbors && k < n_train {
        search_k.max(k + 1).min(n_train)
    } else {
        search_k
    };

    let use_matrix_topk = false;

    if use_matrix_topk {
        let all_indices: Vec<usize> = (0..n_train).collect();
        let (train_x, _, _) = model.rows().train_rows_at(&all_indices).expect("train_rows_at");
        let dist_mat = pairwise_sq_l2(
            x,
            &train_x.view(),
            model.scale_x,
            &model.x_scale.view(),
        );
        let mut topk_scratch = Vec::with_capacity(n_train);
        let mut float_buf = Vec::with_capacity(n_train);
        for i in 0..n_query {
            let row = dist_mat.row(i);
            let row_slice = row.as_slice().expect("contiguous row");
            let best = topk_indices_from_row_dists_with_buffers(
                row_slice,
                k,
                true,
                &mut topk_scratch,
                &mut float_buf,
            );
            for (j, &t) in best.iter().enumerate() {
                dist2s[[i, j]] = row_slice[t];
                idx[[i, j]] = t as i64;
            }
        }
        return Ok(apply_exclude_nearest(dist2s, idx, exclude_nearest));
    }

    let (dist2s_search, idx_faiss) = model.backend_search(x, faiss_search_k as i32, false)?;
    let dist2s_faiss = if tie_break_neighbors {
        dist2s_for_neighbor_indices(model, x, &idx_faiss)
    } else {
        dist2s_search
    };

    let mut tie_scratch: PoolTieScratch = (Vec::new(), Vec::new());
    for i in 0..n_query {
        let mut pairs = faiss_pairs_from_row(&dist2s_faiss, &idx_faiss, i);
        let mut ctx = FaissPoolFinalizeCtx {
            precomputed_row_dists: None,
            faiss_pool_size: faiss_search_k,
            tie_scratch: &mut tie_scratch,
        };
        if finalize_faiss_pool_topk(
            model,
            x.row(i),
            &mut pairs,
            k,
            tie_break_neighbors,
            &mut ctx,
        ) {
            let row_dists_vec = row_dist2s_for_query(model, x.row(i));
            let best = topk_indices_from_row_dists(&row_dists_vec, k, true);
            for (j, &t) in best.iter().enumerate() {
                dist2s[[i, j]] = row_dists_vec[t];
                idx[[i, j]] = t as i64;
            }
            continue;
        }
        for (j, &(d, t)) in pairs.iter().enumerate() {
            dist2s[[i, j]] = d;
            idx[[i, j]] = t;
        }
    }
    Ok(apply_exclude_nearest(dist2s, idx, exclude_nearest))
}

fn apply_exclude_nearest(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    exclude_nearest: bool,
) -> (Array2<f64>, Array2<i64>) {
    if !exclude_nearest {
        return (dist2s, idx);
    }
    let n_query = dist2s.nrows();
    let nc = dist2s.ncols();
    if nc <= 1 {
        (
            Array2::zeros((n_query, nc.saturating_sub(1))),
            Array2::zeros((n_query, nc.saturating_sub(1))),
        )
    } else {
        (
            dist2s
                .slice_axis(Axis(1), ndarray::Slice::from(1..))
                .to_owned(),
            idx.slice_axis(Axis(1), ndarray::Slice::from(1..))
                .to_owned(),
        )
    }
}

pub(crate) fn get_neighbor_data(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    exclude_nearest: bool,
    tie_break_neighbors: bool,
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

    let (dist2s_full, idx_full) = super::index_search(
        model,
        x,
        search_k as i32,
        exclude_nearest,
        tie_break_neighbors,
    )?;

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


    for (i, idx_row) in idx.iter().enumerate().take(n_query) {
        let base_idx = i * k;
        for (j, &neighbor_idx) in idx_row.iter().enumerate() {
            let source_row = model.rows().row_y(neighbor_idx).expect("row_y");
            y_neighbors
                .row_mut(base_idx + j)
                .assign(&source_row.view());
        }
    }
    Ok(Some(NeighborData::new(dist2s, idx, y_neighbors, k)))
}

fn conditional_neighbor_batch_ready(
    dist2_train: &Array2<f64>,
    dist2_whatif: &Array2<f64>,
    batch_size: usize,
    n_candidates: usize,
) -> Result<Option<()>, ENNError> {
    if n_candidates == 0 || batch_size == 0 {
        return Ok(None);
    }
    if dist2_train.iter().any(|v| !v.is_finite()) || dist2_whatif.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "NaN or Inf in distance computation (check x, x_whatif for invalid values)".to_string(),
        ));
    }
    Ok(Some(()))
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
        super::index_search(model, x, train_search_k as i32, false, flags.tie_break_neighbors)?
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
    if conditional_neighbor_batch_ready(&dist2_train, &dist2_whatif, batch_size, n_candidates)?
        .is_none()
    {
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

    for (i, ids_row) in ids_all.iter().enumerate().take(batch_size) {
        let base_idx = i * k_out;
        for (j, &neighbor_idx) in ids_row.iter().enumerate() {
            let source_row = if neighbor_idx < n_train {
                model.rows().row_y(neighbor_idx).expect("row_y")
            } else {
                y_whatif.row(neighbor_idx - n_train).to_owned()
            };
            y_neighbors
                .row_mut(base_idx + j)
                .assign(&source_row.view());
        }
    }

    let dist2s = dist2_all
        .slice_axis(Axis(1), ndarray::Slice::from(..k_out))
        .to_owned();
    Ok(Some(NeighborData::new(dist2s, ids_all, y_neighbors, k_out)))
}

#[cfg(test)]
mod tests {
    use super::{
        apply_exclude_nearest, dist2s_for_neighbor_indices, exact_f64_batch_topk,
        faiss_pairs_from_row, get_conditional_neighbor_data, pairwise_sq_l2,
    };
    use super::super::neighbor_dist::posterior_row_sq_l2;
    use super::{batched_row_dists_threshold, use_matrix_topk_batch};
    use crate::index::IndexDriver;
    use crate::model::EpistemicNearestNeighbors;
    use ndarray::{array, Array2};

    #[test]
    fn faiss_pairs_from_row_and_threshold_helpers() {
        let dist2s = array![[0.0, 1.0], [2.0, 3.0]];
        let idx = array![[0i64, 1], [1, 0]];
        let pairs = faiss_pairs_from_row(&dist2s, &idx, 0);
        assert_eq!(pairs, vec![(0.0, 0), (1.0, 1)]);
        assert_eq!(batched_row_dists_threshold(1024), 256);
        assert_eq!(batched_row_dists_threshold(8), 2);
    }

    #[test]
    fn use_matrix_topk_batch_disabled_for_perf() {
        assert!(!use_matrix_topk_batch(1024, 1024, true));
        assert!(!use_matrix_topk_batch(1024, 300, false));
    }

    #[test]
    fn exact_f64_batch_topk_lattice_self_search_escalation_count() {
        let n = 1024usize;
        let d = 3usize;
        let k = 8usize;
        let mut train_x = Array2::zeros((n, d));
        train_x.column_mut(0).assign(&ndarray::Array1::from(
            (0..n)
                .map(|i| i as f64 / (n as f64 - 1.0))
                .collect::<Vec<_>>(),
        ));
        let train_y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let (dist2s, idx) =
            exact_f64_batch_topk(&model, &train_x.view(), k as i32, false, true).unwrap();
        assert_eq!(dist2s.nrows(), n);
        assert_eq!(idx.ncols(), k);
        // Brute top-k with index tie-break for a few spot rows
        for i in [0usize, 16, 500, n - 1] {
            let x_row = train_x.row(i);
            let mut ref_pairs: Vec<(f64, i64)> = (0..n)
                .map(|j| {
                    (
                        posterior_row_sq_l2(x_row, train_x.row(j), false, model.x_scale.view()),
                        j as i64,
                    )
                })
                .collect();
            ref_pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ref_pairs.truncate(k);
            let got: Vec<i64> = idx.row(i).iter().copied().collect();
            let want: Vec<i64> = ref_pairs.iter().map(|&(_, j)| j).collect();
            assert_eq!(got, want, "row {i}");
        }
    }

    #[test]
    fn exact_f64_batch_topk_matrix_path_on_lattice_self_search() {
        let n = 320usize;
        let d = 3usize;
        let k = 8usize;
        let mut train_x = Array2::zeros((n, d));
        train_x.column_mut(0).assign(&ndarray::Array1::from(
            (0..n).map(|i| i as f64 / (n as f64 - 1.0)).collect::<Vec<_>>(),
        ));
        let train_y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let (dist2s, idx) =
            exact_f64_batch_topk(&model, &train_x.view(), k as i32, false, true).unwrap();
        assert_eq!(dist2s.nrows(), n);
        assert_eq!(idx.ncols(), k);
        assert!(dist2s.iter().all(|v| v.is_finite()));
        assert!(idx.iter().all(|&v| v >= 0));
    }

    #[test]
    fn exact_f64_batch_topk_batch_matches_single_on_train_ties() {
        let train_x = Array2::from_shape_fn((20, 1), |(i, _)| {
            (i as f64 - 9.5) / 3.0 + 0.01 * (i as f64)
        });
        let train_y = Array2::from_shape_fn((20, 1), |(i, _)| {
            ((i as f64 + 1.0) * 0.37 - 2.1) * 100.0
        });
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let (dist2_batch, idx_batch) =
            exact_f64_batch_topk(&model, &train_x.view(), 10, false, true).unwrap();
        for i in 0..train_x.nrows() {
            let row = train_x.slice(ndarray::s![i..i + 1, ..]);
            let (dist2_one, idx_one) =
                exact_f64_batch_topk(&model, &row, 10, false, true).unwrap();
            assert_eq!(idx_batch.row(i).to_vec(), idx_one.row(0).to_vec());
            assert_eq!(dist2_batch.row(i).to_vec(), dist2_one.row(0).to_vec());
        }
    }

    #[test]
    fn exact_f64_batch_topk_tie_break_flag_noop_on_generic_data() {
        let train_x = Array2::from_shape_fn((20, 1), |(i, _)| {
            (i as f64 - 9.5) / 3.0 + 0.01 * (i as f64)
        });
        let train_y = Array2::from_shape_fn((20, 1), |(i, _)| {
            ((i as f64 + 1.0) * 0.37 - 2.1) * 100.0
        });
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let k = 10usize;
        let (_dist2_on, idx_on) =
            exact_f64_batch_topk(&model, &train_x.view(), k as i32, false, true).unwrap();
        let _ = exact_f64_batch_topk(&model, &train_x.view(), k as i32, false, false).unwrap();
        for i in 0..train_x.nrows() {
            let x_row = train_x.row(i);
            let mut ref_pairs: Vec<(f64, i64)> = (0..train_x.nrows())
                .map(|j| {
                    (
                        posterior_row_sq_l2(
                            x_row,
                            train_x.row(j),
                            false,
                            model.x_scale.view(),
                        ),
                        j as i64,
                    )
                })
                .collect();
            ref_pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            ref_pairs.truncate(k);
            let ref_on: Vec<i64> = ref_pairs.iter().map(|&(_, j)| j).collect();
            assert_eq!(idx_on.row(i).to_vec(), ref_on);
        }
    }

    #[test]
    fn exact_f64_batch_topk_exclude_nearest_strips_self() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.0, 0.0]];
        let (dist2s, idx) = exact_f64_batch_topk(&model, &query.view(), 2, true, true).unwrap();
        assert_eq!(dist2s.shape(), [1, 1]);
        assert_eq!(idx.shape(), [1, 1]);
        assert!(dist2s[[0, 0]] > 0.0);
        assert_eq!(idx[[0, 0]], 1);
    }

    #[test]
    fn apply_exclude_nearest_slices_or_zeros() {
        let dist2s = array![[0.0, 1.0], [2.0, 3.0]];
        let idx = array![[0i64, 1], [1, 0]];
        let (d, i) = apply_exclude_nearest(dist2s.clone(), idx.clone(), true);
        assert_eq!(d.shape(), [2, 1]);
        assert_eq!(i[[0, 0]], 1);
        let (d0, i0) = apply_exclude_nearest(
            Array2::from_elem((1, 1), 0.0),
            Array2::from_elem((1, 1), 0i64),
            true,
        );
        assert_eq!(d0.shape(), [1, 0]);
        assert_eq!(i0.shape(), [1, 0]);
        let (d2, i2) = apply_exclude_nearest(dist2s.clone(), idx.clone(), false);
        assert_eq!(d2, dist2s);
        assert_eq!(i2, idx);
    }

    #[test]
    fn exact_f64_batch_topk_exclude_nearest_k_one_yields_empty() {
        let train_x = array![[0.0]];
        let train_y = array![[0.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let (dist2s, idx) = exact_f64_batch_topk(&model, &train_x.view(), 1, true, false).unwrap();
        assert_eq!(dist2s.shape(), [1, 0]);
        assert_eq!(idx.shape(), [1, 0]);
    }

    #[test]
    fn exact_f64_batch_topk_tie_break_on_prefers_lower_index_at_cutoff() {
        let train_x = array![[0.0], [0.0], [1.0], [2.0]];
        let train_y = array![[0.0], [1.0], [2.0], [3.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.0]];
        let (_, idx_on) = exact_f64_batch_topk(&model, &query.view(), 2, false, true).unwrap();
        assert_eq!(idx_on.row(0).to_vec(), vec![0, 1]);
    }

    #[test]
    fn exact_f64_batch_topk_auto_tie_break_expands_when_more_than_k_at_cutoff() {
        let train_x = array![[0.0], [0.0], [0.0], [1.0]];
        let train_y = array![[0.0], [1.0], [2.0], [3.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.0]];
        let (_, idx_on) = exact_f64_batch_topk(&model, &query.view(), 2, false, true).unwrap();
        assert_eq!(idx_on.row(0).to_vec(), vec![0, 1]);
    }

    #[test]
    fn index_search_hnsw_refines_distances() {
        use super::super::index_search;

        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::HNSW)
                .unwrap();
        let query = array![[0.0, 0.0]];
        let (dist2s, idx) = index_search(&model, &query.view(), 2, false, true).unwrap();
        assert_eq!(idx[[0, 0]], 0);
        assert!(dist2s[[0, 0]] < 1e-6);
    }

    #[test]
    fn get_conditional_neighbor_data_empty_batch() {
        use crate::params::{ENNParams, PosteriorFlags};
        use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;

        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let empty_x: Array2<f64> = Array2::zeros((0, 2));
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[1.5]];
        let result = get_conditional_neighbor_data(
            &model,
            &empty_x.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
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
