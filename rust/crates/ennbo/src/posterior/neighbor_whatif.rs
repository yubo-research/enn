use std::collections::HashMap;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::neighbor::{
    conditional_neighbor_batch_ready, index_search, pairwise_sq_l2, SELF_DIST_EPS,
};
use crate::draw::NeighborData;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

struct WhatifSites {
    new_x: Array2<f64>,
    new_y: Array2<f64>,
    new_count: Vec<f64>,
    merged: HashMap<usize, (Array1<f64>, f64)>,
}

fn x_key(row: ArrayView1<f64>) -> Vec<u64> {
    row.iter()
        .map(|&v| {
            if v == 0.0 {
                0.0f64.to_bits()
            } else {
                v.to_bits()
            }
        })
        .collect()
}

type SiteGroups = Vec<(Vec<usize>, Option<usize>)>;

fn group_rows_by_site(
    model: &EpistemicNearestNeighbors,
    x_whatif: &ArrayView2<f64>,
) -> Result<SiteGroups, ENNError> {
    let aggregate = model.aggregates_duplicates();
    let mut order: SiteGroups = Vec::new();
    let mut by_key: HashMap<Vec<u64>, usize> = HashMap::new();
    for (j, row) in x_whatif.rows().into_iter().enumerate() {
        if !aggregate {
            order.push((vec![j], None));
            continue;
        }
        let key = x_key(row);
        if let Some(&g) = by_key.get(&key) {
            order[g].0.push(j);
            continue;
        }
        let site = model
            .rows()
            .find_site(row.as_standard_layout().as_slice().expect("contiguous row"))?;
        by_key.insert(key, order.len());
        order.push((vec![j], site));
    }
    Ok(order)
}

fn group_whatif_sites(
    model: &EpistemicNearestNeighbors,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
) -> Result<WhatifSites, ENNError> {
    let num_metrics = y_whatif.ncols();
    let mut merged = HashMap::new();
    let mut new_count = Vec::new();
    let mut x_rows: Vec<usize> = Vec::new();
    let mut y_means: Vec<Array1<f64>> = Vec::new();
    for (members, site) in group_rows_by_site(model, x_whatif)? {
        let m = members.len() as f64;
        let mut y_sum = Array1::<f64>::zeros(num_metrics);
        for &j in &members {
            y_sum += &y_whatif.row(j);
        }
        if let Some(rep) = site {
            let n0 = model.neighbor_count(rep);
            let y0 = model.neighbor_y(rep)?;
            merged.insert(rep, ((y0 * n0 + y_sum) / (n0 + m), n0 + m));
        } else {
            x_rows.push(members[0]);
            y_means.push(y_sum / m);
            new_count.push(m);
        }
    }
    Ok(WhatifSites {
        new_x: x_whatif.select(Axis(0), &x_rows),
        new_y: Array2::from_shape_fn((y_means.len(), num_metrics), |(i, j)| y_means[i][j]),
        new_count,
        merged,
    })
}

fn select_row_ids(
    mut combined: Vec<(f64, usize)>,
    search_k: usize,
    k: usize,
    exclude_nearest: bool,
) -> Vec<(f64, usize)> {
    let sel_end = search_k.min(combined.len());
    if sel_end > 0 && sel_end < combined.len() {
        combined.select_nth_unstable_by(sel_end - 1, |a, b| a.0.total_cmp(&b.0));
        combined[..sel_end].sort_by(|a, b| a.0.total_cmp(&b.0));
    } else {
        combined.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let start = usize::from(
        exclude_nearest
            && sel_end > 0
            && combined[0].0.is_finite()
            && combined[0].0 <= SELF_DIST_EPS,
    );
    let sel = &combined[start..sel_end];
    sel[..k.min(sel.len())].to_vec()
}

fn gather_neighbors(
    model: &EpistemicNearestNeighbors,
    sites: &WhatifSites,
    ids_all: &[Vec<usize>],
    k_out: usize,
) -> (Array2<f64>, Vec<f64>) {
    let n_train = model.num_obs();
    let mut y_neighbors = Array2::zeros((ids_all.len() * k_out, model.num_metrics()));
    let mut counts: Vec<f64> = Vec::with_capacity(ids_all.len() * k_out);
    for (i, ids_row) in ids_all.iter().enumerate() {
        for (j, &id) in ids_row.iter().enumerate() {
            let (y, count) = if id >= n_train {
                (
                    sites.new_y.row(id - n_train).to_owned(),
                    sites.new_count[id - n_train],
                )
            } else if let Some((y, c)) = sites.merged.get(&id) {
                (y.clone(), *c)
            } else {
                (
                    model.neighbor_y(id).expect("row_y"),
                    model.neighbor_count(id),
                )
            };
            counts.push(count);
            y_neighbors.row_mut(i * k_out + j).assign(&y.view());
        }
    }
    (y_neighbors, counts)
}

pub(crate) fn get_conditional_neighbor_data(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<Option<NeighborData>, ENNError> {
    let sites = group_whatif_sites(model, x_whatif, y_whatif)?;
    let n_new = sites.new_x.nrows();
    let n_train = model.num_obs();
    let n_train_sites = if model.aggregates_duplicates() {
        model.num_rows()
    } else {
        n_train
    };
    let total_n = n_train_sites + n_new;

    if flags.exclude_nearest && total_n <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {}",
            total_n
        )));
    }

    let k = params.k_num_neighbors as usize;
    let search_k = (k + usize::from(flags.exclude_nearest)).min(total_n);
    if search_k == 0 {
        return Ok(None);
    }

    let batch_size = x.nrows();
    let train_search_k = search_k.min(n_train);
    let (dist2_train, idx_train) = if train_search_k > 0 {
        index_search(model, x, train_search_k as i32, false)?
    } else {
        (
            Array2::zeros((batch_size, 0)),
            Array2::from_elem((batch_size, 0), -1i64),
        )
    };
    let dist2_whatif = if n_new > 0 {
        pairwise_sq_l2(x, &sites.new_x.view(), model.scale_x, &model.x_scale.view())
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
            if idx_train[[i, j]] >= 0 {
                combined.push((dist2_train[[i, j]], idx_train[[i, j]] as usize));
            }
        }
        for j in 0..dist2_whatif.ncols() {
            combined.push((dist2_whatif[[i, j]], n_train + j));
        }
        let sel = select_row_ids(combined, search_k, k, flags.exclude_nearest);
        for (col, &(d, _)) in sel.iter().enumerate() {
            dist2_all[[i, col]] = d;
        }
        ids_all.push(sel.iter().map(|&(_, id)| id).collect());
    }

    let k_out = ids_all.iter().map(Vec::len).min().unwrap_or(0);
    if k_out == 0 {
        return Ok(None);
    }
    for ids in &mut ids_all {
        ids.truncate(k_out);
    }
    let (y_neighbors, counts) = gather_neighbors(model, &sites, &ids_all, k_out);
    let dist2s = dist2_all
        .slice_axis(Axis(1), ndarray::Slice::from(..k_out))
        .to_owned();
    let mut data = NeighborData::new(dist2s, ids_all, y_neighbors, k_out);
    if model.aggregates_duplicates() {
        data.counts = Some(counts);
    }
    Ok(Some(data))
}
