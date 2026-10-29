//! BPANN search at every N: indexed tree plus pending mmap rows.

use ndarray::Array2;
use rayon::prelude::*;

use crate::backend::BpannBackend;
use crate::error::BpannError;
use crate::index::MmapSearchStore;
use crate::merge::{find_query_train_id, merge_topk_precomputed_dist_with_self};
use crate::mmap_store::MmapColumnStore;

pub struct SearchPendingArgs<'a> {
    pub total: usize,
    pub k_eff: usize,
    pub pool_k: usize,
    pub exclude_nearest: bool,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
    pub num_dim: usize,
}

#[doc = "kiss-coverage-off"]
fn merge_pending(
    leg_a: &[(u32, f64)],
    leg_b: &[(u32, f64)],
    train_x: &MmapColumnStore,
    query_buf: &[f64],
    k_eff: usize,
    pool_k: usize,
    exclude_nearest: bool,
) -> Vec<(u32, f64)> {
    let self_id = if exclude_nearest {
        let zero = leg_a
            .iter()
            .chain(leg_b.iter())
            .find(|(_, d)| *d <= 1e-15)
            .map(|(id, _)| *id);
        zero.or_else(|| find_query_train_id(train_x, query_buf))
    } else {
        None
    };
    merge_topk_precomputed_dist_with_self(
        leg_a,
        leg_b,
        k_eff,
        pool_k,
        exclude_nearest,
        self_id,
    )
}

/// Pending rows use the same f64 ranking as tree leaves, including after a reset.
fn pending_topk(
    train_x: &MmapColumnStore, start: usize, end: usize, query: &[f64],
    k: usize, scale_x: bool, x_scale: &[f64],
) -> Result<Vec<(u32, f64)>, BpannError> {
    use crate::small_n_search::OrderedFloat;
    let mut heap = std::collections::BinaryHeap::with_capacity(k + 1);
    for row in start..end {
        let distance = crate::distance::row_sq_l2(
            ndarray::ArrayView1::from(query),
            ndarray::ArrayView1::from(train_x.mmap_row_slice(row)?),
            scale_x, ndarray::ArrayView1::from(x_scale),
        );
        let candidate = (OrderedFloat(distance), row as u32);
        if heap.len() < k {
            heap.push(candidate);
        } else if heap.peek().is_some_and(|worst| candidate < *worst) {
            heap.pop();
            heap.push(candidate);
        }
    }
    Ok(heap.into_sorted_vec().into_iter().map(|(distance, row)| (row, distance.0)).collect())
}

pub fn search_indexed_and_pending(
    backend: &BpannBackend,
    query_rows: &[Vec<f64>],
    dist2s: &mut Array2<f64>,
    indices: &mut Array2<i64>,
    args: SearchPendingArgs<'_>,
) -> Result<(), BpannError> {
    let SearchPendingArgs {
        total,
        k_eff,
        pool_k,
        exclude_nearest,
        scale_x,
        x_scale: x_scale_vec,
        num_dim: _,
    } = args;
    let indexed = backend.index.indexed_rows;
    let pending_start = indexed;
    let has_pending = pending_start < total;
    let index_k = if exclude_nearest {
        pool_k.max(k_eff * 2)
    } else {
        k_eff
    };
    let train_x = &backend.train_x;
    let per_query: Vec<(Vec<f64>, Vec<i64>)> = query_rows
        .par_iter()
        .map(|query_buf| {
            let store = MmapSearchStore {
                train_x,
                scale_x,
                x_scale: x_scale_vec,
            };

            let leg_a = if let Some(tree) = backend.index.tree.as_ref().filter(|_| indexed > 0) {
                crate::index::tree_search::search_tree_refined(
                    tree,
                    query_buf, index_k.max(1), &store,
                ).expect("search_tree_refined")
            } else {
                Vec::new()
            };

            let leg_b = if has_pending {
                pending_topk(train_x, pending_start, total, query_buf, pool_k, scale_x, x_scale_vec)
                    .expect("pending_topk")
            } else {
                Vec::new()
            };

            let merged = merge_pending(
                &leg_a, &leg_b, train_x, query_buf, k_eff, pool_k, exclude_nearest,
            );
            let mut dist_row = vec![f64::INFINITY; k_eff];
            let mut idx_row = vec![-1i64; k_eff];
            for (j, (id, dist)) in merged.into_iter().enumerate() {
                dist_row[j] = dist;
                idx_row[j] = id as i64;
            }
            (dist_row, idx_row)
        })
        .collect();
    for (q, (dist_row, idx_row)) in per_query.into_iter().enumerate() {
        for j in 0..k_eff {
            dist2s[[q, j]] = dist_row[j];
            indices[[q, j]] = idx_row[j];
        }
    }
    Ok(())
}
