//! Large-N BPANN search: indexed fragments + pending mmap brute-force leg.

use ndarray::Array2;
use rayon::prelude::*;

use crate::backend::BpannBackend;
use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::{bpann_brute_force_topk_mmap, MmapSearchStore};
use crate::merge::{bpann_merge_topk_candidates, merge_topk_precomputed_dist};

pub struct SearchPendingArgs<'a> {
    pub total: usize,
    pub k_eff: usize,
    pub pool_k: usize,
    pub exclude_nearest: bool,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
    pub num_dim: usize,
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
        num_dim,
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
            let mut query_f32 = Vec::with_capacity(num_dim);
            bpann_row_to_f32(query_buf, scale_x, x_scale_vec, &mut query_f32);
            let store = MmapSearchStore {
                train_x,
                scale_x,
                x_scale: x_scale_vec,
            };

            let leg_a = if indexed > 0 && !backend.index.indices.is_empty() {
                backend
                    .index
                    .search_candidates(&query_f32, index_k.max(1), Some(&store))
                    .expect("search_candidates")
            } else {
                Vec::new()
            };

            let leg_b = if has_pending {
                bpann_brute_force_topk_mmap(
                    train_x,
                    pending_start,
                    total,
                    query_buf,
                    pool_k,
                    scale_x,
                    x_scale_vec,
                )
                .expect("bpann_brute_force_topk_mmap")
            } else {
                Vec::new()
            };

            let merged = if scale_x {
                bpann_merge_topk_candidates(
                    train_x,
                    query_buf,
                    &leg_a,
                    &leg_b,
                    k_eff,
                    pool_k,
                    exclude_nearest,
                    scale_x,
                    x_scale_vec,
                )
                .expect("bpann_merge_topk_candidates")
            } else {
                merge_topk_precomputed_dist(&leg_a, &leg_b, k_eff, pool_k, exclude_nearest)
            };
            let mut dist_row = vec![0.0; k_eff];
            let mut idx_row = vec![0; k_eff];
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
