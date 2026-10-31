
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distance::row_sq_l2;
use crate::merge::{
    find_query_train_id_flat, merge_topk_precomputed_dist_with_self,
};
use rayon::prelude::*;

pub const SMALL_N_INCORE_SEARCH_LIMIT: usize = 8192;

#[derive(Copy, Clone, Debug)]
pub struct OrderedFloat<T = f32>(pub T);

pub use OrderedFloat as OrderedF32;

impl<T: PartialEq> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: PartialEq> Eq for OrderedFloat<T> {}
impl<T: PartialOrd> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: PartialOrd> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

pub fn topk_flat_sq_l2(
    query: &[f64],
    flat: &[f64],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<(u32, f64)> {
    assert_eq!(query.len(), d);
    assert_eq!(flat.len(), n * d);
    if k == 0 || n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let mut heap: BinaryHeap<(OrderedFloat<f64>, u32)> = BinaryHeap::with_capacity(k);
    for i in 0..n {
        let row = &flat[i * d..(i + 1) * d];
        let dist = row_sq_l2(query.into(), row.into(), false, ndarray::ArrayView1::from(&[]));
        let id = i as u32;
        if heap.len() < k {
            heap.push((OrderedFloat(dist), id));
        } else if let Some(&(OrderedFloat(worst), _)) = heap.peek() {
            if dist < worst {
                heap.pop();
                heap.push((OrderedFloat(dist), id));
            }
        }
    }
    let mut out: Vec<(u32, f64)> = heap
        .into_iter()
        .map(|(OrderedFloat(d), id)| (id, d))
        .collect();
    out.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

pub struct ScoreQueriesFlat<'a> {
    pub flat: &'a [f64],
    pub total: usize,
    pub num_dim: usize,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
    pub k_eff: usize,
    pub pool_k: usize,
    pub exclude_nearest: bool,
}

pub fn score_queries_flat(
    query_rows: &[Vec<f64>],
    args: &ScoreQueriesFlat<'_>,
) -> Vec<(Vec<f64>, Vec<i64>)> {
    let ScoreQueriesFlat {
        flat,
        total,
        num_dim,
        scale_x,
        x_scale,
        k_eff,
        pool_k,
        exclude_nearest,
    } = *args;
    query_rows
        .par_iter()
        .map(|query_buf| {
            let mut query_f64 = Vec::with_capacity(num_dim);
            scaled_row(query_buf, scale_x, x_scale, &mut query_f64);
            let leg = topk_flat_sq_l2(&query_f64, flat, total, num_dim, pool_k);
            let self_id = if exclude_nearest {
                find_query_train_id_flat(flat, total, num_dim, &query_f64)
            } else {
                None
            };
            let merged = merge_topk_precomputed_dist_with_self(
                &leg,
                &[],
                k_eff,
                pool_k,
                exclude_nearest,
                self_id,
            );

            let mut dist_row = vec![f64::INFINITY; k_eff];
            let mut idx_row = vec![-1i64; k_eff];
            for (j, (id, dist)) in merged.into_iter().enumerate() {
                dist_row[j] = dist;
                idx_row[j] = id as i64;
            }
            (dist_row, idx_row)
        })
        .collect()
}

fn scaled_row(row: &[f64], scale_x: bool, x_scale: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.extend(row.iter().enumerate().map(|(j, &v)| {
        if scale_x { v / x_scale[j] } else { v }
    }));
}

pub fn load_or_build_small_n_cache(
    backend: &crate::backend::BpannBackend,
    n: usize,
) -> Result<std::sync::Arc<[f64]>, crate::error::BpannError> {
    assert!(n <= SMALL_N_INCORE_SEARCH_LIMIT);
    {
        let guard = backend.small_n_x_cache.lock().expect("small_n_x_cache");
        if let Some((cached_n, ref data)) = *guard {
            if cached_n == n && data.len() == n * backend.num_dim {
                return Ok(std::sync::Arc::clone(data));
            }
        }
    }
    let mut flat = Vec::with_capacity(n * backend.num_dim);
    let mut buf = Vec::with_capacity(backend.num_dim);
    let x_scale = backend.x_scale.as_slice().unwrap();
    for i in 0..n {
        let row = backend.train_x.mmap_row_slice(i)?;
        scaled_row(row, backend.scale_x, x_scale, &mut buf);
        flat.extend_from_slice(&buf);
    }
    let arc: std::sync::Arc<[f64]> = std::sync::Arc::from(flat.into_boxed_slice());
    *backend.small_n_x_cache.lock().expect("small_n_x_cache") =
        Some((n, std::sync::Arc::clone(&arc)));
    Ok(arc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn small_n_incore_limit_is_8192() {
        assert_eq!(SMALL_N_INCORE_SEARCH_LIMIT, 8192);
    }

    #[test]
    fn topk_flat_empty_and_basic() {
        assert!(topk_flat_sq_l2(&[0.0], &[], 0, 1, 1).is_empty());
        let flat = [0.0f64, 1.0, 2.0];
        let hits = topk_flat_sq_l2(&[0.0], &flat, 3, 1, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits[1].0, 1);
        assert!(OrderedF32(1.0) > OrderedF32(0.0));
    }

    #[test]
    fn topk_flat_prefers_closer_2d_rows() {

        let flat = [0.0f64, 0.0, 3.0, 0.0, 1.0, 0.0];
        let hits = topk_flat_sq_l2(&[0.0, 0.0], &flat, 3, 2, 2);
        assert_eq!(hits.iter().map(|h| h.0).collect::<Vec<_>>(), vec![0, 2]);
        assert!((hits[0].1 - 0.0).abs() < 1e-6);
        assert!((hits[1].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn score_queries_flat_exclude_nearest_skips_self() {
        let flat = [0.0f64, 0.0, 1.0, 0.0, 4.0, 0.0];
        let out = score_queries_flat(
            &[vec![0.0, 0.0]],
            &ScoreQueriesFlat {
                flat: &flat,
                total: 3,
                num_dim: 2,
                scale_x: false,
                x_scale: &[1.0, 1.0],
                k_eff: 1,
                pool_k: 2,
                exclude_nearest: true,
            },
        );
        assert_eq!(out[0].1[0], 1);
    }

    #[test]
    fn small_n_cache_rebuilds_after_append() {
        let dir = TempDir::new().unwrap();
        let mut b = crate::backend::BpannBackend::new_empty(dir.path().to_path_buf(), 1, 1)
            .unwrap();
        b.append_rows(&array![[0.0]].view(), &array![[0.0]].view(), None)
            .unwrap();
        let c1 = load_or_build_small_n_cache(&b, 1).unwrap();
        assert_eq!(c1.len(), 1);
        b.append_rows(&array![[5.0]].view(), &array![[1.0]].view(), None)
            .unwrap();
        let c2 = load_or_build_small_n_cache(&b, 2).unwrap();
        assert_eq!(c2.len(), 2);
        assert!((c2[1] - 5.0).abs() < 1e-5);
    }
}
