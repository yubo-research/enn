use std::collections::HashMap;

use crate::error::BpannError;
use crate::mmap_store::MmapColumnStore;

/// Squared-distance threshold treated as an exact self hit (LOO identity).
const SELF_DIST_EPS: f64 = 1e-15;

/// Exclude the query's own training row when present among candidates.
///
/// - If `self_id` is known and present: remove that id (even if not nearest).
/// - If `self_id` is known but absent: do not strip a neighbor (approx miss).
/// - If `self_id` is unknown: remove a zero-distance hit if any; otherwise keep
///   the true NN (Exact LOO parity for novel queries that are not training rows).
pub fn bpann_apply_exclude_nearest(
    ranked: &mut Vec<(u32, f64)>,
    exclude_nearest: bool,
    self_id: Option<u32>,
) {
    if !exclude_nearest || ranked.is_empty() {
        return;
    }
    if let Some(sid) = self_id {
        if let Some(pos) = ranked.iter().position(|(id, _)| *id == sid) {
            ranked.remove(pos);
        }
        return;
    }
    if let Some(pos) = ranked.iter().position(|(_, d)| *d <= SELF_DIST_EPS) {
        ranked.remove(pos);
    }
}

/// Exact float row match of `query` against training rows (identity for LOO).
pub fn find_query_train_id(train_x: &MmapColumnStore, query: &[f64]) -> Option<u32> {
    let n = train_x.nrows;
    for i in 0..n {
        let Ok(row) = train_x.mmap_row_slice(i) else {
            continue;
        };
        if row.len() == query.len() && row.iter().zip(query.iter()).all(|(a, b)| a == b) {
            return Some(i as u32);
        }
    }
    None
}

/// Exact float match of `query` in a flat f32 train matrix.
pub fn find_query_train_id_flat(
    flat: &[f32],
    total: usize,
    num_dim: usize,
    query_f32: &[f32],
) -> Option<u32> {
    if query_f32.len() != num_dim {
        return None;
    }
    for i in 0..total {
        let off = i * num_dim;
        if flat[off..off + num_dim] == *query_f32 {
            return Some(i as u32);
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub fn bpann_merge_topk_candidates(
    train_x: &MmapColumnStore,
    query: &[f64],
    leg_a: &[(u32, f32)],
    leg_b: &[(u32, f32)],
    k_out: usize,
    pool_k: usize,
    exclude_nearest: bool,
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<(u32, f64)>, BpannError> {
    let mut seen: HashMap<u32, f64> = HashMap::new();
    for &(id, _) in leg_a.iter().chain(leg_b.iter()) {
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = seen.entry(id) {
            let row = train_x.mmap_row_slice(id as usize)?;
            let dist = crate::distance::row_sq_l2(
                ndarray::ArrayView1::from(query),
                ndarray::ArrayView1::from(row),
                scale_x,
                ndarray::ArrayView1::from(x_scale),
            );
            e.insert(dist);
        }
    }
    let mut ranked: Vec<(u32, f64)> = seen.into_iter().collect();
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    let self_id = if exclude_nearest {
        if let Some((id, _)) = ranked.iter().find(|(_, d)| *d <= SELF_DIST_EPS) {
            Some(*id)
        } else {
            // No zero-dist hit in the pool: identify whether query is a train row so we
            // do not strip a true neighbor when approx search missed self.
            find_query_train_id(train_x, query)
        }
    } else {
        None
    };
    bpann_apply_exclude_nearest(&mut ranked, exclude_nearest, self_id);
    ranked.truncate(pool_k.min(ranked.len()));
    ranked.truncate(k_out);
    Ok(ranked)
}

pub fn merge_topk_precomputed_dist(
    leg_a: &[(u32, f32)],
    leg_b: &[(u32, f32)],
    k_out: usize,
    pool_k: usize,
    exclude_nearest: bool,
) -> Vec<(u32, f64)> {
    merge_topk_precomputed_dist_with_self(leg_a, leg_b, k_out, pool_k, exclude_nearest, None)
}

pub fn merge_topk_precomputed_dist_with_self(
    leg_a: &[(u32, f32)],
    leg_b: &[(u32, f32)],
    k_out: usize,
    pool_k: usize,
    exclude_nearest: bool,
    self_id: Option<u32>,
) -> Vec<(u32, f64)> {
    let mut seen: HashMap<u32, f64> = HashMap::new();
    for &(id, dist) in leg_a.iter().chain(leg_b.iter()) {
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = seen.entry(id) {
            e.insert(f64::from(dist));
        }
    }
    let mut ranked: Vec<(u32, f64)> = seen.into_iter().collect();
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    bpann_apply_exclude_nearest(&mut ranked, exclude_nearest, self_id);
    ranked.truncate(pool_k.min(ranked.len()));
    ranked.truncate(k_out);
    ranked
}

#[cfg(test)]
mod kiss_coverage_tests {
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn merge_units_are_linked() {
        let dir = TempDir::new().unwrap();
        let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store
            .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        let merged = crate::merge::bpann_merge_topk_candidates(
            &store,
            &[0.0, 0.0],
            &[(0, 0.0)],
            &[(1, 1.0)],
            1,
            2,
            false,
            false,
            &[1.0, 1.0],
        )
        .unwrap();
        assert!(!merged.is_empty());
        let _ = crate::merge::merge_topk_precomputed_dist(&[(0, 0.0)], &[(1, 1.0)], 1, 2, false);
    }

    #[test]
    fn exclude_removes_self_id_not_only_nearest() {
        let mut ranked = vec![(3u32, 0.5), (0u32, 0.0), (4u32, 1.0)];
        crate::merge::bpann_apply_exclude_nearest(&mut ranked, true, Some(0));
        assert!(!ranked.iter().any(|(id, _)| *id == 0));
        assert_eq!(ranked[0].0, 3);
    }

    #[test]
    fn exclude_skips_strip_when_known_self_absent() {
        let mut ranked = vec![(3u32, 0.5), (4u32, 1.0)];
        crate::merge::bpann_apply_exclude_nearest(&mut ranked, true, Some(0));
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].0, 3);
    }

    #[test]
    fn exclude_noop_when_disabled_or_empty() {
        let mut ranked = vec![(1u32, 0.0)];
        crate::merge::bpann_apply_exclude_nearest(&mut ranked, false, Some(1));
        assert_eq!(ranked.len(), 1);
        let mut empty: Vec<(u32, f64)> = vec![];
        crate::merge::bpann_apply_exclude_nearest(&mut empty, true, Some(0));
        assert!(empty.is_empty());
    }

    #[test]
    fn exclude_unknown_self_drops_zero_dist_keeps_novel_nn() {
        let mut ranked = vec![(3u32, 0.5), (7u32, 0.0), (4u32, 1.0)];
        crate::merge::bpann_apply_exclude_nearest(&mut ranked, true, None);
        assert!(!ranked.iter().any(|(id, _)| *id == 7));

        // Novel query (no self_id, no zero-dist): keep true NN.
        let mut ranked2 = vec![(3u32, 0.5), (4u32, 1.0)];
        crate::merge::bpann_apply_exclude_nearest(&mut ranked2, true, None);
        assert_eq!(ranked2[0].0, 3);
        assert_eq!(ranked2.len(), 2);
    }

    #[test]
    fn find_query_train_id_matches_exact_row() {
        let dir = TempDir::new().unwrap();
        let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store
            .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        assert_eq!(
            crate::merge::find_query_train_id(&store, &[1.0, 0.0]),
            Some(1)
        );
        assert_eq!(crate::merge::find_query_train_id(&store, &[9.0, 9.0]), None);
        assert_eq!(
            crate::merge::find_query_train_id_flat(&[0.0, 0.0, 1.0, 0.0], 2, 2, &[1.0, 0.0]),
            Some(1)
        );
        assert_eq!(
            crate::merge::find_query_train_id_flat(&[0.0, 0.0], 1, 2, &[0.0]),
            None
        );
    }

    #[test]
    fn merge_with_self_excludes_zero_dist() {
        let out = crate::merge::merge_topk_precomputed_dist_with_self(
            &[(0, 0.0), (1, 1.0)],
            &[],
            1,
            2,
            true,
            Some(0),
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, 1);
    }
}
