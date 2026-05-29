use ndarray::ArrayView1;

use crate::model::EpistemicNearestNeighbors;

use super::neighbor_dist::row_dist2s_for_query;

pub(crate) type PoolTieScratch = (Vec<(f64, i64)>, Vec<(f64, i64)>);

pub(crate) struct FaissPoolFinalizeCtx<'a> {
    pub precomputed_row_dists: Option<&'a [f64]>,
    pub faiss_pool_size: usize,
    pub tie_scratch: &'a mut PoolTieScratch,
}

#[cfg(test)]
pub(crate) fn faiss_pool_might_need_escalation(
    sorted_pairs: &[(f64, i64)],
    k: usize,
    tie_break_neighbors: bool,
) -> bool {
    if !tie_break_neighbors || k == 0 || sorted_pairs.len() < k {
        return false;
    }
    if sorted_pairs.len() > k && sorted_pairs[k - 1].0 == sorted_pairs[k].0 {
        return true;
    }
    faiss_pool_needs_tie_resolution(sorted_pairs, k)
}

pub(crate) fn faiss_pool_needs_tie_resolution(sorted_pairs: &[(f64, i64)], k: usize) -> bool {
    if k < 2 || sorted_pairs.len() < k {
        return false;
    }
    let d_cut = sorted_pairs[k - 1].0;
    sorted_pairs[..k].partition_point(|p| p.0 < d_cut) < k - 1
}

pub(crate) fn faiss_pool_needs_full_row_scan_from_row(row: &[f64], k: usize) -> bool {
    if k == 0 || row.len() <= k {
        return false;
    }
    let mut nth: Vec<f64> = row.to_vec();
    nth.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
    let d_cut = nth[k - 1];
    row.iter()
        .filter(|d| d.total_cmp(&d_cut) != std::cmp::Ordering::Greater)
        .count()
        > k
}

pub(crate) fn topk_indices_from_row_dists(
    row: &[f64],
    k: usize,
    tie_break_neighbors: bool,
) -> Vec<usize> {
    topk_indices_from_row_dists_with_scratch(row, k, tie_break_neighbors, &mut Vec::new())
}

pub(crate) fn topk_indices_from_row_dists_with_scratch(
    row: &[f64],
    k: usize,
    tie_break_neighbors: bool,
    scratch: &mut Vec<usize>,
) -> Vec<usize> {
    topk_indices_from_row_dists_with_buffers(row, k, tie_break_neighbors, scratch, &mut Vec::new())
}

pub(crate) fn topk_indices_from_row_dists_with_buffers(
    row: &[f64],
    k: usize,
    tie_break_neighbors: bool,
    scratch: &mut Vec<usize>,
    float_buf: &mut Vec<f64>,
) -> Vec<usize> {
    let n_train = row.len();
    if k == 0 {
        return Vec::new();
    }
    if !tie_break_neighbors {
        scratch.clear();
        scratch.extend(0..n_train);
        scratch.select_nth_unstable_by(k - 1, |&a, &b| row[a].total_cmp(&row[b]));
        scratch.truncate(k);
        scratch.sort_by(|&a, &b| row[a].total_cmp(&row[b]));
        return scratch.clone();
    }
    float_buf.clear();
    float_buf.extend_from_slice(row);
    float_buf.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
    let d_cut = float_buf[k - 1];
    scratch.clear();
    scratch.reserve(n_train);
    for (i, &d) in row.iter().enumerate() {
        if d.total_cmp(&d_cut) != std::cmp::Ordering::Greater {
            scratch.push(i);
        }
    }
    if scratch.len() <= k {
        scratch.sort_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
        return scratch.clone();
    }
    scratch.select_nth_unstable_by(k - 1, |&a, &b| {
        row[a].total_cmp(&row[b]).then(a.cmp(&b))
    });
    scratch.truncate(k);
    scratch.sort_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
    scratch.clone()
}

fn apply_index_tie_break_at_cutoff(pairs: &mut [(f64, i64)], k: usize) {
    if k == 0 {
        return;
    }
    let d_cut = pairs[k - 1].0;
    let tie_start = pairs[..k].partition_point(|p| p.0 < d_cut);
    pairs[tie_start..k].sort_by_key(|p| p.1);
}

fn row_dists_for_check<'a>(
    model: &EpistemicNearestNeighbors,
    x_row: ArrayView1<f64>,
    precomputed_row_dists: Option<&'a [f64]>,
    cache: &'a mut Option<Vec<f64>>,
) -> &'a [f64] {
    if let Some(d) = precomputed_row_dists {
        return d;
    }
    if cache.is_none() {
        *cache = Some(row_dist2s_for_query(model, x_row));
    }
    cache.as_deref().expect("row dist cache")
}

fn resolve_pool_tie_break_at_cutoff(pairs: &mut [(f64, i64)], k: usize) {
    if faiss_pool_needs_tie_resolution(pairs, k) {
        apply_index_tie_break_at_cutoff(pairs, k);
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    }
}

/// When the FAISS pool has a tie at the k-th cutoff, pick top-k using only pool
/// members at or below `d_cut` if the pool is not saturated at that distance.
fn try_resolve_boundary_tie_in_pool(
    pairs: &mut Vec<(f64, i64)>,
    k: usize,
    _faiss_pool_size: usize,
    below: &mut Vec<(f64, i64)>,
    at_cut: &mut Vec<(f64, i64)>,
) -> bool {
    if k == 0
        || pairs.len() <= k
        || pairs[k - 1].0.total_cmp(&pairs[k].0) != std::cmp::Ordering::Equal
    {
        return false;
    }
    let d_cut = pairs[k - 1].0;
    below.clear();
    at_cut.clear();
    for &p in pairs.iter() {
        match p.0.total_cmp(&d_cut) {
            std::cmp::Ordering::Less => below.push(p),
            std::cmp::Ordering::Equal => at_cut.push(p),
            std::cmp::Ordering::Greater => {}
        }
    }
    if below.len() + at_cut.len() <= k {
        return false;
    }
    at_cut.sort_by_key(|p| p.1);
    let need = k.saturating_sub(below.len());
    below.extend(at_cut.iter().copied().take(need));
    below.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    pairs.clear();
    pairs.extend_from_slice(below);
    true
}

pub(crate) fn finalize_faiss_pool_topk(
    model: &EpistemicNearestNeighbors,
    x_row: ArrayView1<f64>,
    pairs: &mut Vec<(f64, i64)>,
    k: usize,
    tie_break_neighbors: bool,
    ctx: &mut FaissPoolFinalizeCtx<'_>,
) -> bool {
    if !tie_break_neighbors {
        pairs.truncate(k);
        return false;
    }
    let mut row_dists_cache = None;
    if pairs.len() > k && pairs[k - 1].0.total_cmp(&pairs[k].0) == std::cmp::Ordering::Equal {
        if try_resolve_boundary_tie_in_pool(
            pairs,
            k,
            ctx.faiss_pool_size,
            &mut ctx.tie_scratch.0,
            &mut ctx.tie_scratch.1,
        ) {
            return false;
        }
        let row_dists = row_dists_for_check(
            model,
            x_row,
            ctx.precomputed_row_dists,
            &mut row_dists_cache,
        );
        if faiss_pool_needs_full_row_scan_from_row(row_dists, k) {
            return true;
        }
        pairs.truncate(k);
        resolve_pool_tie_break_at_cutoff(pairs, k);
        return false;
    }
    if pairs.len() > k {
        pairs.truncate(k);
    }
    if faiss_pool_needs_tie_resolution(pairs, k) {
        let row_dists = row_dists_for_check(
            model,
            x_row,
            ctx.precomputed_row_dists,
            &mut row_dists_cache,
        );
        if faiss_pool_needs_full_row_scan_from_row(row_dists, k) {
            return true;
        }
        resolve_pool_tie_break_at_cutoff(pairs, k);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        apply_index_tie_break_at_cutoff, faiss_pool_might_need_escalation,
        faiss_pool_needs_full_row_scan_from_row, faiss_pool_needs_tie_resolution,
        finalize_faiss_pool_topk, resolve_pool_tie_break_at_cutoff, row_dists_for_check,
        FaissPoolFinalizeCtx, PoolTieScratch,
        topk_indices_from_row_dists, topk_indices_from_row_dists_with_buffers,
        try_resolve_boundary_tie_in_pool,
    };
    use crate::index::IndexDriver;
    use crate::model::EpistemicNearestNeighbors;
    use ndarray::array;

    use super::super::neighbor_dist::row_dist2s_for_query;

    #[test]
    fn faiss_pool_might_need_escalation_detects_boundary_tie() {
        let tied = vec![(1.0, 0i64), (2.0, 1), (2.0, 2), (3.0, 3)];
        assert!(faiss_pool_might_need_escalation(&tied, 3, true));
        assert!(!faiss_pool_might_need_escalation(&tied, 3, false));
        let distinct = vec![(1.0, 0i64), (2.0, 1), (3.0, 2)];
        assert!(!faiss_pool_might_need_escalation(&distinct, 3, true));
    }

    #[test]
    fn faiss_pool_needs_tie_resolution_detects_cutoff_band() {
        let tied = vec![(1.0, 0i64), (2.0, 1), (2.0, 2), (3.0, 3)];
        assert!(faiss_pool_needs_tie_resolution(&tied, 3));
        let distinct = vec![(1.0, 0i64), (2.0, 1), (3.0, 2)];
        assert!(!faiss_pool_needs_tie_resolution(&distinct, 3));
        assert!(!faiss_pool_needs_tie_resolution(&distinct, 1));
    }

    #[test]
    fn faiss_pool_needs_full_row_scan_when_more_than_k_at_cutoff() {
        let row = vec![0.0, 0.0, 0.0, 1.0];
        assert!(faiss_pool_needs_full_row_scan_from_row(&row, 2));
        let row2 = vec![0.0, 0.0, 1.0, 2.0];
        assert!(!faiss_pool_needs_full_row_scan_from_row(&row2, 2));
    }

    #[test]
    fn topk_indices_from_row_dists_with_buffers_n_le_gt_k() {
        let row: Vec<f64> = (0..30).map(|i| ((i as f64 - 15.0).abs()).powi(2)).collect();
        let mut scratch = Vec::new();
        let mut float_buf = Vec::new();
        let best = topk_indices_from_row_dists_with_buffers(&row, 5, true, &mut scratch, &mut float_buf);
        assert_eq!(best.len(), 5);
        assert!(best.contains(&15));
    }

    #[test]
    fn row_dists_for_check_uses_precomputed_without_copy() {
        let train_x = array![[0.0], [1.0], [2.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let precomputed = vec![0.0, 1.0, 4.0];
        let mut cache = None;
        let out = row_dists_for_check(
            &model,
            array![0.0].view(),
            Some(&precomputed),
            &mut cache,
        );
        assert_eq!(out, &[0.0, 1.0, 4.0]);
        assert!(cache.is_none());
    }

    #[test]
    fn resolve_pool_tie_break_at_cutoff_orders_cutoff_band() {
        let mut pairs = vec![(0.0, 2i64), (0.0, 0), (0.0, 1), (1.0, 3)];
        resolve_pool_tie_break_at_cutoff(&mut pairs, 3);
        assert_eq!(pairs.iter().take(3).map(|p| p.1).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn topk_indices_from_row_dists_tie_break_and_plain() {
        let row = vec![3.0, 0.0, 0.0, 0.0, 1.0];
        let plain = topk_indices_from_row_dists(&row, 2, false);
        assert_eq!(plain.len(), 2);
        let tied = topk_indices_from_row_dists(&row, 2, true);
        assert_eq!(tied, vec![1, 2]);
    }

    #[test]
    fn row_dist2s_for_query_matches_train_rows() {
        let train_x = array![[0.0], [3.0], [4.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let dists = row_dist2s_for_query(&model, array![0.0].view());
        assert_eq!(dists.len(), 3);
        assert_eq!(dists[0], 0.0);
        assert_eq!(dists[1], 9.0);
    }

    #[test]
    fn finalize_faiss_pool_topk_paths() {
        let train_x = array![[0.0], [0.0], [0.0], [1.0]];
        let train_y = array![[0.0], [1.0], [2.0], [3.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let query = array![[0.0]];
        let mut escalate = vec![(0.0, 0i64), (0.0, 1), (0.0, 2), (1.0, 3)];
        let pool_len = escalate.len();
        let mut scratch: PoolTieScratch = (Vec::new(), Vec::new());
        let mut ctx = FaissPoolFinalizeCtx {
            precomputed_row_dists: None,
            faiss_pool_size: pool_len,
            tie_scratch: &mut scratch,
        };
        assert!(!finalize_faiss_pool_topk(
            &model,
            query.row(0),
            &mut escalate,
            2,
            true,
            &mut ctx,
        ));
        assert_eq!(
            escalate.iter().map(|p| p.1).collect::<Vec<_>>(),
            vec![0, 1]
        );

        let train_x2 = array![[0.0], [0.0], [1.0], [2.0]];
        let train_y2 = array![[0.0], [1.0], [2.0], [3.0]];
        let model2 =
            EpistemicNearestNeighbors::new(train_x2, train_y2, None, false, IndexDriver::Exact)
                .unwrap();
        let mut resolve = vec![(0.0, 1i64), (0.0, 0), (1.0, 2), (4.0, 3)];
        let pool_len = resolve.len();
        let mut scratch: PoolTieScratch = (Vec::new(), Vec::new());
        let mut ctx = FaissPoolFinalizeCtx {
            precomputed_row_dists: None,
            faiss_pool_size: pool_len,
            tie_scratch: &mut scratch,
        };
        assert!(!finalize_faiss_pool_topk(
            &model2,
            query.row(0),
            &mut resolve,
            2,
            true,
            &mut ctx,
        ));
        assert_eq!(
            resolve.iter().take(2).map(|p| p.1).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn try_resolve_boundary_tie_in_pool_picks_lowest_indices() {
        let mut pairs = vec![
            (0.0, 2i64),
            (1.0, 0),
            (1.0, 1),
            (1.0, 3),
            (1.0, 4),
            (2.0, 5),
        ];
        let mut below = Vec::new();
        let mut at_cut = Vec::new();
        assert!(try_resolve_boundary_tie_in_pool(
            &mut pairs, 3, 6, &mut below, &mut at_cut
        ));
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs.iter().map(|p| p.1).collect::<Vec<_>>(), vec![2, 0, 1]);
    }

    #[test]
    fn apply_index_tie_break_at_cutoff_orders_by_index() {
        let mut pairs = vec![(0.0, 2i64), (0.0, 0), (0.0, 1)];
        apply_index_tie_break_at_cutoff(&mut pairs, 3);
        assert_eq!(pairs.iter().map(|p| p.1).collect::<Vec<_>>(), vec![0, 1, 2]);
    }
}
