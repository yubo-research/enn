use ndarray::ArrayView1;

use crate::model::EpistemicNearestNeighbors;

use super::neighbor_dist::row_dist2s_for_query;

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
    let mut best: Vec<usize> = (0..row.len()).collect();
    best.select_nth_unstable_by(k - 1, |&a, &b| row[a].total_cmp(&row[b]));
    let d_cut = row[best[k - 1]];
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
    let n_train = row.len();
    let mut best: Vec<usize> = (0..n_train).collect();
    best.select_nth_unstable_by(k - 1, |&a, &b| row[a].total_cmp(&row[b]));
    best.truncate(k);
    let d_cut = row[best[k - 1]];
    let n_le = row
        .iter()
        .filter(|d| d.total_cmp(&d_cut) != std::cmp::Ordering::Greater)
        .count();
    if tie_break_neighbors && n_le > k {
        best.clear();
        best.extend(0..n_train);
        best.select_nth_unstable_by(k - 1, |&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
        best.truncate(k);
    }
    best.sort_by(|&a, &b| row[a].total_cmp(&row[b]).then(a.cmp(&b)));
    best
}

fn apply_index_tie_break_at_cutoff(pairs: &mut [(f64, i64)], k: usize) {
    if k == 0 {
        return;
    }
    let d_cut = pairs[k - 1].0;
    let tie_start = pairs[..k].partition_point(|p| p.0 < d_cut);
    pairs[tie_start..k].sort_by_key(|p| p.1);
}

pub(crate) fn finalize_faiss_pool_topk(
    model: &EpistemicNearestNeighbors,
    x_row: ArrayView1<f64>,
    pairs: &mut Vec<(f64, i64)>,
    k: usize,
    tie_break_neighbors: bool,
) -> bool {
    if !tie_break_neighbors {
        pairs.truncate(k);
        return false;
    }
    if pairs.len() > k && pairs[k - 1].0 == pairs[k].0 {
        let row_dists = row_dist2s_for_query(model, x_row);
        if faiss_pool_needs_full_row_scan_from_row(&row_dists, k) {
            return true;
        }
    }
    if pairs.len() > k {
        pairs.truncate(k);
    }
    if faiss_pool_needs_tie_resolution(pairs, k) {
        let row_dists = row_dist2s_for_query(model, x_row);
        if faiss_pool_needs_full_row_scan_from_row(&row_dists, k) {
            return true;
        }
        apply_index_tie_break_at_cutoff(pairs, k);
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::{
        apply_index_tie_break_at_cutoff, faiss_pool_needs_full_row_scan_from_row,
        faiss_pool_needs_tie_resolution, finalize_faiss_pool_topk, topk_indices_from_row_dists,
    };
    use crate::index::IndexDriver;
    use crate::model::EpistemicNearestNeighbors;
    use ndarray::array;

    use super::super::neighbor_dist::row_dist2s_for_query;

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
        assert!(finalize_faiss_pool_topk(
            &model,
            query.row(0),
            &mut escalate,
            2,
            true
        ));

        let train_x2 = array![[0.0], [0.0], [1.0], [2.0]];
        let train_y2 = array![[0.0], [1.0], [2.0], [3.0]];
        let model2 =
            EpistemicNearestNeighbors::new(train_x2, train_y2, None, false, IndexDriver::Exact)
                .unwrap();
        let mut resolve = vec![(0.0, 1i64), (0.0, 0), (1.0, 2), (4.0, 3)];
        assert!(!finalize_faiss_pool_topk(
            &model2,
            query.row(0),
            &mut resolve,
            2,
            true
        ));
        assert_eq!(
            resolve.iter().take(2).map(|p| p.1).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn apply_index_tie_break_at_cutoff_orders_by_index() {
        let mut pairs = vec![(0.0, 2i64), (0.0, 0), (0.0, 1)];
        apply_index_tie_break_at_cutoff(&mut pairs, 3);
        assert_eq!(pairs.iter().map(|p| p.1).collect::<Vec<_>>(), vec![0, 1, 2]);
    }
}
