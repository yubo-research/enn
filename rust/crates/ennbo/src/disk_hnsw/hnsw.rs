//! HNSW insert and search (Faiss/HNSW32-style).

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use rand::Rng;
use rand::rngs::StdRng;

use crate::disk_hnsw::access::GraphAccess;
use crate::disk_hnsw::graph_mut::GraphMut;
use crate::disk_hnsw::params::{self, LMAX, M};
use crate::error::ENNError;
use crate::knn::MmapColumnStore;

pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

pub fn assign_level(rng: &mut StdRng) -> u8 {
    let u: f64 = rng.gen();
    let u = u.clamp(1e-10, 1.0 - 1e-10);
    let ml = 1.0 / (M as f64).ln();
    let level = (-u.ln() * ml).floor() as usize;
    (level.min(LMAX - 1)) as u8
}

#[derive(Clone, Copy)]
pub struct HnswHeader {
    pub entry_point: u32,
    pub max_level: u8,
    pub num_dim: usize,
}

pub fn search_layer<G: GraphAccess>(
    graph: &G,
    query: &[f32],
    entry: u32,
    ef: usize,
    layer: u8,
    max_id: u32,
) -> Vec<(u32, f32)> {
    let mut visited = HashSet::new();
    visited.insert(entry);
    let entry_dist = graph.vector_l2_sq(entry, query);

    let mut candidates = BinaryHeap::new();
    candidates.push(Reverse((entry_dist.to_bits(), entry)));

    let mut results = BinaryHeap::new();
    results.push((entry_dist.to_bits(), entry));

    while let Some(Reverse((c_dist_bits, c_id))) = candidates.pop() {
        let worst = results
            .peek()
            .map(|(d, _)| *d)
            .unwrap_or(u32::MAX);
        if c_dist_bits > worst && results.len() >= ef {
            break;
        }
        for &n_id in &graph.neighbors(c_id, layer) {
            if n_id >= max_id || visited.contains(&n_id) {
                continue;
            }
            visited.insert(n_id);
            let dist = graph.vector_l2_sq(n_id, query);
            let dist_bits = dist.to_bits();
            if results.len() < ef {
                results.push((dist_bits, n_id));
                candidates.push(Reverse((dist_bits, n_id)));
            } else if let Some((worst_bits, _)) = results.peek().copied() {
                if dist_bits < worst_bits {
                    results.pop();
                    results.push((dist_bits, n_id));
                    candidates.push(Reverse((dist_bits, n_id)));
                }
            }
        }
    }

    let mut out: Vec<(u32, f32)> = results
        .into_iter()
        .map(|(bits, id)| (id, f32::from_bits(bits)))
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out
}

pub(crate) fn select_neighbors(candidates: Vec<(u32, f32)>, max: usize) -> Vec<u32> {
    let mut sorted = candidates;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sorted.truncate(max);
    sorted.into_iter().map(|(id, _)| id).collect()
}

pub(crate) fn shrink_neighbor_list<G: GraphMut>(
    graph: &G,
    neighbors: &mut Vec<u32>,
    layer: u8,
    query: &[f32],
) {
    let max = params::max_neighbors(layer);
    if neighbors.len() <= max {
        return;
    }
    let mut scored: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&id| (id, graph.vector_l2_sq(id, query)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(max);
    *neighbors = scored.into_iter().map(|(id, _)| id).collect();
}

/// Insert without reverse-edge updates; call `rebuild_reverse_edges` before querying.
pub fn insert_forward<G: GraphMut>(
    graph: &mut G,
    header: &mut HnswHeader,
    id: u32,
    vector: &[f32],
    rng: &mut StdRng,
) {
    let level = assign_level(rng);
    graph.write_node(id, level, vector);

    if id == 0 {
        header.entry_point = 0;
        header.max_level = level;
        return;
    }

    let mut curr_ep = vec![header.entry_point];
    for lc in ((level + 1)..=header.max_level).rev() {
        let found = search_layer(graph, vector, curr_ep[0], 1, lc, id);
        if let Some(&(ep, _)) = found.first() {
            curr_ep = vec![ep];
        }
    }

    let start_lc = level.min(header.max_level);
    for lc in (0..=start_lc).rev() {
        let ef = params::ef_construction();
        let candidates = search_layer(graph, vector, curr_ep[0], ef, lc, id);
        let m = params::max_neighbors(lc);
        let selected = select_neighbors(candidates, m);
        graph.set_neighbors(id, lc, &selected);
        if !selected.is_empty() {
            curr_ep = vec![selected[0]];
        }
    }

    if level > header.max_level {
        header.max_level = level;
        header.entry_point = id;
    }
}

/// Add reverse edges for nodes in `[start, end)` after forward-only insertion.
pub fn rebuild_reverse_edges<G: GraphMut>(
    graph: &mut G,
    header: &HnswHeader,
    start: u32,
    end: u32,
) {
    for id in start..end {
        let level = graph.node_level(id);
        let start_lc = level.min(header.max_level);
        for lc in 0..=start_lc {
            let forward: Vec<u32> = graph.neighbors(id, lc);
            for &n_id in &forward {
                let n_vec = graph.vector(n_id);
                let mut back = graph.neighbors(n_id, lc);
                if !back.contains(&id) {
                    back.push(id);
                    shrink_neighbor_list(graph, &mut back, lc, &n_vec);
                    graph.set_neighbors(n_id, lc, &back);
                }
            }
        }
    }
}

pub fn insert<G: GraphMut>(
    graph: &mut G,
    header: &mut HnswHeader,
    id: u32,
    vector: &[f32],
    rng: &mut StdRng,
) {
    insert_forward(graph, header, id, vector, rng);
    rebuild_reverse_edges(graph, header, id, id + 1);
}

pub fn search<G: GraphAccess>(
    graph: &G,
    header: &HnswHeader,
    query: &[f32],
    k: usize,
    ef: usize,
    max_id: u32,
) -> Vec<(u32, f32)> {
    if max_id == 0 {
        return Vec::new();
    }
    let mut curr = header.entry_point;
    if header.max_level > 0 {
        for lc in (1..=header.max_level).rev() {
            let found = search_layer(graph, query, curr, 1, lc, max_id);
            if let Some(&(ep, _)) = found.first() {
                curr = ep;
            }
        }
    }
    let mut results = search_layer(graph, query, curr, ef.max(k), 0, max_id);
    results.truncate(k);
    results
}

pub fn brute_force_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_sq(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Brute-force top-k over mmap rows `[start, end)` with on-the-fly scaling.
pub fn brute_force_topk_mmap(
    train_x: &MmapColumnStore,
    start: usize,
    end: usize,
    query: &[f64],
    k: usize,
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<(u32, f32)>, ENNError> {
    if k == 0 || start >= end {
        return Ok(Vec::new());
    }
    let mut query_f32 = Vec::with_capacity(query.len());
    if scale_x {
        query_f32.extend(
            query
                .iter()
                .zip(x_scale.iter())
                .map(|(&v, &s)| (v / s) as f32),
        );
    } else {
        query_f32.extend(query.iter().map(|&v| v as f32));
    }
    let mut vec_buf = Vec::with_capacity(query.len());
    let mut scored: Vec<(u32, f32)> = Vec::with_capacity(end - start);
    for i in start..end {
        let row = train_x.mmap_row_slice(i)?;
        vec_buf.clear();
        if scale_x {
            vec_buf.extend(
                row.iter()
                    .zip(x_scale.iter())
                    .map(|(&v, &s)| (v / s) as f32),
            );
        } else {
            vec_buf.extend(row.iter().map(|&v| v as f32));
        }
        scored.push((i as u32, l2_sq(&query_f32, &vec_buf)));
    }
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}

/// Merge leg-A and leg-B candidates; rerank by f64 squared L2; apply exclude_nearest on merged pool.
#[allow(clippy::too_many_arguments)]
pub fn merge_topk_candidates(
    train_x: &MmapColumnStore,
    query: &[f64],
    leg_a: &[(u32, f32)],
    leg_b: &[(u32, f32)],
    k_out: usize,
    pool_k: usize,
    exclude_nearest: bool,
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<(u32, f64)>, ENNError> {
    use std::collections::HashMap;
    let mut seen: HashMap<u32, f64> = HashMap::new();
    for &(id, _) in leg_a.iter().chain(leg_b.iter()) {
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = seen.entry(id) {
            let row = train_x.mmap_row_slice(id as usize)?;
            let mut acc = 0.0;
            if scale_x {
                for j in 0..query.len() {
                    let sc = x_scale[j];
                    let d = query[j] / sc - row[j] / sc;
                    acc += d * d;
                }
            } else {
                for (&q, &r) in query.iter().zip(row.iter()) {
                    let d = q - r;
                    acc += d * d;
                }
            }
            e.insert(acc.max(0.0));
        }
    }
    let mut ranked: Vec<(u32, f64)> = seen.into_iter().collect();
    ranked.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    if exclude_nearest && ranked.len() > 1 {
        ranked.remove(0);
    }
    ranked.truncate(pool_k.min(ranked.len()));
    ranked.truncate(k_out);
    Ok(ranked)
}

pub fn mean_recall_at_k(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    ef: usize,
    graph: &impl GraphAccess,
    header: &HnswHeader,
    max_id: u32,
) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for q in queries {
        let bf = brute_force_topk(vectors, q, k);
        let bf_set: HashSet<u32> = bf.iter().map(|(id, _)| *id).collect();
        let approx = search(graph, header, q, k, ef, max_id);
        let hits = approx.iter().filter(|(id, _)| bf_set.contains(id)).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

#[cfg(test)]
mod hnsw_algo_tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn search_layer_upper_levels_and_empty() {
        let graph = crate::disk_hnsw::store::RamGraph::new(2);
        let header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        assert!(search(&graph, &header, &[0.0, 0.0], 1, 16, 0).is_empty());

        let mut graph = crate::disk_hnsw::store::RamGraph::new(2);
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut rng = StdRng::seed_from_u64(77);
        for i in 0..64 {
            insert(
                &mut graph,
                &mut header,
                i,
                &[i as f32 * 0.05, (i % 5) as f32],
                &mut rng,
            );
        }
        if header.max_level > 0 {
            let upper = search_layer(&graph, &[1.0, 1.0], header.entry_point, 1, header.max_level, 64);
            assert!(!upper.is_empty());
        }
        let l0 = search_layer(&graph, &[1.0, 1.0], header.entry_point, 16, 0, 64);
        assert!(l0.len() >= 10);
    }

    #[test]
    fn l2_sq_and_multilevel_search() {
        assert_eq!(l2_sq(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        let mut graph = crate::disk_hnsw::store::RamGraph::new(2);
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut rng = StdRng::seed_from_u64(1);
        for i in 0..20 {
            let v = [i as f32 * 0.1, (i % 3) as f32];
            insert(&mut graph, &mut header, i, &v, &mut rng);
        }
        let hits = search(&graph, &header, &[0.5, 0.0], 3, 32, 20);
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn assign_level_stays_below_lmax_and_brute_force() {
        let mut rng = StdRng::seed_from_u64(999);
        for _ in 0..2000 {
            assert!(assign_level(&mut rng) < LMAX as u8);
        }
        let bf = brute_force_topk(&[vec![0.0, 0.0], vec![1.0, 0.0]], &[0.1, 0.0], 1);
        assert_eq!(bf[0].0, 0);
    }

    #[test]
    fn insert_forward_rebuild_matches_full_insert() {
        let mut full = crate::disk_hnsw::store::RamGraph::new(2);
        let mut fwd = crate::disk_hnsw::store::RamGraph::new(2);
        let mut h_full = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut h_fwd = h_full;
        let mut rng_full = StdRng::seed_from_u64(42);
        let mut rng_fwd = StdRng::seed_from_u64(42);
        for i in 0..12u32 {
            let v = [i as f32 * 0.1, (i % 4) as f32];
            insert(&mut full, &mut h_full, i, &v, &mut rng_full);
            insert_forward(&mut fwd, &mut h_fwd, i, &v, &mut rng_fwd);
            rebuild_reverse_edges(&mut fwd, &h_fwd, i, i + 1);
        }
        assert_eq!(h_full.entry_point, h_fwd.entry_point);
        assert_eq!(h_full.max_level, h_fwd.max_level);
        for id in 0..12u32 {
            for lc in 0..=h_full.max_level {
                assert_eq!(
                    full.neighbors(id, lc),
                    fwd.neighbors(id, lc),
                    "neighbors mismatch id={id} layer={lc}"
                );
            }
        }
    }

    #[test]
    fn brute_force_topk_mmap_scaled_and_empty() {
        use ndarray::Array2;
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let rows = Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 4.0, 8.0]).unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store.mmap_append(&rows.view()).unwrap();
        assert!(brute_force_topk_mmap(&store, 0, 0, &[0.0, 0.0], 1, false, &[]).unwrap().is_empty());
        let hits = brute_force_topk_mmap(
            &store,
            0,
            2,
            &[2.0, 2.0],
            1,
            true,
            &[2.0, 2.0],
        )
        .unwrap();
        assert_eq!(hits[0].0, 0);
        let unscaled = brute_force_topk_mmap(&store, 0, 2, &[2.0, 4.0], 1, false, &[]).unwrap();
        assert_eq!(unscaled[0].0, 0);
    }

    #[test]
    fn merge_topk_candidates_scaled_rerank() {
        use ndarray::Array2;
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let rows = Array2::from_shape_vec((2, 2), vec![2.0, 4.0, 4.0, 8.0]).unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store.mmap_append(&rows.view()).unwrap();
        let scale = [2.0, 2.0];
        let query = [2.0, 2.0];
        let merged = merge_topk_candidates(
            &store,
            &query,
            &[(0, 0.1)],
            &[(1, 0.2)],
            1,
            1,
            false,
            true,
            &scale,
        )
        .unwrap();
        assert_eq!(merged[0].0, 0);
    }

    #[test]
    fn merge_topk_candidates_dedup_and_order() {
        use ndarray::Array2;
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let rows = Array2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0],
        )
        .unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store.mmap_append(&rows.view()).unwrap();
        let query = [0.1, 0.1];
        let leg_a = vec![(0, 0.01), (1, 1.0)];
        let leg_b = vec![(2, 0.5), (0, 0.02)];
        let merged = merge_topk_candidates(
            &store, &query, &leg_a, &leg_b, 2, 2, false, false, &[],
        )
        .unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, 0);
        let with_excl = merge_topk_candidates(
            &store, &query, &leg_a, &leg_b, 1, 2, true, false, &[],
        )
        .unwrap();
        assert_eq!(with_excl.len(), 1);
        assert_ne!(with_excl[0].0, 0);
    }

    #[test]
    fn search_respects_max_id_and_brute_force_k_gt_n() {
        let mut graph = crate::disk_hnsw::store::RamGraph::new(2);
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut rng = StdRng::seed_from_u64(7);
        for i in 0..5u32 {
            insert(
                &mut graph,
                &mut header,
                i,
                &[i as f32, 0.0],
                &mut rng,
            );
        }
        let hits = search(&graph, &header, &[0.0, 0.0], 3, 16, 3);
        assert!(hits.len() <= 3);
        for (id, _) in &hits {
            assert!(*id < 3);
        }
        let vecs: Vec<Vec<f32>> = (0..3).map(|i| vec![i as f32, 0.0]).collect();
        let bf = brute_force_topk(&vecs, &[0.0, 0.0], 5);
        assert_eq!(bf.len(), 3);
    }

    #[test]
    fn select_neighbors_and_shrink_direct() {
        let picked = select_neighbors(vec![(2, 1.0), (0, 0.1), (1, 0.5)], 2);
        assert_eq!(picked, vec![0, 1]);
        let mut graph = crate::disk_hnsw::store::RamGraph::new(2);
        let mut rng = StdRng::seed_from_u64(5);
        insert(&mut graph, &mut HnswHeader { entry_point: 0, max_level: 0, num_dim: 2 }, 0, &[0.0, 0.0], &mut rng);
        insert(&mut graph, &mut HnswHeader { entry_point: 0, max_level: 0, num_dim: 2 }, 1, &[1.0, 0.0], &mut rng);
        let mut nbrs = vec![0, 1];
        shrink_neighbor_list(&graph, &mut nbrs, 0, &[0.5, 0.0]);
        assert_eq!(nbrs.len(), 2);
    }

    #[test]
    fn kiss_hnsw_header_and_search_fn() {
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut graph = crate::disk_hnsw::store::RamGraph::new(2);
        let mut rng = StdRng::seed_from_u64(9);
        insert(&mut graph, &mut header, 0, &[0.0, 0.0], &mut rng);
        let hits = crate::disk_hnsw::hnsw::search(&graph, &header, &[0.0, 0.0], 1, 8, 2);
        assert!(!hits.is_empty());
        let vecs = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
        let bf = crate::disk_hnsw::hnsw::brute_force_topk(&vecs, &[0.0, 0.0], 1);
        assert!(!bf.is_empty());
        let dir = tempfile::TempDir::new().unwrap();
        let mut store =
            crate::knn::MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None)
                .unwrap();
        store
            .mmap_append(&ndarray::array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        let mmap_hits = crate::disk_hnsw::hnsw::brute_force_topk_mmap(
            &store,
            0,
            2,
            &[0.0, 0.0],
            1,
            false,
            &[1.0, 1.0],
        )
        .unwrap();
        assert!(!mmap_hits.is_empty());
        let merged = crate::disk_hnsw::hnsw::merge_topk_candidates(
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
    }
}
