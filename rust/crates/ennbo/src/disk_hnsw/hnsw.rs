//! HNSW insert and search (Faiss/HNSW32-style).

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

use rand::Rng;
use rand::rngs::StdRng;

use crate::disk_hnsw::access::GraphAccess;
use crate::disk_hnsw::graph_mut::GraphMut;
use crate::disk_hnsw::params::{self, EF_CONSTRUCTION, LMAX, M};

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
        let ef = EF_CONSTRUCTION;
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
}
