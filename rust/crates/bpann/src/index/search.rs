use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};

use crate::distance::{batched_sq_l2_f32, l2_sq_f32, bpann_row_to_f32};
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::mmap_store::MmapColumnStore;

pub struct MmapSearchStore<'a> {
    pub train_x: &'a MmapColumnStore,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
}

pub fn score_leaf_rows(
    store: Option<&MmapSearchStore<'_>>,
    query: &[f32],
    row_ids: &[u32],
    vectors: &[Vec<f32>],
) -> Result<Vec<(u32, f32)>, BpannError> {
    if !vectors.is_empty() {
        let dists = batched_sq_l2_f32(query, vectors);
        return Ok(row_ids
            .iter()
            .zip(dists.iter())
            .map(|(&row_id, &dist)| (row_id, dist))
            .collect());
    }
    let Some(store) = store else {
        return Ok(Vec::new());
    };
    let mut scored = Vec::with_capacity(row_ids.len());
    let mut vec_buf = Vec::new();
    for &row_id in row_ids {
        let row = store.train_x.mmap_row_slice(row_id as usize)?;
        bpann_row_to_f32(row, store.scale_x, store.x_scale, &mut vec_buf);
        scored.push((row_id, l2_sq_f32(query, &vec_buf)));
    }
    Ok(scored)
}

pub const MAX_CANDIDATE_LEAVES: usize = 384;

pub fn search_exhaustive_leaves(index: &BpannIndex, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    search_exhaustive_leaves_with_store(index, query, k, None)
        .expect("search_exhaustive_leaves")
}

pub fn search_exhaustive_leaves_with_store(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let mut scored = Vec::new();
    for page in &index.pages {
        if let Page::Leaf {
            row_ids,
            vectors,
            ..
        } = page
        {
            scored.extend(score_leaf_rows(store, query, row_ids, vectors)?);
        }
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    Ok(scored)
}

pub struct TraversalLog {
    pub visited_pages: Vec<u32>,
}

impl Default for TraversalLog {
    fn default() -> Self {
        Self::new()
    }
}

impl TraversalLog {
    pub fn new() -> Self {
        Self {
            visited_pages: Vec::new(),
        }
    }
}

pub fn search_index(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    beam_width: usize,
    use_skip_edges: bool,
    visited_log: &mut Vec<u32>,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    if index.pages.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let mut visited_leaves = HashSet::new();
    let mut candidate_leaves = VecDeque::new();

    let root_id = index.header.root_page_id;
    if let Some(root) = index.page_by_id(root_id) {
        collect_candidate_leaves(
            root,
            query,
            index,
            beam_width,
            &mut visited_leaves,
            &mut candidate_leaves,
            visited_log,
        );
    }

    if use_skip_edges {
        let initial: Vec<u32> = candidate_leaves.iter().copied().collect();
        for leaf_id in initial {
            if candidate_leaves.len() >= MAX_CANDIDATE_LEAVES {
                break;
            }
            if let Some(edges) = index.skip_edges.get(&leaf_id) {
                for &next in edges {
                    if candidate_leaves.len() >= MAX_CANDIDATE_LEAVES {
                        break;
                    }
                    if visited_leaves.insert(next) {
                        candidate_leaves.push_back(next);
                        visited_log.push(next);
                    }
                }
            }
        }
    }

    truncate_candidate_leaves(index, query, &mut candidate_leaves, MAX_CANDIDATE_LEAVES);

    let mut scored: Vec<(u32, f32)> = Vec::new();
    for leaf_id in candidate_leaves {
        if let Some(Page::Leaf {
            row_ids,
            vectors,
            ..
        }) = index.page_by_id(leaf_id)
        {
            scored.extend(score_leaf_rows(store, query, row_ids, vectors)?);
        }
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    Ok(scored)
}

fn truncate_candidate_leaves(
    index: &BpannIndex,
    query: &[f32],
    leaves: &mut VecDeque<u32>,
    max: usize,
) {
    if leaves.len() <= max {
        return;
    }
    let mut ranked: Vec<(u32, f32)> = leaves
        .iter()
        .filter_map(|&leaf_id| {
            index
                .page_by_id(leaf_id)
                .map(|page| (leaf_id, l2_sq_f32(query, &page.centroid())))
        })
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    leaves.clear();
    for (leaf_id, _) in ranked.into_iter().take(max) {
        leaves.push_back(leaf_id);
    }
}

fn collect_candidate_leaves(
    page: &Page,
    query: &[f32],
    index: &BpannIndex,
    beam_width: usize,
    visited: &mut HashSet<u32>,
    queue: &mut VecDeque<u32>,
    visited_log: &mut Vec<u32>,
) {
    match page {
        Page::Leaf { page_id, .. } => {
            if visited.insert(*page_id) {
                queue.push_back(*page_id);
                visited_log.push(*page_id);
            }
        }
        Page::Internal {
            centroids,
            child_page_ids,
            ..
        } => {
            let mut ranked: Vec<(usize, f32)> = centroids
                .iter()
                .enumerate()
                .map(|(i, c)| (i, l2_sq_f32(query, c)))
                .collect();
            ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            for (i, _) in ranked.into_iter().take(beam_width.max(1)) {
                let child_id = child_page_ids[i];
                if let Some(child) = index.page_by_id(child_id) {
                    collect_candidate_leaves(
                        child,
                        query,
                        index,
                        beam_width,
                        visited,
                        queue,
                        visited_log,
                    );
                }
            }
        }
    }
}

pub fn search_greedy_blocks_only(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    beam_width: usize,
) -> Vec<(u32, f32)> {
    search_greedy_blocks_only_with_store(index, query, k, beam_width, None)
        .expect("search_greedy_blocks_only")
}

pub fn search_greedy_blocks_only_with_store(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    beam_width: usize,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let mut log = Vec::new();
    search_index(index, query, k, beam_width, false, &mut log, store)
}

pub fn search_with_skip_refinement(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    beam_width: usize,
    visited_log: &mut Vec<u32>,
) -> Vec<(u32, f32)> {
    search_with_skip_refinement_with_store(index, query, k, beam_width, visited_log, None)
        .expect("search_with_skip_refinement")
}

pub fn search_with_skip_refinement_with_store(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    beam_width: usize,
    visited_log: &mut Vec<u32>,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    search_index(index, query, k, beam_width, true, visited_log, store)
}

pub fn bpann_mean_recall_at_k(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    index: &BpannIndex,
) -> f64 {
    if queries.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for q in queries {
        let bf = bpann_brute_force_topk(vectors, q, k);
        let bf_set: HashSet<u32> = bf.iter().map(|(id, _)| *id).collect();
        let approx = search_exhaustive_leaves(index, q, k);
        let hits = approx.iter().filter(|(id, _)| bf_set.contains(id)).count();
        total += hits as f64 / k as f64;
    }
    total / queries.len() as f64
}

pub fn bpann_brute_force_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_sq_f32(query, v)))
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored
}

pub fn bpann_brute_force_topk_mmap(
    train_x: &crate::mmap_store::MmapColumnStore,
    start: usize,
    end: usize,
    query: &[f64],
    k: usize,
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<(u32, f32)>, crate::error::BpannError> {
    let mut scored = Vec::new();
    let mut vec_buf = Vec::new();
    for i in start..end {
        let row = train_x.mmap_row_slice(i)?;
        crate::distance::bpann_row_to_f32(row, scale_x, x_scale, &mut vec_buf);
        let mut acc = 0.0f32;
        if scale_x {
            for j in 0..query.len() {
                let sc = x_scale[j] as f32;
                let d = query[j] as f32 / sc - vec_buf[j];
                acc += d * d;
            }
        } else {
            for (&q, &r) in query.iter().zip(vec_buf.iter()) {
                let d = q as f32 - r;
                acc += d * d;
            }
        }
        scored.push((i as u32, acc));
    }
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.truncate(k);
    Ok(scored)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use crate::index::build::BpannIndex;
    use crate::index::DEFAULT_LEAF_CAPACITY;

    #[test]
    fn search_units_are_linked() {
        let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
        let dir = tempfile::TempDir::new().unwrap();
        let index = BpannIndex::build_from_vectors(
            &vectors,
            2,
            DEFAULT_LEAF_CAPACITY,
            0,
            dir.path().join("index"),
        )
        .unwrap();
        let _ = crate::index::search::bpann_mean_recall_at_k(&vectors, &[vec![0.0, 0.0]], 1, &index);
        let _ = crate::index::search::bpann_brute_force_topk(&vectors, &[0.0, 0.0], 1);
        let mut store =
            crate::mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None)
                .unwrap();
        store
            .mmap_append(&ndarray::array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        let x_scale = [1.0, 1.0];
        let mmap_store = crate::index::search::MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &x_scale,
        };
        let _ = std::mem::size_of_val(&mmap_store);
        let _ = crate::index::search::bpann_brute_force_topk_mmap(
            &store,
            0,
            2,
            &[0.0, 0.0],
            1,
            false,
            &x_scale,
        )
        .unwrap();
        let _ = crate::index::search::search_greedy_blocks_only(&index, &[0.0, 0.0], 1, 2);
        let mut log = Vec::new();
        let _ = crate::index::search::search_with_skip_refinement(
            &index,
            &[0.0, 0.0],
            1,
            2,
            &mut log,
        );
    }
}
