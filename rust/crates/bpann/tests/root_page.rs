use std::collections::HashSet;

use ennbo_bpann::index::build::BpannIndex;
use ennbo_bpann::index::page::Page;
use ennbo_bpann::index::search::search_index;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn reachable_leaves(index: &BpannIndex) -> HashSet<u32> {
    let mut leaves = HashSet::new();
    let mut stack = vec![index.header.root_page_id];
    while let Some(id) = stack.pop() {
        match index.page_by_id(id) {
            Some(Page::Internal { child_page_ids, .. }) => stack.extend(child_page_ids),
            Some(Page::Leaf { page_id, .. }) => {
                leaves.insert(*page_id);
            }
            None => {}
        }
    }
    leaves
}

fn build(n: usize, d: usize, leaf_capacity: usize, dir: &TempDir) -> (Vec<Vec<f32>>, BpannIndex) {
    let vectors = synth(n, d, 7);
    let index =
        BpannIndex::build_from_vectors(&vectors, d, leaf_capacity, 0, dir.path().join("index"))
            .unwrap();
    (vectors, index)
}

#[test]
fn kmeans_tree_root_is_internal_and_reaches_every_leaf() {
    let dir = TempDir::new().unwrap();
    let (_, index) = build(1500, 4, 32, &dir);
    assert!(matches!(
        index.page_by_id(index.header.root_page_id),
        Some(Page::Internal { .. })
    ));
    let all: HashSet<u32> = index.leaf_page_ids().into_iter().collect();
    assert!(all.len() > 1);
    assert_eq!(reachable_leaves(&index), all);
}

#[test]
fn kmeans_tree_full_beam_search_is_exact() {
    let dir = TempDir::new().unwrap();
    let (vectors, index) = build(1500, 4, 32, &dir);
    let k = 10;
    let beam = index.pages.len();
    for q in synth(5, 4, 11) {
        let mut exact: Vec<(u32, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = v.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum();
                (i as u32, d)
            })
            .collect();
        exact.sort_by(|a, b| a.1.total_cmp(&b.1));
        let want: HashSet<u32> = exact[..k].iter().map(|(i, _)| *i).collect();
        let mut log = Vec::new();
        let got: HashSet<u32> = search_index(&index, &q, k, beam, false, &mut log, None)
            .unwrap()
            .into_iter()
            .map(|(i, _)| i)
            .collect();
        assert_eq!(got, want);
    }
}
