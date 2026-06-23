use std::collections::HashMap;

use crate::error::BpannError;
use crate::mmap_store::MmapColumnStore;

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
    if exclude_nearest && ranked.len() > 1 {
        ranked.remove(0);
    }
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
    if exclude_nearest && ranked.len() > 1 {
        ranked.remove(0);
    }
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
}
