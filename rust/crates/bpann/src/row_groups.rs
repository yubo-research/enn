
use std::collections::HashMap;
use std::sync::RwLock;

use crate::error::BpannError;
use crate::mmap_store::MmapColumnStore;

const NO_ROW: u32 = u32::MAX;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn hash_x_bits(x: &[f64]) -> u64 {
    x.iter().fold(FNV_OFFSET, |h, &v| {
        let bits = if v == 0.0 { 0.0f64.to_bits() } else { v.to_bits() };
        (h ^ bits).wrapping_mul(FNV_PRIME)
    })
}

#[derive(Default)]
struct Groups {
    head: HashMap<u64, u32>,
    next_with_hash: Vec<u32>,
    row_of: Vec<u32>,
    rep: Vec<u32>,
    count: Vec<u32>,
}

impl Groups {
    fn find(&self, store: &MmapColumnStore, hash: u64, x: &[f64]) -> Result<Option<u32>, BpannError> {
        let mut row = self.head.get(&hash).copied().unwrap_or(NO_ROW);
        while row != NO_ROW {
            if store.mmap_row_slice(self.rep[row as usize] as usize)? == x {
                return Ok(Some(row));
            }
            row = self.next_with_hash[row as usize];
        }
        Ok(None)
    }

    fn add_observation(&mut self, obs: usize, hash: u64, found: Option<u32>) {
        let row = found.unwrap_or_else(|| {
            let row = self.rep.len() as u32;
            self.rep.push(obs as u32);
            self.count.push(0);
            self.next_with_hash
                .push(self.head.insert(hash, row).unwrap_or(NO_ROW));
            row
        });
        self.count[row as usize] += 1;
        self.row_of.push(row);
    }
}

pub struct RowGroups {
    groups: RwLock<Groups>,
    hash: fn(&[f64]) -> u64,
}

impl Default for RowGroups {
    fn default() -> Self {
        Self::with_hasher(hash_x_bits)
    }
}

impl RowGroups {
    pub fn with_hasher(hash: fn(&[f64]) -> u64) -> Self {
        Self {
            groups: RwLock::default(),
            hash,
        }
    }

    pub fn register(&self, store: &MmapColumnStore, upto: usize) -> Result<(), BpannError> {
        let mut g = self.groups.write().expect("row groups");
        for obs in g.row_of.len()..upto {
            let x = store.mmap_row_slice(obs)?;
            let hash = (self.hash)(x);
            let found = g.find(store, hash, x)?;
            g.add_observation(obs, hash, found);
        }
        Ok(())
    }

    pub fn is_representative(&self, obs: u32) -> bool {
        let g = self.groups.read().expect("row groups");
        g.row_of
            .get(obs as usize)
            .is_none_or(|&row| g.rep[row as usize] == obs)
    }

    pub fn row_of(&self, obs: usize) -> Option<usize> {
        let g = self.groups.read().expect("row groups");
        g.row_of.get(obs).map(|&row| row as usize)
    }

    pub fn count(&self, row: usize) -> usize {
        self.groups.read().expect("row groups").count[row] as usize
    }

    pub fn num_rows(&self) -> usize {
        self.groups.read().expect("row groups").rep.len()
    }

    pub fn num_registered(&self) -> usize {
        self.groups.read().expect("row groups").row_of.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    fn store_of(dir: &TempDir, x: &ndarray::Array2<f64>) -> MmapColumnStore {
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), x.ncols(), None)
                .unwrap();
        store.mmap_append(&x.view()).unwrap();
        store
    }

    #[test]
    fn first_observation_at_each_x_represents_its_group() {
        let dir = TempDir::new().unwrap();
        let store = store_of(&dir, &array![[1.0], [2.0], [1.0], [1.0], [3.0], [2.0]]);
        let groups = RowGroups::default();
        groups.register(&store, 6).unwrap();
        assert_eq!((groups.num_rows(), groups.num_registered()), (3, 6));
        let reps: Vec<bool> = (0..6).map(|i| groups.is_representative(i)).collect();
        assert_eq!(reps, [true, true, false, false, true, false]);
        let rows: Vec<usize> = (0..6).map(|i| groups.row_of(i).unwrap()).collect();
        assert_eq!(rows, [0, 1, 0, 0, 2, 1]);
        assert_eq!(
            (0..3).map(|r| groups.count(r)).collect::<Vec<_>>(),
            [3, 2, 1]
        );
        assert!(groups.is_representative(99));
        assert_eq!(groups.row_of(99), None);
    }

    #[test]
    fn registering_in_steps_matches_registering_at_once() {
        let dir = TempDir::new().unwrap();
        let store = store_of(&dir, &array![[0.5, 1.0], [0.5, 1.0], [0.5, 2.0], [0.5, 1.0]]);
        let groups = RowGroups::default();
        groups.register(&store, 2).unwrap();
        groups.register(&store, 4).unwrap();
        groups.register(&store, 4).unwrap();
        assert_eq!((groups.num_rows(), groups.count(0), groups.count(1)), (2, 3, 1));
    }

    #[test]
    fn negative_zero_and_zero_are_the_same_x() {
        let dir = TempDir::new().unwrap();
        let store = store_of(&dir, &array![[0.0, 1.0], [-0.0, 1.0]]);
        let groups = RowGroups::default();
        groups.register(&store, 2).unwrap();
        assert_eq!(groups.num_rows(), 1);
        assert_eq!(hash_x_bits(&[0.0]), hash_x_bits(&[-0.0]));
    }

    #[test]
    fn hash_collisions_are_resolved_by_comparing_x() {
        let dir = TempDir::new().unwrap();
        let store = store_of(&dir, &array![[1.0], [2.0], [3.0], [2.0], [1.0]]);
        let groups = RowGroups::with_hasher(|_| 7);
        groups.register(&store, 5).unwrap();
        assert_eq!(groups.num_rows(), 3);
        let rows: Vec<usize> = (0..5).map(|i| groups.row_of(i).unwrap()).collect();
        assert_eq!(rows, [0, 1, 2, 1, 0]);
    }

    use crate::backend::BpannBackend;
    use ndarray::Array2;

    fn aggregating(dir: &TempDir, flush: usize) -> BpannBackend {
        BpannBackend::new_aggregating(
            dir.path().to_path_buf(),
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            ndarray::Array1::ones(2),
            true,
        )
        .unwrap()
        .with_pending_flush_threshold(flush)
    }

    fn repeated_rows(n: usize, copies: usize) -> (Array2<f64>, Array2<f64>) {
        let x = Array2::from_shape_fn((n * copies, 2), |(i, j)| {
            let site = (i * 7919 + 13) % n;
            (site as f64 * 0.37 + j as f64).sin()
        });
        (x, Array2::zeros((n * copies, 1)))
    }

    fn unique_first_rows(x: &Array2<f64>) -> Array2<f64> {
        let mut keep: Vec<usize> = Vec::new();
        for (i, row) in x.rows().into_iter().enumerate() {
            if !keep.iter().any(|&k| x.row(k) == row) {
                keep.push(i);
            }
        }
        x.select(ndarray::Axis(0), &keep)
    }

    fn entries(b: &BpannBackend) -> Vec<u32> {
        b.index_snapshot().map_or(Vec::new(), |t| t.leaf_row_ids())
    }

    #[test]
    fn index_holds_representatives_only_and_matches_the_filtered_build() {
        let (x, y) = repeated_rows(60, 5);
        let dir = TempDir::new().unwrap();
        let mut b = aggregating(&dir, 1000);
        for lo in (0..x.nrows()).step_by(50) {
            let hi = (lo + 50).min(x.nrows());
            let (xs, ys) = (x.slice(ndarray::s![lo..hi, ..]), y.slice(ndarray::s![lo..hi, ..]));
            b.append_rows(&xs, &ys, None).unwrap();
        }
        b.ensure_index_sync().unwrap();
        let groups = b.row_groups().unwrap();
        assert_eq!((b.len(), groups.num_rows()), (300, 60));
        let reps: Vec<u32> = (0..300u32).filter(|&i| groups.is_representative(i)).collect();
        assert_eq!(entries(&b), reps);

        let unique = unique_first_rows(&x);
        let dir2 = TempDir::new().unwrap();
        let mut plain = BpannBackend::new_empty(dir2.path().to_path_buf(), 2, 1).unwrap();
        plain
            .append_rows(&unique.view(), &Array2::zeros((unique.nrows(), 1)).view(), None)
            .unwrap();
        let q = Array2::from_shape_fn((6, 2), |(i, j)| (i * 3 + j) as f64 * 0.11 - 0.4);
        let (d_agg, i_agg) = b.search(&q.view(), 8, false).unwrap();
        let (d_plain, i_plain) = plain.search(&q.view(), 8, false).unwrap();
        assert_eq!(d_agg, d_plain);
        for (a, p) in i_agg.iter().zip(i_plain.iter()) {
            assert_eq!(*a as u32, reps[*p as usize]);
        }
        for row in i_agg.rows() {
            let mut sorted = row.to_vec();
            sorted.dedup();
            assert_eq!(sorted.len(), row.len());
        }
    }

    #[test]
    fn pending_rows_skip_duplicates_and_exclude_nearest_removes_the_group() {
        let dir = TempDir::new().unwrap();
        let mut b = aggregating(&dir, 1000).with_defer_append_indexing(true);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 0.0]];
        b.append_rows(&x.view(), &Array2::zeros((5, 1)).view(), None)
            .unwrap();
        let q = array![[0.0, 0.0]];
        let (_, idx) = b.search(&q.view(), 5, false).unwrap();
        assert_eq!(idx.row(0).to_vec(), vec![0, 1, 4]);
        let (_, idx) = b.search(&q.view(), 3, true).unwrap();
        assert_eq!(idx.row(0).to_vec(), vec![1, 4]);
    }

    #[test]
    fn reopen_rebuilds_the_groups_and_keeps_the_index_free_of_duplicates() {
        let (x, y) = repeated_rows(40, 4);
        let dir = TempDir::new().unwrap();
        let mut b = aggregating(&dir, 1000);
        b.append_rows(&x.view(), &y.view(), None).unwrap();
        b.persist_index_to_disk().unwrap();
        let before = entries(&b);
        let q = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64 * 0.2);
        let expected = b.search(&q.view(), 6, false).unwrap();
        drop(b);
        let mut again = BpannBackend::reopen_aggregating(dir.path().to_path_buf(), true).unwrap();
        assert_eq!(again.row_groups().unwrap().num_rows(), 40);
        assert_eq!(entries(&again), before);
        assert_eq!(again.search(&q.view(), 6, false).unwrap(), expected);
        again
            .append_rows(&x.slice(ndarray::s![..8, ..]), &y.slice(ndarray::s![..8, ..]), None)
            .unwrap();
        again.ensure_index_sync().unwrap();
        assert_eq!(entries(&again), before);
    }

    #[test]
    fn soft_sync_and_metric_rebuild_keep_duplicates_out() {
        let (x, y) = repeated_rows(50, 6);
        let dir = TempDir::new().unwrap();
        let mut b = aggregating(&dir, 1000).with_defer_append_indexing(true);
        b.append_rows(&x.view(), &y.view(), None).unwrap();
        if let Some(built) = crate::backend::soft_sync_build(&b).unwrap() {
            crate::backend::soft_sync_publish(&mut b, built);
        }
        let reps: Vec<u32> = {
            let g = b.row_groups().unwrap();
            (0..300u32).filter(|&i| g.is_representative(i)).collect()
        };
        assert_eq!(entries(&b), reps);
        b.rebuild_metric(&ndarray::Array1::from(vec![2.0, 0.5])).unwrap();
        b.mark_index_stale();
        b.ensure_index_sync().unwrap();
        assert_eq!(entries(&b), reps);
        assert_eq!(b.indexed_rows(), 300);
    }

    #[test]
    fn a_large_batch_of_duplicates_bulk_builds_representatives_only() {
        let n = 3000;
        let x = Array2::from_shape_fn((n * 10, 1), |(i, _)| (i % n) as f64);
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_aggregating(
            dir.path().to_path_buf(),
            x.clone(),
            Array2::zeros((n * 10, 1)),
            None,
            false,
            ndarray::Array1::ones(1),
            true,
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        assert_eq!(entries(&b), (0..n as u32).collect::<Vec<_>>());
        assert_eq!(b.indexed_rows(), n * 10);
    }
}
