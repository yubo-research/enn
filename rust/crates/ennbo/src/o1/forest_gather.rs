use super::forest_lrows::NearRows;
use super::forest_order::diag_faults;
use super::forest_tree::Tree;
use super::row_order::Rows;
use crate::error::ENNError;

const AGES: [u32; 4] = [1 << 14, 1 << 16, 1 << 18, u32::MAX];

pub(crate) fn split_gap() -> Option<usize> {
    static GAP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| {
        match std::env::var("ENN_SPLIT_SEQ") {
            Ok(v) if v == "off" => None,
            Ok(v) => Some(v.parse::<usize>().map_or(0, |k| k << 10)),
            Err(_) => Some(0),
        }
    })
}

pub(crate) fn runs(ids: &[u32], row_bytes: usize, gap: usize) -> u64 {
    let mut n = 0;
    let mut a = 0;
    while a < ids.len() {
        let mut b = a + 1;
        while b < ids.len()
            && (ids[b] - ids[b - 1]) as usize * row_bytes <= gap + row_bytes
            && (ids[b] - ids[a] + 1) as usize * row_bytes <= 1 << 20
        {
            b += 1;
        }
        n += 1;
        a = b;
    }
    n
}

pub(crate) fn pages(ids: &[u32], row_bytes: usize) -> u64 {
    let mut last = None;
    let mut n = 0;
    for &i in ids {
        for p in [i as usize * row_bytes / 4096, (i as usize * row_bytes + row_bytes - 1) / 4096] {
            if last != Some(p) {
                n += 1;
                last = Some(p);
            }
        }
    }
    n
}

pub(crate) fn fill(
    near: &mut NearRows,
    miss: &[u32],
    rows: Rows<'_>,
    gap: Option<usize>,
) -> Result<(), ENNError> {
    let Some(gap) = gap else {
        let mut acc = 0.0;
        for &i in miss {
            acc += rows.mapped(i as usize)?.first().copied().unwrap_or(0.0);
        }
        std::hint::black_box(acc);
        return Ok(());
    };
    let d = rows.x.ncols.max(1);
    let mut xs = Vec::with_capacity(miss.len() * d);
    rows.x.pread_runs(miss, gap, &mut xs)?;
    for (&i, x) in miss.iter().zip(xs.chunks_exact(d)) {
        near.insert(i, x.to_vec());
    }
    Ok(())
}

impl Tree {
    pub(super) fn split_rows(
        &mut self,
        leaf_pages: &[u32],
        ids: &[u32],
        newest: u32,
        rows: Rows<'_>,
    ) -> Result<Option<NearRows>, ENNError> {
        let t0 = std::time::Instant::now();
        let Some(mut near) = self.near_rows(leaf_pages, ids)? else {
            return Ok(None);
        };
        let t1 = std::time::Instant::now();
        let mut miss: Vec<u32> = ids.iter().copied().filter(|i| !near.contains_key(i)).collect();
        miss.sort_unstable();
        let rb = rows.x.row_bytes();
        let d = &mut self.stats.split_diag;
        d[0] += (ids.len() - miss.len()) as u64;
        d[1] += miss.len() as u64;
        d[2] += pages(&miss, rb);
        d[3] += runs(&miss, rb, 100 << 10);
        d[4] += (t1 - t0).as_nanos() as u64;
        d[7] += 1;
        for &i in &miss {
            let age = newest.saturating_sub(i);
            d[8 + AGES.iter().position(|&b| age < b).unwrap_or(3)] += 1;
        }
        let f0 = diag_faults();
        fill(&mut near, &miss, rows, split_gap().filter(|_| rows.order.is_none()))?;
        let d = &mut self.stats.split_diag;
        d[5] += t1.elapsed().as_nanos() as u64;
        d[6] += diag_faults().saturating_sub(f0);
        Ok(Some(near))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_and_pages_count_merged_reads() {
        assert_eq!(runs(&[], 80, 0), 0);
        assert_eq!(runs(&[1, 2, 3], 80, 0), 1);
        assert_eq!(runs(&[1, 3], 80, 0), 2);
        assert_eq!(runs(&[1, 3], 80, 80), 1);
        assert_eq!(runs(&[0, 20_000], 80, 1 << 30), 2);
        assert_eq!(pages(&[0, 1, 2], 80), 1);
        assert_eq!(pages(&[0, 51], 80), 2);
        assert_eq!(pages(&[51], 80), 2);
    }

    #[test]
    fn fill_reads_the_same_rows_as_the_map() {
        use crate::backend::MmapColumnStore;
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 3, None).unwrap();
        let x = ndarray::Array2::from_shape_fn((5_000, 3), |(i, j)| (i * 3 + j) as f64);
        store.mmap_append(&x.view()).unwrap();
        let miss: Vec<u32> = vec![2, 3, 40, 41, 900, 4_999];
        for gap in [0, 1 << 10, 100 << 10] {
            let mut near = NearRows::new();
            near.insert(7, vec![9.0; 3]);
            fill(&mut near, &miss, Rows::plain(&store), Some(gap)).unwrap();
            let rows = Rows {
                near: Some(&near),
                ..Rows::plain(&store)
            };
            for &i in &miss {
                assert_eq!(near[&i], store.mmap_row_slice(i as usize).unwrap().to_vec());
                assert_eq!(rows.mapped(i as usize).unwrap(), store.mmap_row_slice(i as usize).unwrap());
            }
            assert_eq!(near.len(), miss.len() + 1);
        }
        let mut near = NearRows::new();
        fill(&mut near, &miss, Rows::plain(&store), None).unwrap();
        assert!(near.is_empty());
        assert!(fill(&mut near, &[5_000], Rows::plain(&store), Some(0)).is_err());
    }

    #[test]
    fn fetch_by_pread_matches_the_map_and_prefers_near_rows() {
        use crate::backend::MmapColumnStore;
        use std::borrow::Cow;
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 3, None).unwrap();
        let x = ndarray::Array2::from_shape_fn((700, 3), |(i, j)| (i * 3 + j) as f64 + 0.5);
        store.mmap_append(&x.view()).unwrap();
        let mut near = NearRows::new();
        near.insert(5, vec![-1.0; 3]);
        let rows = Rows {
            near: Some(&near),
            ..Rows::plain(&store)
        };
        for i in [0usize, 1, 6, 511, 699] {
            let by_read = rows.fetch(i, true).unwrap();
            assert!(matches!(by_read, Cow::Owned(_)));
            assert_eq!(&*by_read, x.row(i).as_slice().unwrap());
            assert_eq!(by_read, rows.fetch(i, false).unwrap());
        }
        assert!(matches!(rows.fetch(5, true).unwrap(), Cow::Borrowed(r) if r == [-1.0; 3]));
        assert!(rows.fetch(700, true).is_err());
        assert!(rows.fetch(700, false).is_err());
    }
}
