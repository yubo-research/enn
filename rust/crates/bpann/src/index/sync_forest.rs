use std::path::PathBuf;

use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::mmap_store::MmapColumnStore;

pub(crate) struct IndexBuildContext<'a> {
    pub train_x: &'a MmapColumnStore,
    pub num_dim: usize,
    pub scale_x: bool,
    pub x_scale: &'a [f64],
    pub work_dir: &'a std::path::Path,
    pub num_metrics: usize,
}

pub(crate) fn first_row_centroid_from_mmap(
    ctx: &IndexBuildContext<'_>,
    row: usize,
) -> Result<Vec<f32>, BpannError> {
    let raw = ctx.train_x.mmap_row_slice(row)?;
    let mut out = Vec::with_capacity(ctx.num_dim);
    bpann_row_to_f32(raw, ctx.scale_x, ctx.x_scale, &mut out);
    Ok(out)
}

pub(crate) fn mean_centroid_f32(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let dim = vectors[0].len();
    let mut acc = vec![0.0f32; dim];
    for v in vectors {
        for (j, &x) in v.iter().enumerate() {
            acc[j] += x;
        }
    }
    let n = vectors.len() as f32;
    acc.iter().map(|&s| s / n).collect()
}

pub(crate) fn load_vectors_from_mmap(
    ctx: &IndexBuildContext<'_>,
    start: usize,
    end: usize,
) -> Result<Vec<Vec<f32>>, BpannError> {
    let mut vectors = Vec::with_capacity(end - start);
    let mut vec_buf = Vec::with_capacity(ctx.num_dim);
    for i in start..end {
        let row = ctx.train_x.mmap_row_slice(i)?;
        bpann_row_to_f32(row, ctx.scale_x, ctx.x_scale, &mut vec_buf);
        vectors.push(std::mem::take(&mut vec_buf));
    }
    Ok(vectors)
}

pub(crate) fn centroid_from_mmap_rows(
    ctx: &IndexBuildContext<'_>,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, BpannError> {
    let dim = ctx.num_dim;
    let mut acc = vec![0.0f64; dim];
    let count = end.saturating_sub(start);
    if count == 0 {
        return Ok(Vec::new());
    }
    for i in start..end {
        let row = ctx.train_x.mmap_row_slice(i)?;
        for (j, &v) in row.iter().enumerate() {
            acc[j] += if ctx.scale_x {
                v / ctx.x_scale[j]
            } else {
                v
            };
        }
    }
    Ok(acc
        .iter()
        .map(|&s| (s / count as f64) as f32)
        .collect())
}

pub(crate) fn build_empty_leaf_forest_index(
    ctx: &IndexBuildContext<'_>,
    start: usize,
    end: usize,
    leaf_rows: usize,
    index_dir: PathBuf,
) -> Result<BpannIndex, BpannError> {
    let leaf_rows = leaf_rows.max(1);
    let mut pages = Vec::new();
    let mut child_page_ids = Vec::new();
    let mut child_centroids = Vec::new();
    let mut next_id = 1u32;
    let mut cur = start;
    while cur < end {
        let chunk_end = (cur + leaf_rows).min(end);
        let centroid = first_row_centroid_from_mmap(ctx, cur)?;
        child_page_ids.push(next_id);
        child_centroids.push(centroid.clone());
        pages.push(Page::empty_range_leaf(
            next_id,
            cur as u32,
            chunk_end as u32,
            centroid,
        ));
        next_id += 1;
        cur = chunk_end;
    }
    pages.push(Page::Internal {
        page_id: 0,
        centroids: child_centroids,
        child_page_ids,
    });
    BpannIndex::from_pages_unpersisted(ctx.num_dim, end - start, leaf_rows, pages, index_dir)
}

pub(crate) fn build_vector_leaf_forest_index(
    ctx: &IndexBuildContext<'_>,
    start: usize,
    end: usize,
    leaf_rows: usize,
    index_dir: PathBuf,
) -> Result<BpannIndex, BpannError> {
    let leaf_rows = leaf_rows.max(1);
    let mut pages = Vec::new();
    let mut child_page_ids = Vec::new();
    let mut child_centroids = Vec::new();
    let mut next_id = 1u32;
    let mut cur = start;
    while cur < end {
        let chunk_end = (cur + leaf_rows).min(end);
        let row_ids: Vec<u32> = (cur..chunk_end).map(|i| i as u32).collect();
        let vectors = load_vectors_from_mmap(ctx, cur, chunk_end)?;
        let centroid = mean_centroid_f32(&vectors);
        child_page_ids.push(next_id);
        child_centroids.push(centroid.clone());
        pages.push(Page::Leaf {
            page_id: next_id,
            row_ids,
            row_range: None,
            vectors,
            stored_centroid: Some(centroid),
        });
        next_id += 1;
        cur = chunk_end;
    }
    pages.push(Page::Internal {
        page_id: 0,
        centroids: child_centroids,
        child_page_ids,
    });
    BpannIndex::from_pages_unpersisted(ctx.num_dim, end - start, leaf_rows, pages, index_dir)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;
    use crate::mmap_store::MmapColumnStore;
    use tempfile::TempDir;

    fn ctx_with_rows(n: usize) -> (TempDir, MmapColumnStore, [f64; 2], Vec<f64>) {
        let dir = TempDir::new().unwrap();
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(x_path, 2, None).unwrap();
        let mut data = ndarray::Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = 1.0;
        }
        store.mmap_append(&data.view()).unwrap();
        let scale = [1.0_f64, 1.0];
        (dir, store, scale, vec![])
    }

    #[test]
    fn forest_helpers_and_builders_cover_paths() {
        let (dir, store, scale, _) = ctx_with_rows(300);
        let ctx = IndexBuildContext {
            train_x: &store,
            num_dim: 2,
            scale_x: false,
            x_scale: &scale,
            work_dir: dir.path(),
            num_metrics: 1,
        };
        assert_eq!(ctx.num_dim, 2);
        assert!(!ctx.scale_x);
        assert_eq!(ctx.num_metrics, 1);
        assert_eq!(ctx.x_scale, &scale);
        assert_eq!(ctx.train_x.nrows, 300);
        assert!(ctx.work_dir.exists());

        let c0 = first_row_centroid_from_mmap(&ctx, 0).unwrap();
        assert_eq!(c0.len(), 2);
        let mean = mean_centroid_f32(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!(mean_centroid_f32(&[]).is_empty());
        let vectors = load_vectors_from_mmap(&ctx, 0, 4).unwrap();
        assert_eq!(vectors.len(), 4);
        let cent = centroid_from_mmap_rows(&ctx, 0, 10).unwrap();
        assert_eq!(cent.len(), 2);
        assert!(centroid_from_mmap_rows(&ctx, 5, 5).unwrap().is_empty());

        let empty = build_empty_leaf_forest_index(&ctx, 0, 250, 64, dir.path().join("e")).unwrap();
        assert_eq!(empty.header.indexed_rows, 250);
        assert!(empty.pages.len() >= 3);
        let vect = build_vector_leaf_forest_index(&ctx, 0, 200, 50, dir.path().join("v")).unwrap();
        assert_eq!(vect.header.indexed_rows, 200);
        assert_eq!(vect.header.leaf_capacity, 50);
        let has_vectors = vect.pages.iter().any(|p| {
            matches!(p, Page::Leaf { vectors, .. } if !vectors.is_empty())
        });
        assert!(has_vectors);

        // Scaled path for centroid_from_mmap_rows / first_row
        let ctx_s = IndexBuildContext {
            train_x: &store,
            num_dim: 2,
            scale_x: true,
            x_scale: &scale,
            work_dir: dir.path(),
            num_metrics: 1,
        };
        let _ = first_row_centroid_from_mmap(&ctx_s, 1).unwrap();
        let _ = centroid_from_mmap_rows(&ctx_s, 0, 3).unwrap();
        let _ = load_vectors_from_mmap(&ctx_s, 0, 2).unwrap();
    }

    #[test]
    fn empty_leaf_forest_row_id_bytes_scale_with_n() {
        let (dir, store, scale, _) = ctx_with_rows(4096);
        let ctx = IndexBuildContext {
            train_x: &store,
            num_dim: 2,
            scale_x: false,
            x_scale: &scale,
            work_dir: dir.path(),
            num_metrics: 1,
        };
        let index = build_empty_leaf_forest_index(&ctx, 0, 4096, 64, dir.path().join("e")).unwrap();
        let mut row_id_bytes = 0usize;
        let mut range_leaf_count = 0usize;
        let mut vector_elems = 0usize;
        for page in &index.pages {
            if let Page::Leaf {
                row_ids,
                row_range,
                vectors,
                ..
            } = page
            {
                row_id_bytes += row_ids.len() * 4;
                if row_range.is_some() {
                    range_leaf_count += 1;
                }
                vector_elems += vectors.iter().map(|v| v.len()).sum::<usize>();
            }
        }
        // Contiguous empty forests must not materialize Θ(N) identities.
        assert_eq!(row_id_bytes, 0);
        assert_eq!(range_leaf_count, 4096 / 64);
        assert_eq!(vector_elems, 0, "empty-leaf forest must not store vectors");
        let ids = index.leaf_row_ids();
        assert_eq!(ids.len(), 4096);
        assert_eq!(ids[0], 0);
        assert_eq!(*ids.last().unwrap(), 4095);
    }

    #[test]
    fn range_leaf_scores_match_explicit_id_leaf() {
        use crate::index::search::{score_leaf_page, MmapSearchStore};
        let (dir, store, scale, _) = ctx_with_rows(128);
        let mmap_store = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &scale,
        };
        let query = vec![10.0f32, 1.0];
        let start = 16u32;
        let end = 80u32;
        let range_page = Page::empty_range_leaf(1, start, end, vec![0.0, 0.0]);
        let ids: Vec<u32> = (start..end).collect();
        let id_page = Page::Leaf {
            page_id: 2,
            row_ids: ids.clone(),
            row_range: None,
            vectors: Vec::new(),
            stored_centroid: Some(vec![0.0, 0.0]),
        };
        let (r_ids, r_range, r_vecs) = match &range_page {
            Page::Leaf {
                row_ids,
                row_range,
                vectors,
                ..
            } => (row_ids.as_slice(), *row_range, vectors.as_slice()),
            _ => unreachable!(),
        };
        let (i_ids, i_range, i_vecs) = match &id_page {
            Page::Leaf {
                row_ids,
                row_range,
                vectors,
                ..
            } => (row_ids.as_slice(), *row_range, vectors.as_slice()),
            _ => unreachable!(),
        };
        let a = score_leaf_page(Some(&mmap_store), &query, r_ids, r_range, r_vecs).unwrap();
        let b = score_leaf_page(Some(&mmap_store), &query, i_ids, i_range, i_vecs).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), (end - start) as usize);
        let _ = dir;
    }
}
