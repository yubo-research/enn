pub mod backend;
pub mod distance;
pub mod error;
pub mod index;
pub mod merge;
pub mod mmap_store;
pub mod observation;

pub use backend::{BpannBackend, DEFAULT_PENDING_FLUSH_THRESHOLD, PAPER_TEX_PATH};
pub use error::BpannError;
pub use index::BpannIndex;
pub use observation::{MAX_NUM_DIM, MAX_RECORD_STRIDE};

#[cfg(test)]
mod acceptance_tests {
    use super::*;
    use crate::distance::row_sq_l2;
    use crate::index::bpann_mean_recall_at_k;
    use ndarray::{array, Array1, Array2};
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    const DATA_SEED: u64 = 42;
    const QUERY_SEED: u64 = 1;
    const N: usize = 128;
    const D: usize = 32;
    const K: usize = 10;
    const NUM_RECALL_QUERIES: usize = 5;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }

    fn synthetic_train(n: usize, d: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let x: Array2<f64> = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
        let y: Array2<f64> = Array2::from_shape_fn((n, 1), |_| rng.gen::<f64>());
        (x, y)
    }

    fn synthetic_f32(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_paper_tex_exists_and_nonempty() {
        let path = repo_root().join(PAPER_TEX_PATH);
        let meta = fs::metadata(&path).expect("paper tex");
        assert!(meta.len() > 0);
        assert_eq!(path, repo_root().join(PAPER_TEX_PATH));
    }

    #[test]
    fn test_open_rejects_num_dim_above_max() {
        let err = backend::open_rejects_num_dim(MAX_NUM_DIM + 1).unwrap_err();
        assert!(err.to_string().contains("MAX_NUM_DIM") || err.to_string().contains("8192"));
    }

    #[test]
    fn test_open_rejects_record_stride_above_max() {
        let huge_dim = MAX_RECORD_STRIDE / 8 + 1;
        let err = backend::open_rejects_record_stride(huge_dim).unwrap_err();
        assert!(err.to_string().contains("record_stride") || err.to_string().contains("8388608"));
    }

    #[test]
    fn test_append_rejects_shape_mismatch() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        let err = b
            .append_rows(&array![[0.0, 0.0, 0.0]].view(), &array![[0.0]].view(), None)
            .unwrap_err();
        assert!(matches!(err, BpannError::InvalidShape { .. }));
    }

    #[test]
    #[allow(non_snake_case)]
    fn MmapColumnStore() {
        use crate::mmap_store::MmapColumnStore;
        let dir = TempDir::new().unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store.mmap_append(&array![[1.0, 0.0]].view()).unwrap();
        assert_eq!(store.nrows, 1);
    }

    #[test]
    fn test_mmap_row_roundtrip_f64() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        b.append_rows(
            &array![[1.23456789, 9.87654321]].view(),
            &array![[0.5]].view(),
            None,
        )
        .unwrap();
        let row = b.mmap_row_slice(0).unwrap();
        assert_eq!(row[0].to_bits(), 1.23456789f64.to_bits());
        assert_eq!(row[1].to_bits(), 9.87654321f64.to_bits());
    }

    #[test]
    fn test_observation_files_created_on_open() {
        let dir = TempDir::new().unwrap();
        let _b = BpannBackend::new_empty(dir.path().to_path_buf(), 3, 2).unwrap();
        assert!(dir.path().join("train_x.bin").exists());
        assert!(dir.path().join("train_y.bin").exists());
        assert!(dir.path().join("metadata.json").exists());
    }

    #[test]
    fn test_y_yvar_not_in_index_pages() {
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(64, 4, 7);
        let y_sentinel = Array2::from_elem((64, 1), 12345.6789);
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            x,
            y_sentinel,
            Some(y),
            false,
            Array1::ones(4),
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        let page_bytes = b.page_bytes();
        let sentinel_bytes = 12345.6789f64.to_le_bytes();
        assert!(
            !page_bytes
                .windows(8)
                .any(|w| w == sentinel_bytes),
            "y payload found in index pages"
        );
    }

    #[test]
    fn test_train_rows_at_returns_matching_y_yvar() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        b.append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            Some(&array![[0.1], [0.2]].view()),
        )
        .unwrap();
        let (tx, ty, yv) = b.train_rows_at(&[1]).unwrap();
        assert_eq!(tx[[0, 0]], 1.0);
        assert_eq!(ty[[0, 0]], 1.0);
        assert_eq!(yv.unwrap()[[0, 0]], 0.2);
    }

    #[test]
    fn test_search_returns_correct_shapes() {
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(32, 8, 1);
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x, y, None, false, Array1::ones(8)).unwrap();
        b.ensure_index_sync().unwrap();
        let queries = Array2::zeros((3, 8));
        let (d, i) = b.search(&queries.view(), 5, false).unwrap();
        assert_eq!(d.shape(), &[3, 5]);
        assert_eq!(i.shape(), &[3, 5]);
    }

    fn brute_force_oracle(b: &BpannBackend, query: &[f64], k: usize) -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..b.len())
            .map(|i| {
                let row = b.mmap_row_slice(i).unwrap();
                let mut acc = 0.0;
                for (&q, &r) in query.iter().zip(row.iter()) {
                    let d = q - r;
                    acc += d * d;
                }
                (i as i64, acc)
            })
            .collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap()
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn test_search_single_query_matches_brute_force_small_n() {
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(64, 8, 3);
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x.clone(), y, None, false, Array1::ones(8)).unwrap();
        b.ensure_index_sync().unwrap();
        let q = x.row(0).to_owned();
        let (d, idx) = b.search(&q.view().insert_axis(ndarray::Axis(0)), 10, false).unwrap();
        let expected = brute_force_oracle(&b, q.as_slice().unwrap(), 10);
        let got: Vec<i64> = idx.row(0).iter().copied().collect();
        assert_eq!(got, expected);
        for j in 0..10 {
            let row = b.mmap_row_slice(expected[j] as usize).unwrap();
            let mut acc = 0.0;
            for (&qv, &rv) in q.iter().zip(row.iter()) {
                let diff = qv - rv;
                acc += diff * diff;
            }
            assert!((d[[0, j]] - acc).abs() < 1e-3);
        }
    }

    #[test]
    fn test_search_exclude_nearest_drops_self() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        b.append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            None,
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        let (_, idx) = b
            .search(&array![[0.0, 0.0]].view(), 1, true)
            .unwrap();
        assert_eq!(idx[[0, 0]], 1);
    }

    #[test]
    fn test_search_batch_matches_rowwise() {
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(48, 6, 11);
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x.clone(), y, None, false, Array1::ones(6)).unwrap();
        b.ensure_index_sync().unwrap();
        let batch = x.slice(ndarray::s![0..3, ..]).to_owned();
        let (bd, bi) = b.search(&batch.view(), 5, false).unwrap();
        for r in 0..3 {
            let (sd, si) = b
                .search(&batch.slice(ndarray::s![r..r + 1, ..]), 5, false)
                .unwrap();
            assert_eq!(bi.row(r).to_vec(), si.row(0).to_vec());
            for c in 0..5 {
                assert!((bd[[r, c]] - sd[[0, c]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_cp0_bpann_ram_recall() {
        let vectors = synthetic_f32(N, D, DATA_SEED);
        let queries = synthetic_f32(NUM_RECALL_QUERIES, D, QUERY_SEED);
        let dir = TempDir::new().unwrap();
        let index = BpannIndex::build_from_vectors(
            &vectors,
            D,
            crate::index::DEFAULT_LEAF_CAPACITY,
            0,
            dir.path().join("index"),
        )
        .unwrap();
        let recall = bpann_mean_recall_at_k(&vectors, &queries, K, &index);
        assert!(recall >= 0.90, "recall@10 = {recall}");
    }

    #[test]
    fn test_search_scaled_l2_consistent_with_enn() {
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(16, 4, 5);
        let scale = array![2.0, 1.0, 4.0, 2.0];
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x.clone(), y, None, true, scale.clone()).unwrap();
        b.ensure_index_sync().unwrap();
        let q = x.row(0);
        let (d, idx) = b.search(&q.view().insert_axis(ndarray::Axis(0)), 3, false).unwrap();
        let neighbor = b.mmap_row_slice(idx[[0, 0]] as usize).unwrap();
        let expected = row_sq_l2(q.view(), neighbor.into(), true, scale.view());
        assert!((d[[0, 0]] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_append_single_row_increments_len() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        b.append_row(&array![0.0, 0.0], &array![0.0], None).unwrap();
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn test_append_batch_rows_increments_len() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        b.append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            None,
        )
        .unwrap();
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn test_search_pending_scaled_l2_non_uniform_scale() {
        let dir = TempDir::new().unwrap();
        let scale = array![2.0, 2.0];
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [10.0, 10.0]],
            array![[0.0], [1.0]],
            None,
            true,
            scale.clone(),
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        // Pending row 2 is the true nearest for query [0, 4]; row 3 ranks higher only
        // when bpann_brute_force_topk_mmap double-applies x_scale, so k=1 leg_b drops row 2.
        b.append_rows(
            &array![[2.0, 4.0], [4.0, 8.0]].view(),
            &array![[2.0], [3.0]].view(),
            None,
        )
        .unwrap();
        let query = array![[0.0, 4.0]];
        let (_, idx) = b.search(&query.view(), 1, false).unwrap();
        assert_eq!(idx[[0, 0]], 2);
    }

    #[test]
    fn test_search_includes_pending_without_sync() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        b.append_rows(
            &array![[100.0, 100.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
        let (_, idx) = b.search(&array![[100.0, 100.0]].view(), 1, false).unwrap();
        assert_eq!(idx[[0, 0]], 2);
        assert_eq!(b.indexed_rows(), 2);
    }

    #[test]
    fn test_search_no_index_build_on_query() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        b.append_rows(
            &array![[2.0, 2.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
        let pages_path = dir.path().join("index/pages.bin");
        let size_before = fs::metadata(&pages_path).map(|m| m.len()).unwrap_or(0);
        let indexed_before = b.indexed_rows();
        b.search(&array![[0.5, 0.5]].view(), 1, false).unwrap();
        assert_eq!(b.indexed_rows(), indexed_before);
        let size_after = fs::metadata(&pages_path).map(|m| m.len()).unwrap_or(0);
        assert_eq!(size_before, size_after);
    }

    #[test]
    fn test_search_mixed_matches_brute_force() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            array![[0.0], [1.0], [2.0]],
            None,
            false,
            Array1::ones(2),
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        b.append_rows(
            &array![[2.0, 2.0], [3.0, 0.0]].view(),
            &array![[3.0], [4.0]].view(),
            None,
        )
        .unwrap();
        let query = [0.9, 0.9];
        let (_, idx) = b.search(&array![query].view(), 2, false).unwrap();
        let mut scored: Vec<(i64, f64)> = (0..b.len())
            .map(|i| {
                let row = b.mmap_row_slice(i).unwrap();
                let mut acc = 0.0;
                for (&q, &r) in query.iter().zip(row.iter()) {
                    let d = q - r;
                    acc += d * d;
                }
                (i as i64, acc)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then_with(|| a.0.cmp(&b.0)));
        scored.truncate(2);
        let expected: Vec<i64> = scored.into_iter().map(|(id, _)| id).collect();
        assert_eq!(idx.row(0).to_vec(), expected);
    }

    #[test]
    fn test_flush_at_threshold_updates_indexed_rows() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 4, 1)
            .unwrap()
            .with_pending_flush_threshold(3)
            .with_defer_append_indexing(false);
        for i in 0..3 {
            b.append_row(&array![i as f64, 0.0, 0.0, 0.0], &array![i as f64], None)
                .unwrap();
        }
        assert_eq!(b.indexed_rows(), 3);
        assert_eq!(b.indexed_rows(), b.len());
    }

    #[test]
    fn test_pending_flush_threshold_default_is_1000() {
        let dir = TempDir::new().unwrap();
        let b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        assert_eq!(b.pending_flush_threshold(), 1000);
    }

    #[test]
    fn test_index_memory_bytes_bounded_after_large_append() {
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 8, 1).unwrap();
        let mem0 = b.index_memory_bytes();
        for i in 0..100 {
            b.append_row(
                &Array1::from_elem(8, i as f64),
                &array![i as f64],
                None,
            )
            .unwrap();
        }
        let mem1 = b.index_memory_bytes();
        assert_eq!(mem0, mem1);
        assert!(mem1 < 100 * 8 * 8);
    }

    #[test]
    fn test_reopen_observation_row_count_preserved() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        {
            let mut b = BpannBackend::new_empty(path.clone(), 2, 1).unwrap();
            b.append_rows(
                &array![[0.0, 0.0], [1.0, 0.0]].view(),
                &array![[0.0], [1.0]].view(),
                None,
            )
            .unwrap();
            assert_eq!(b.len(), 2);
        }
        let b2 = BpannBackend::reopen(path).unwrap();
        assert_eq!(b2.len(), 2);
    }

    #[test]
    fn test_reopen_search_matches_pre_close() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        let (pre_d, pre_i) = {
            let mut b = BpannBackend::new_empty(path.clone(), 4, 1).unwrap();
            let (x, y) = synthetic_train(32, 4, 9);
            b.append_rows(&x.view(), &y.view(), None).unwrap();
            b.ensure_index_sync().unwrap();
            b.search(&x.slice(ndarray::s![0..2, ..]), 5, false).unwrap()
        };
        let b2 = BpannBackend::reopen(path).unwrap();
        let (x, _) = synthetic_train(32, 4, 9);
        let (post_d, post_i) = b2.search(&x.slice(ndarray::s![0..2, ..]), 5, false).unwrap();
        assert_eq!(pre_i, post_i);
        for i in 0..2 {
            for j in 0..5 {
                assert!((pre_d[[i, j]] - post_d[[i, j]]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_reopen_index_pages_unchanged() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        let checksum_before = {
            let mut b = BpannBackend::new_empty(path.clone(), 4, 1).unwrap();
            let (x, y) = synthetic_train(24, 4, 2);
            b.append_rows(&x.view(), &y.view(), None).unwrap();
            b.ensure_index_sync().unwrap();
            fs::read(path.join("index/pages.bin")).unwrap()
        };
        let b2 = BpannBackend::reopen(path.clone()).unwrap();
        assert_eq!(b2.indexed_rows(), 24);
        let checksum_after = fs::read(path.join("index/pages.bin")).unwrap();
        assert_eq!(checksum_before, checksum_after);
    }

    #[test]
    fn test_scale_checkpoints_ci() {
        let n = 128usize;
        let d = 32usize;
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(n, d, 42);
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x.clone(), y, None, false, Array1::ones(d)).unwrap();
        b.ensure_index_sync().unwrap();
        let (d_out, i_out) = b.search(&x.slice(ndarray::s![0..1, ..]), 10.min(n), false).unwrap();
        assert!(d_out[[0, 0]].is_finite());
        assert!(i_out[[0, 0]] >= 0);
        let q = x.row(0).to_owned();
        let (_, i_out) = b.search(&q.view().insert_axis(ndarray::Axis(0)), 10.min(n), false).unwrap();
        let expected = brute_force_oracle(&b, q.as_slice().unwrap(), 10.min(n));
        let got: Vec<i64> = i_out.row(0).iter().copied().collect();
        assert_eq!(got, expected);
    }

    #[test]
    #[ignore = "Run manually: cargo test -p bpann test_scale_recall_ignored -- --ignored --nocapture"]
    fn test_scale_recall_ignored() {
        let n = 1000usize;
        let d = 32usize;
        let dir = TempDir::new().unwrap();
        let (x, y) = synthetic_train(n, d, 42);
        let mut b = BpannBackend::new(dir.path().to_path_buf(), x.clone(), y, None, false, Array1::ones(d)).unwrap();
        b.ensure_index_sync().unwrap();
        let vectors: Vec<Vec<f32>> = (0..b.len())
            .map(|i| {
                b.mmap_row_slice(i)
                    .unwrap()
                    .iter()
                    .map(|&v| v as f32)
                    .collect()
            })
            .collect();
        let queries: Vec<Vec<f32>> = (0..3)
            .map(|i| x.row(i).iter().map(|&v| v as f32).collect())
            .collect();
        let index = b.index_snapshot().unwrap();
        let recall = bpann_mean_recall_at_k(&vectors, &queries, 10, index);
        assert!(recall >= 0.90, "N={n} recall={recall}");
    }

    #[test]
    #[ignore = "Run manually: cargo test -p bpann test_scale_10m_ignored -- --ignored --nocapture"]
    fn test_scale_10m_ignored() {
        let n = 10_000_000usize;
        let d = 8usize;
        let dir = TempDir::new().unwrap();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), d, 1).unwrap();
        let batch = 1000;
        for start in (0..n).step_by(batch) {
            let end = (start + batch).min(n);
            let rows = end - start;
            let x = Array2::from_shape_fn((rows, d), |(i, j)| (start + i + j) as f64);
            let y = Array2::from_shape_fn((rows, 1), |(i, _)| (start + i) as f64);
            b.append_rows(&x.view(), &y.view(), None).unwrap();
            if start % 100_000 == 0 {
                b.ensure_index_sync().unwrap();
                let _ = b.search(&x.slice(ndarray::s![0..1, ..]), 10, false);
            }
        }
    }
}
