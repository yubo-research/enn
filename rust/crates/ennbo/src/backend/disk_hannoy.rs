//! Disk-backed ENN backend (mmap observation store + hannoy LMDB HNSW).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::knn::MmapColumnStore;

#[cfg(feature = "hannoy")]
use hannoy::{distances::Euclidean, Database, Reader, Writer};
#[cfg(feature = "hannoy")]
use heed::EnvOpenOptions;

use crate::backend::disk_observation::{
    load_indexed_rows, mmap_row_yvar, open_or_append_yvar, set_index_stale,
    validate_index_backend, write_metadata, INDEX_SYNC_CHUNK_ROWS,
};
const DEFAULT_MAP_SIZE: usize = 1 << 40;
const HANNOY_M: usize = 16;
const HANNOY_M0: usize = 32;

pub struct DiskHannoyEnnBackend {
    work_dir: PathBuf,
    train_x: MmapColumnStore,
    train_y: MmapColumnStore,
    train_yvar: Option<MmapColumnStore>,
    num_dim: usize,
    num_metrics: usize,
    driver: IndexDriver,
    scale_x: bool,
    x_scale: Array1<f64>,
    hannoy_dir: PathBuf,
    #[cfg(feature = "hannoy")]
    env: heed::Env,
    #[cfg(feature = "hannoy")]
    db: Database<Euclidean>,
    indexed_rows: usize,
    index_dirty: Mutex<bool>,
    index_stale: Mutex<bool>,
}

impl DiskHannoyEnnBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::HNSWHannoy {
            return Err(ENNError::InvalidParameter(
                "DiskHannoyEnnBackend requires IndexDriver::HNSWHannoy".to_string(),
            ));
        }
        validate_index_backend(&work_dir, "hannoy")?;
        fs::create_dir_all(&work_dir).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let x_path = work_dir.join("train_x.bin");
        let y_path = work_dir.join("train_y.bin");
        let mut train_x_store = MmapColumnStore::mmap_open_or_create(x_path, num_dim)?;
        let mut train_y_store = MmapColumnStore::mmap_open_or_create(y_path, num_metrics)?;
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store.mmap_append(&train_x.view())?;
            train_y_store.mmap_append(&train_y.view())?;
        }
        let train_yvar_store = open_or_append_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;
        let n = train_x_store.nrows;
        let hannoy_dir = work_dir.join("hannoy");
        let (env, db, indexed_rows) = open_or_create_hannoy(&hannoy_dir)?;
        let indexed_rows = load_indexed_rows(&work_dir).unwrap_or(indexed_rows).min(n);
        write_metadata(&work_dir, n, num_dim, num_metrics, scale_x, indexed_rows, "hannoy")?;
        Ok(Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
            driver,
            scale_x,
            x_scale,
            hannoy_dir,
            env,
            db,
            indexed_rows,
            index_dirty: Mutex::new(indexed_rows < n),
            index_stale: Mutex::new(false),
        })
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, ENNError> {
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            false,
            Array1::ones(num_dim),
            IndexDriver::HNSWHannoy,
        )
    }

    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    pub fn mark_index_stale(&self) {
        set_index_stale(&self.index_stale);
    }

    fn index_row_range(&mut self, start: usize, end: usize) -> Result<(), ENNError> {
        #[cfg(feature = "hannoy")]
        {
            use rand::rngs::StdRng;
            use rand::SeedableRng;

            let ef = std::env::var("ENN_HANNOY_EF_BUILD")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(128);
            let dim = self.num_dim;
            let mut pos = start;
            while pos < end {
                let chunk_end = (pos + INDEX_SYNC_CHUNK_ROWS).min(end);
                let chunk = self.train_x.mmap_row_range(pos, chunk_end)?;
                let scaled = if self.scale_x {
                    (&chunk / &self.x_scale.view().insert_axis(Axis(0))).to_owned()
                } else {
                    chunk
                };
                let mut wtxn = self
                    .env
                    .write_txn()
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                let writer = Writer::new(self.db, 0, dim);
                for (j, row) in scaled.rows().into_iter().enumerate() {
                    let id = (pos + j) as u32;
                    let vec: Vec<f32> = row.iter().map(|&v| v as f32).collect();
                    writer
                        .add_item(&mut wtxn, id, &vec)
                        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                }
                let mut build_rng = StdRng::seed_from_u64(pos as u64);
                writer
                    .builder(&mut build_rng)
                    .ef_construction(ef)
                    .build::<HANNOY_M, HANNOY_M0>(&mut wtxn)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                wtxn.commit()
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                pos = chunk_end;
            }
            self.indexed_rows = end;
            write_metadata(
                &self.work_dir,
                self.len(),
                self.num_dim,
                self.num_metrics,
                self.scale_x,
                self.indexed_rows,
                "hannoy",
            )?;
            Ok(())
        }
        #[cfg(not(feature = "hannoy"))]
        {
            let _ = (start, end);
            Err(ENNError::InvalidParameter(
                "Disk hannoy backend requires the `hannoy` feature".to_string(),
            ))
        }
    }

    pub fn ensure_index_sync(
        &mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        self.scale_x = scale_x;
        self.x_scale = x_scale.to_owned();
        if scale_x {
            let rebuild = {
                let stale = self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned");
                *stale
            };
            if !rebuild && self.indexed_rows >= self.len() {
                return Ok(());
            }
            if rebuild {
                self.indexed_rows = 0;
                self.index_row_range(0, self.len())?;
                *self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned") = false;
            } else {
                self.index_row_range(self.indexed_rows, self.len())?;
            }
            *self
                .index_dirty
                .lock()
                .expect("index_dirty mutex poisoned") = false;
            return Ok(());
        }
        let n = self.len();
        if self.indexed_rows < n {
            self.index_row_range(self.indexed_rows, n)?;
        }
        *self
            .index_dirty
            .lock()
            .expect("index_dirty mutex poisoned") = false;
        Ok(())
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.train_x.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.train_y.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        mmap_row_yvar(self.train_yvar.as_ref(), i)
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let n = self.indexed_rows;
        if n == 0 {
            return Ok((Array2::zeros((x.nrows(), 0)), Array2::zeros((x.nrows(), 0))));
        }
        let k_eff = (search_k as usize).min(n);
        let search_k_usize = if exclude_nearest {
            ((search_k + 1) as usize).min(n)
        } else {
            k_eff
        };
        let ef = search_k_usize.max(64) * 2;

        let mut dist2s = Array2::zeros((x.nrows(), k_eff));
        let mut indices = Array2::zeros((x.nrows(), k_eff));

        #[cfg(feature = "hannoy")]
        {
            let rtxn = self
                .env
                .read_txn()
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            let reader = Reader::<Euclidean>::open(&rtxn, 0, self.db)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            for q in 0..x.nrows() {
                let row = x.slice(ndarray::s![q, ..]);
                let scaled: Array1<f64> = if self.scale_x {
                    (&row / &self.x_scale.view()).to_owned()
                } else {
                    row.to_owned()
                };
                let query: Vec<f32> = scaled.iter().map(|&v| v as f32).collect();
                let nns = reader
                    .nns(k_eff)
                    .ef_search(ef)
                    .by_vector(&rtxn, &query)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
                let pairs: Vec<(u32, f32)> = nns.into_nns().into_iter().collect();
                let mut pairs = pairs;
                if exclude_nearest && pairs.len() > 1 {
                    pairs.remove(0);
                }
                pairs.truncate(k_eff);
                for (j, (id, dist)) in pairs.into_iter().enumerate() {
                    dist2s[[q, j]] = dist as f64;
                    indices[[q, j]] = id as i64;
                }
            }
        }

        Ok((dist2s, indices))
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        let mut total = 0usize;
        if self.hannoy_dir.exists() {
            for name in ["data.mdb", "lock.mdb"] {
                let p = self.hannoy_dir.join(name);
                if p.exists() {
                    total += p.metadata().map(|m| m.len() as usize).unwrap_or(0);
                }
            }
        }
        Ok(total)
    }
}

#[cfg(feature = "hannoy")]
fn open_or_create_hannoy(
    hannoy_dir: &Path,
) -> Result<(heed::Env, Database<Euclidean>, usize), ENNError> {
    fs::create_dir_all(hannoy_dir).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let map_size = std::env::var("ENN_HANNOY_MAP_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAP_SIZE);
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(map_size)
            .max_dbs(1)
            .open(hannoy_dir)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
    };
    let mut wtxn = env
        .write_txn()
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let db = match env
        .open_database(&wtxn, None)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
    {
        Some(db) => db,
        None => env
            .create_database(&mut wtxn, None)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?,
    };
    wtxn.commit()
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok((env, db, 0))
}

crate::impl_disk_mmap_observation_api!(DiskHannoyEnnBackend);

#[cfg(all(test, feature = "hannoy"))]
mod disk_hannoy_unit_tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn disk_hannoy_search_and_metadata() {
        let dir = TempDir::new().expect("tempdir");
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let mut backend = DiskHannoyEnnBackend::new(
            dir.path().to_path_buf(),
            train_x,
            train_y,
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWHannoy,
        )
        .unwrap();
        backend
            .append_rows(
                &array![[2.0, 2.0]].view(),
                &array![[3.0]].view(),
                None,
            )
            .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        let (_, idx) = backend
            .search(&array![[0.9, 0.9]].view(), 1, false)
            .unwrap();
        assert!(idx[[0, 0]] >= 0);
        assert!((idx[[0, 0]] as usize) < backend.len());
        assert!(backend.index_memory_bytes().unwrap() > 0);
        let (x, y, _) = backend.train_rows_at(&[0, 2]).unwrap();
        assert_eq!(x.nrows(), 2);
        assert_eq!(y.nrows(), 2);
        assert_eq!(backend.row_x(1).unwrap()[0], 1.0);
        assert_eq!(backend.row_y(2).unwrap()[0], 2.0);
        assert!(backend.row_yvar(0).unwrap().is_none());
        assert!(backend.row_x(999).is_err());
        assert!(backend.train_rows_at(&[999]).is_err());
    }

    #[test]
    fn disk_hannoy_metadata_roundtrip() {
        let dir = TempDir::new().expect("tempdir");
        write_metadata(dir.path(), 7, 3, 2, true, 5, "hannoy").unwrap();
        assert_eq!(load_indexed_rows(dir.path()), Some(5));
    }

    #[test]
    fn disk_hannoy_scale_x_first_sync_without_mark_stale() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHannoyEnnBackend::new(
            dir.path().to_path_buf(),
            array![[2.0, 4.0], [4.0, 8.0]],
            array![[0.0], [1.0]],
            None,
            true,
            Array1::from_elem(2, 2.0),
            IndexDriver::HNSWHannoy,
        )
        .unwrap();
        assert_eq!(load_indexed_rows(dir.path()), Some(0));
        backend
            .ensure_index_sync(true, &Array1::from_elem(2, 2.0))
            .unwrap();
        assert_eq!(
            load_indexed_rows(dir.path()),
            Some(backend.len())
        );
    }

    #[test]
    fn disk_hannoy_scale_x_rebuild_and_env_helpers() {
        std::env::set_var("ENN_HANNOY_EF_BUILD", "96");
        std::env::set_var("ENN_HANNOY_MAP_SIZE", "1073741824");
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHannoyEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWHannoy,
        )
        .unwrap();
        backend.mark_index_stale();
        backend
            .ensure_index_sync(true, &Array1::from_elem(2, 2.0))
            .unwrap();
        let rows = load_indexed_rows(dir.path()).expect("metadata indexed_rows");
        assert_eq!(rows, backend.len());
        let meta = std::fs::read_to_string(dir.path().join("metadata.json")).unwrap();
        assert!(meta.contains("hannoy"));
    }

    #[test]
    fn disk_hannoy_reopen_existing_mmap_files() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        DiskHannoyEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWHannoy,
        )
        .unwrap();
        let reopened = DiskHannoyEnnBackend::new(
            path,
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWHannoy,
        )
        .unwrap();
        assert_eq!(reopened.len(), 1);
    }
}
