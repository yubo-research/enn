use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::error::BpannError;
use crate::index::build::IndexHeader;
use crate::index::page::{write_pages_index, Page};

pub(crate) fn skip_edges_bytes(edges: &HashMap<u32, Vec<u32>>) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    for (&from, tos) in edges {
        buf.extend_from_slice(&from.to_le_bytes());
        buf.extend_from_slice(&(tos.len() as u32).to_le_bytes());
        for &to in tos {
            buf.extend_from_slice(&to.to_le_bytes());
        }
    }
    buf
}

pub(crate) fn write_skip_edges_file(
    path: &Path,
    edges: &HashMap<u32, Vec<u32>>,
) -> Result<(), BpannError> {
    fs::write(path, skip_edges_bytes(edges))
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub(crate) fn persist_index_files(
    index_dir: &Path,
    header: &IndexHeader,
    pages: &[Page],
    skip_edges: &HashMap<u32, Vec<u32>>,
) -> Result<(), BpannError> {
    fs::create_dir_all(index_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let pages_path = index_dir.join("pages.bin");
    let pages_tmp = index_dir.join("pages.bin.tmp");
    let skip_path = index_dir.join("skip_edges.bin");
    let skip_tmp = index_dir.join("skip_edges.bin.tmp");
    let pages_backup = pages_path
        .exists()
        .then(|| fs::read(&pages_path))
        .transpose()
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let skip_backup = skip_path
        .exists()
        .then(|| fs::read(&skip_path))
        .transpose()
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let persist_result = (|| {
        {
            let file = File::create(&pages_tmp)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            let mut writer = BufWriter::new(file);
            write_pages_index(pages, header.num_dim, &mut writer)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            writer
                .flush()
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }
        write_skip_edges_file(&skip_tmp, skip_edges)?;
        fs::rename(&pages_tmp, &pages_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::rename(&skip_tmp, &skip_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        if std::env::var("BPANN_TEST_PERSIST_FAIL").as_deref() == Ok("after_pages_rename") {
            return Err(BpannError::InvalidParameter(
                "test-injected persist failure after pages rename".to_string(),
            ));
        }
        let header_json = serde_json::to_string_pretty(header)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::write(index_dir.join("header.json"), header_json)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        Ok(())
    })();
    if persist_result.is_err() {
        if let Some(bytes) = pages_backup {
            let _ = fs::write(&pages_path, bytes);
        }
        if let Some(bytes) = skip_backup {
            let _ = fs::write(&skip_path, bytes);
        }
        let _ = fs::remove_file(&pages_tmp);
        let _ = fs::remove_file(&skip_tmp);
    }
    persist_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::build::BpannIndex;
    use crate::index::DEFAULT_LEAF_CAPACITY;
    use tempfile::TempDir;

    #[test]
    fn skip_edges_roundtrip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("skip_edges.bin");
        let mut edges = HashMap::new();
        edges.insert(1u32, vec![2, 3]);
        write_skip_edges_file(&path, &edges).unwrap();
        let data = fs::read(&path).unwrap();
        assert!(!data.is_empty());
    }

    #[test]
    fn persist_first_write_and_rollback() {
        let dir = TempDir::new().unwrap();
        let index_dir = dir.path().join("index");
        let index = BpannIndex::build_single_leaf_from_rows_with_persist(
            &[0u32, 1],
            &[vec![0.0f32, 0.0], vec![1.0, 0.0]],
            2,
            index_dir.clone(),
            false,
        )
        .unwrap();
        persist_index_files(
            &index_dir,
            &index.header,
            &index.pages,
            &index.skip_edges,
        )
        .unwrap();
        let pages_before = fs::read(index_dir.join("pages.bin")).unwrap();
        unsafe {
            std::env::set_var("BPANN_TEST_PERSIST_FAIL", "after_pages_rename");
        }
        assert!(persist_index_files(
            &index_dir,
            &index.header,
            &index.pages,
            &index.skip_edges,
        )
        .is_err());
        unsafe {
            std::env::remove_var("BPANN_TEST_PERSIST_FAIL");
        }
        assert_eq!(pages_before, fs::read(index_dir.join("pages.bin")).unwrap());
        BpannIndex::open(index_dir).expect("index must open after rollback");
        let _ = DEFAULT_LEAF_CAPACITY;
    }

    #[test]
    fn persist_atomic_survives_partial_write() {
        let dir = TempDir::new().unwrap();
        let index_dir = dir.path().join("index");
        let index = BpannIndex::build_single_leaf_from_rows_with_persist(
            &[0u32, 1, 2],
            &[vec![0.0f32, 0.0], vec![1.0, 0.0], vec![0.5, 1.0]],
            2,
            index_dir.clone(),
            false,
        )
        .unwrap();
        persist_index_files(
            &index_dir,
            &index.header,
            &index.pages,
            &index.skip_edges,
        )
        .unwrap();
        let opened = BpannIndex::open(index_dir.clone()).expect("initial open");
        assert_eq!(opened.header.indexed_rows, 3);

        unsafe {
            std::env::set_var("BPANN_TEST_PERSIST_FAIL", "after_pages_rename");
        }
        assert!(persist_index_files(
            &index_dir,
            &index.header,
            &index.pages,
            &index.skip_edges,
        )
        .is_err());
        unsafe {
            std::env::remove_var("BPANN_TEST_PERSIST_FAIL");
        }

        let reopened = BpannIndex::open(index_dir).expect("index must open after partial write");
        assert_eq!(reopened.header.indexed_rows, 3);
    }
}
