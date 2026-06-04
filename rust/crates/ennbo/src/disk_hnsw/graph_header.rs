//! JSON header for `nodes.bin` graph directory.

use std::path::Path;

use crate::disk_hnsw::params::{EF_CONSTRUCTION, GRAPH_FORMAT_VERSION, LMAX, M, M0};

pub struct GraphHeader {
    pub format_version: u32,
    pub num_dim: usize,
    pub m: usize,
    pub m0: usize,
    pub lmax: usize,
    pub ef_construction: usize,
    pub entry_point: u32,
    pub max_level: u8,
}

impl GraphHeader {
    pub fn defaults(num_dim: usize) -> Self {
        Self {
            format_version: GRAPH_FORMAT_VERSION,
            num_dim,
            m: M,
            m0: M0,
            lmax: LMAX,
            ef_construction: EF_CONSTRUCTION,
            entry_point: 0,
            max_level: 0,
        }
    }

    pub fn write_json(&self, path: &Path) -> Result<(), String> {
        let json = format!(
            "{{\"format_version\":{},\"num_dim\":{},\"M\":{},\"M0\":{},\"LMAX\":{},\"ef_construction\":{},\"entry_point\":{},\"max_level\":{}}}",
            self.format_version,
            self.num_dim,
            self.m,
            self.m0,
            self.lmax,
            self.ef_construction,
            self.entry_point,
            self.max_level
        );
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn read_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Self::from_json_str(&text)
    }

    fn from_json_str(text: &str) -> Result<Self, String> {
        let max_level: usize = parse_json_number(text, "max_level")?;
        let max_level = u8::try_from(max_level).map_err(|_| "max_level out of range".to_string())?;
        Ok(Self {
            format_version: parse_json_number(text, "format_version")?,
            num_dim: parse_json_number(text, "num_dim")?,
            m: parse_json_number(text, "M")?,
            m0: parse_json_number(text, "M0")?,
            lmax: parse_json_number(text, "LMAX")?,
            ef_construction: parse_json_number(text, "ef_construction")?,
            entry_point: parse_json_number(text, "entry_point")?,
            max_level,
        })
    }
}

pub(crate) fn parse_json_number<T: std::str::FromStr>(text: &str, field: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let key = format!("\"{field}\":");
    let pos = text.find(&key).ok_or_else(|| format!("missing {field}"))? + key.len();
    let tail = text[pos..].trim_start();
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end]
        .parse::<T>()
        .map_err(|e| format!("parse {field}: {e}"))
}

#[cfg(test)]
mod graph_header_tests {
    use super::*;
    use crate::disk_hnsw::params::M;
    use tempfile::TempDir;

    #[test]
    fn graph_header_json_roundtrip() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("header.json");
        let hdr = GraphHeader::defaults(8);
        hdr.write_json(&path).unwrap();
        let loaded = GraphHeader::read_json(&path).unwrap();
        assert_eq!(loaded.num_dim, 8);
        assert_eq!(loaded.m, M);
    }

    #[test]
    fn graph_header_parse_all_fields() {
        let text = "{\"format_version\":1,\"num_dim\":4,\"M\":16,\"M0\":32,\"LMAX\":16,\"ef_construction\":200,\"entry_point\":3,\"max_level\":2}";
        let hdr = GraphHeader::from_json_str(text).unwrap();
        assert_eq!(hdr.entry_point, 3);
        assert_eq!(hdr.max_level, 2);
    }

    #[test]
    fn parse_json_number_branches() {
        assert!(parse_json_number::<usize>("{\"x\":12}", "x").is_ok());
        assert!(parse_json_number::<usize>("{\"y\":}", "y").is_err());
    }

    #[test]
    fn graph_header_rejects_bad_max_level() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("header.json");
        std::fs::write(&path, "{\"format_version\":1,\"num_dim\":2,\"M\":16,\"M0\":32,\"LMAX\":16,\"ef_construction\":200,\"entry_point\":0,\"max_level\":999}").unwrap();
        assert!(GraphHeader::read_json(&path).is_err());
    }

    #[test]
    fn graph_header_read_rejects_missing_field() {
        assert!(GraphHeader::from_json_str("{\"format_version\":1}").is_err());
    }
}
