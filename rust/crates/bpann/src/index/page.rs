use std::io::{Read, Write};

use crate::distance::l2_sq_f32;

pub const PAGE_MAGIC: u32 = 0x4250_414E; // "BPAN"

#[derive(Clone, Debug, PartialEq)]
pub enum Page {
    Internal {
        page_id: u32,
        centroids: Vec<Vec<f32>>,
        child_page_ids: Vec<u32>,
    },
    Leaf {
        page_id: u32,
        /// Explicit row membership. Empty when [`Self::Leaf::row_range`] is set.
        row_ids: Vec<u32>,
        /// Half-open contiguous membership `[start, end)`. Used by empty-leaf
        /// forests so soft-sync never materializes Θ(N) `u32` identities.
        row_range: Option<(u32, u32)>,
        vectors: Vec<Vec<f32>>,
        stored_centroid: Option<Vec<f32>>,
    },
}

impl Page {
    /// Empty (mmap-backed) leaf over a contiguous row span.
    pub fn empty_range_leaf(
        page_id: u32,
        start: u32,
        end: u32,
        centroid: Vec<f32>,
    ) -> Self {
        debug_assert!(start <= end);
        Page::Leaf {
            page_id,
            row_ids: Vec::new(),
            row_range: Some((start, end)),
            vectors: Vec::new(),
            stored_centroid: Some(centroid),
        }
    }

    pub fn page_id(&self) -> u32 {
        match self {
            Page::Internal { page_id, .. } | Page::Leaf { page_id, .. } => *page_id,
        }
    }

    pub fn centroid(&self) -> Vec<f32> {
        match self {
            Page::Internal { centroids, .. } => {
                if centroids.is_empty() {
                    return Vec::new();
                }
                let dim = centroids[0].len();
                let mut acc = vec![0.0f32; dim];
                for c in centroids {
                    for (j, &v) in c.iter().enumerate() {
                        acc[j] += v;
                    }
                }
                let n = centroids.len() as f32;
                acc.iter().map(|&s| s / n).collect()
            }
            Page::Leaf {
                vectors,
                stored_centroid,
                ..
            } => {
                if let Some(c) = stored_centroid {
                    return c.clone();
                }
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
        }
    }

    pub fn serialize(&self, num_dim: usize) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&PAGE_MAGIC.to_le_bytes());
        match self {
            Page::Internal {
                page_id,
                centroids,
                child_page_ids,
            } => {
                buf.push(0u8);
                buf.extend_from_slice(&page_id.to_le_bytes());
                buf.extend_from_slice(&(num_dim as u32).to_le_bytes());
                buf.extend_from_slice(&(centroids.len() as u32).to_le_bytes());
                for (&child_id, centroid) in child_page_ids.iter().zip(centroids.iter()) {
                    buf.extend_from_slice(&child_id.to_le_bytes());
                    for j in 0..num_dim {
                        buf.extend_from_slice(&centroid.get(j).copied().unwrap_or(0.0).to_le_bytes());
                    }
                }
            }
            Page::Leaf {
                page_id,
                row_ids,
                row_range,
                vectors,
                stored_centroid,
            } => {
                if let Some((start, end)) = row_range {
                    // Kind 3: contiguous empty range leaf (no per-row id list).
                    buf.push(3u8);
                    buf.extend_from_slice(&page_id.to_le_bytes());
                    buf.extend_from_slice(&(num_dim as u32).to_le_bytes());
                    buf.extend_from_slice(&start.to_le_bytes());
                    buf.extend_from_slice(&end.to_le_bytes());
                    let centroid = stored_centroid.as_deref().unwrap_or(&[]);
                    for j in 0..num_dim {
                        buf.extend_from_slice(&centroid.get(j).copied().unwrap_or(0.0).to_le_bytes());
                    }
                    return buf;
                }
                if vectors.is_empty() {
                    buf.push(2u8);
                    buf.extend_from_slice(&page_id.to_le_bytes());
                    buf.extend_from_slice(&(num_dim as u32).to_le_bytes());
                    buf.extend_from_slice(&(row_ids.len() as u32).to_le_bytes());
                    for &row_id in row_ids {
                        buf.extend_from_slice(&row_id.to_le_bytes());
                    }
                    let centroid = stored_centroid.as_deref().unwrap_or(&[]);
                    for j in 0..num_dim {
                        buf.extend_from_slice(&centroid.get(j).copied().unwrap_or(0.0).to_le_bytes());
                    }
                    return buf;
                }
                buf.push(1u8);
                buf.extend_from_slice(&page_id.to_le_bytes());
                buf.extend_from_slice(&(num_dim as u32).to_le_bytes());
                buf.extend_from_slice(&(row_ids.len() as u32).to_le_bytes());
                for (&row_id, vector) in row_ids.iter().zip(vectors.iter()) {
                    buf.extend_from_slice(&row_id.to_le_bytes());
                    for j in 0..num_dim {
                        buf.extend_from_slice(&vector.get(j).copied().unwrap_or(0.0).to_le_bytes());
                    }
                }
            }
        }
        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, String> {
        let (kind, page_id, num_dim, off) = Self::parse_header(data)?;
        match kind {
            0 => Self::deserialize_internal(data, page_id, num_dim, off),
            1 => Self::deserialize_leaf(data, page_id, num_dim, off),
            2 => Self::deserialize_row_id_leaf(data, page_id, num_dim, off),
            3 => Self::deserialize_range_leaf(data, page_id, num_dim, off),
            _ => Err(format!("unknown page kind {kind}")),
        }
    }

    fn parse_header(data: &[u8]) -> Result<(u8, u32, usize, usize), String> {
        if data.len() < 13 {
            return Err("page too short".to_string());
        }
        let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if magic != PAGE_MAGIC {
            return Err("bad page magic".to_string());
        }
        let kind = data[4];
        let page_id = u32::from_le_bytes(data[5..9].try_into().unwrap());
        let num_dim = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        Ok((kind, page_id, num_dim, 13))
    }

    fn read_u32(data: &[u8], off: &mut usize) -> Result<u32, String> {
        if *off + 4 > data.len() {
            return Err("truncated u32".to_string());
        }
        let value = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap());
        *off += 4;
        Ok(value)
    }

    fn read_f32_vec(data: &[u8], off: &mut usize, num_dim: usize) -> Result<Vec<f32>, String> {
        let mut values = Vec::with_capacity(num_dim);
        for _ in 0..num_dim {
            if *off + 4 > data.len() {
                return Err("truncated f32".to_string());
            }
            values.push(f32::from_le_bytes(data[*off..*off + 4].try_into().unwrap()));
            *off += 4;
        }
        Ok(values)
    }

    fn deserialize_internal(
        data: &[u8],
        page_id: u32,
        num_dim: usize,
        mut off: usize,
    ) -> Result<Self, String> {
        let n = Self::read_u32(data, &mut off)? as usize;
        let mut centroids = Vec::with_capacity(n);
        let mut child_page_ids = Vec::with_capacity(n);
        for _ in 0..n {
            child_page_ids.push(Self::read_u32(data, &mut off)?);
            centroids.push(Self::read_f32_vec(data, &mut off, num_dim)?);
        }
        Ok(Page::Internal {
            page_id,
            centroids,
            child_page_ids,
        })
    }

    fn deserialize_leaf(
        data: &[u8],
        page_id: u32,
        num_dim: usize,
        mut off: usize,
    ) -> Result<Self, String> {
        let n = Self::read_u32(data, &mut off)? as usize;
        let mut row_ids = Vec::with_capacity(n);
        let mut vectors = Vec::with_capacity(n);
        for _ in 0..n {
            row_ids.push(Self::read_u32(data, &mut off)?);
            vectors.push(Self::read_f32_vec(data, &mut off, num_dim)?);
        }
        Ok(Page::Leaf {
            page_id,
            row_ids,
            row_range: None,
            vectors,
            stored_centroid: None,
        })
    }

    fn deserialize_row_id_leaf(
        data: &[u8],
        page_id: u32,
        num_dim: usize,
        mut off: usize,
    ) -> Result<Self, String> {
        let n = Self::read_u32(data, &mut off)? as usize;
        let mut row_ids = Vec::with_capacity(n);
        for _ in 0..n {
            row_ids.push(Self::read_u32(data, &mut off)?);
        }
        let centroid = Self::read_f32_vec(data, &mut off, num_dim)?;
        Ok(Page::Leaf {
            page_id,
            row_ids,
            row_range: None,
            vectors: Vec::new(),
            stored_centroid: Some(centroid),
        })
    }

    fn deserialize_range_leaf(
        data: &[u8],
        page_id: u32,
        num_dim: usize,
        mut off: usize,
    ) -> Result<Self, String> {
        let start = Self::read_u32(data, &mut off)?;
        let end = Self::read_u32(data, &mut off)?;
        if start > end {
            return Err(format!("range leaf start {start} > end {end}"));
        }
        let centroid = Self::read_f32_vec(data, &mut off, num_dim)?;
        Ok(Page::empty_range_leaf(page_id, start, end, centroid))
    }
}

pub fn write_pages_index(pages: &[Page], num_dim: usize, w: &mut impl Write) -> std::io::Result<()> {
    w.write_all(&(pages.len() as u32).to_le_bytes())?;
    for page in pages {
        let bytes = page.serialize(num_dim);
        w.write_all(&(bytes.len() as u32).to_le_bytes())?;
        w.write_all(&bytes)?;
    }
    Ok(())
}

pub fn read_pages_index(r: &mut impl Read) -> std::io::Result<Vec<Page>> {
    let mut count_buf = [0u8; 4];
    r.read_exact(&mut count_buf)?;
    let n = u32::from_le_bytes(count_buf) as usize;
    let mut pages = Vec::with_capacity(n);
    for _ in 0..n {
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut data = vec![0u8; len];
        r.read_exact(&mut data)?;
        pages.push(
            Page::deserialize(&data).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?,
        );
    }
    Ok(pages)
}

pub fn closest_child(query: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            l2_sq_f32(query, a)
                .partial_cmp(&l2_sq_f32(query, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn btree_pages_roundtrip() {
        let pages = vec![
            Page::Internal {
                page_id: 0,
                centroids: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                child_page_ids: vec![1, 2],
            },
            Page::Leaf {
                page_id: 1,
                row_ids: vec![0, 1],
                row_range: None,
                vectors: vec![vec![1.0, 2.0], vec![1.1, 2.1]],
                stored_centroid: None,
            },
            Page::Leaf {
                page_id: 2,
                row_ids: vec![2],
                row_range: None,
                vectors: vec![vec![3.0, 4.0]],
                stored_centroid: None,
            },
        ];
        let mut buf = Vec::new();
        write_pages_index(&pages, 2, &mut buf).unwrap();
        let mut cursor = Cursor::new(buf);
        let back = read_pages_index(&mut cursor).unwrap();
        assert_eq!(pages.len(), back.len());
        assert_eq!(pages[0].page_id(), back[0].page_id());
        match (&pages[1], &back[1]) {
            (Page::Leaf { row_ids: a, .. }, Page::Leaf { row_ids: b, .. }) => assert_eq!(a, b),
            _ => panic!("expected leaf"),
        }
    }

    #[test]
    fn range_leaf_roundtrip_and_no_id_list() {
        let page = Page::empty_range_leaf(7, 100, 1100, vec![1.0, 2.0]);
        let bytes = page.serialize(2);
        let back = Page::deserialize(&bytes).unwrap();
        match back {
            Page::Leaf {
                page_id,
                row_ids,
                row_range,
                vectors,
                stored_centroid,
            } => {
                assert_eq!(page_id, 7);
                assert!(row_ids.is_empty());
                assert_eq!(row_range, Some((100, 1100)));
                assert!(vectors.is_empty());
                assert_eq!(stored_centroid.as_deref(), Some(&[1.0f32, 2.0][..]));
            }
            _ => panic!("expected leaf"),
        }
        // Kind byte is 3 (after magic).
        assert_eq!(bytes[4], 3);
    }
}
