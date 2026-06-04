//! Fixed-stride node record layout for `nodes.bin`.

use crate::disk_hnsw::params::{self, EMPTY_NEIGHBOR, LMAX, M, M0};

pub struct NodeLayout {
    pub num_dim: usize,
    pub record_stride: usize,
}

impl NodeLayout {
    pub fn new(num_dim: usize) -> Self {
        let vector_bytes = num_dim * std::mem::size_of::<f32>();
        let layer0_bytes = M0 * std::mem::size_of::<u32>();
        let upper_bytes = (LMAX - 1) * M * std::mem::size_of::<u32>();
        let record_stride = 4 + vector_bytes + layer0_bytes + upper_bytes;
        Self {
            num_dim,
            record_stride,
        }
    }

    pub fn neighbors_offset(&self, layer: u8) -> usize {
        debug_assert!(layer < LMAX as u8);
        let base = 4 + self.num_dim * std::mem::size_of::<f32>();
        if layer == 0 {
            base
        } else {
            base + M0 * std::mem::size_of::<u32>() + (layer as usize - 1) * M * std::mem::size_of::<u32>()
        }
    }

    pub fn max_neighbors(layer: u8) -> usize {
        params::max_neighbors(layer)
    }

    pub fn write_record(
        &self,
        buf: &mut [u8],
        level: u8,
        vector: &[f32],
        neighbors: &[Vec<u32>],
    ) {
        assert_eq!(buf.len(), self.record_stride);
        assert_eq!(vector.len(), self.num_dim);
        buf[0] = level;
        buf[1..4].fill(0);
        let v_off = 4;
        for (j, &v) in vector.iter().enumerate() {
            let off = v_off + j * 4;
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        for layer in 0..LMAX {
            let cap = Self::max_neighbors(layer as u8);
            let off = self.neighbors_offset(layer as u8);
            for slot in 0..cap {
                let val = neighbors
                    .get(layer)
                    .and_then(|n| n.get(slot))
                    .copied()
                    .unwrap_or(EMPTY_NEIGHBOR);
                let slot_off = off + slot * 4;
                buf[slot_off..slot_off + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
    }

    pub fn read_level(&self, buf: &[u8]) -> u8 {
        buf[0]
    }

    pub fn read_vector(&self, buf: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.num_dim);
        let v_off = 4;
        for j in 0..self.num_dim {
            let off = v_off + j * 4;
            out.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        out
    }

    pub fn read_neighbors(&self, buf: &[u8], layer: u8) -> Vec<u32> {
        let cap = Self::max_neighbors(layer);
        let off = self.neighbors_offset(layer);
        let mut out = Vec::with_capacity(cap);
        for slot in 0..cap {
            let slot_off = off + slot * 4;
            let id = u32::from_le_bytes(buf[slot_off..slot_off + 4].try_into().unwrap());
            if id != EMPTY_NEIGHBOR {
                out.push(id);
            }
        }
        out
    }

    /// Write one layer's neighbor slots in place without touching vector or other layers.
    pub fn write_neighbors_layer(&self, buf: &mut [u8], layer: u8, neighbors: &[u32]) {
        assert_eq!(buf.len(), self.record_stride);
        let cap = Self::max_neighbors(layer);
        let off = self.neighbors_offset(layer);
        for slot in 0..cap {
            let val = neighbors.get(slot).copied().unwrap_or(EMPTY_NEIGHBOR);
            let slot_off = off + slot * 4;
            buf[slot_off..slot_off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }

    pub fn l2_sq_from_record(&self, buf: &[u8], query: &[f32]) -> f32 {
        let v_off = 4;
        let mut sum = 0.0f32;
        for (j, &q) in query.iter().enumerate().take(self.num_dim) {
            let off = v_off + j * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let d = v - q;
            sum += d * d;
        }
        sum
    }

    pub fn patch_neighbor(buf: &mut [u8], layout: &NodeLayout, layer: u8, slot: usize, id: u32) {
        let off = layout.neighbors_offset(layer) + slot * 4;
        buf[off..off + 4].copy_from_slice(&id.to_le_bytes());
    }

    pub fn find_neighbor_slot(buf: &[u8], layout: &NodeLayout, layer: u8, target: u32) -> Option<usize> {
        let cap = Self::max_neighbors(layer);
        let off = layout.neighbors_offset(layer);
        for slot in 0..cap {
            let slot_off = off + slot * 4;
            let id = u32::from_le_bytes(buf[slot_off..slot_off + 4].try_into().unwrap());
            if id == target {
                return Some(slot);
            }
        }
        None
    }

    pub fn first_empty_neighbor_slot(buf: &[u8], layout: &NodeLayout, layer: u8) -> Option<usize> {
        let cap = Self::max_neighbors(layer);
        let off = layout.neighbors_offset(layer);
        for slot in 0..cap {
            let slot_off = off + slot * 4;
            let id = u32::from_le_bytes(buf[slot_off..slot_off + 4].try_into().unwrap());
            if id == EMPTY_NEIGHBOR {
                return Some(slot);
            }
        }
        None
    }
}

#[cfg(test)]
mod node_layout_tests {
    use super::*;
    use crate::disk_hnsw::params::{M, M0};

    #[test]
    fn node_layout_roundtrip_and_neighbor_helpers() {
        let layout = NodeLayout::new(4);
        assert!(layout.record_stride > 4 + 4 * 4);
        assert_eq!(layout.neighbors_offset(0), 4 + 16);
        assert_eq!(layout.neighbors_offset(1), 4 + 16 + M0 * 4);
        assert_eq!(NodeLayout::max_neighbors(0), M0);
        assert_eq!(NodeLayout::max_neighbors(1), M);

        let mut buf = vec![0u8; layout.record_stride];
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let neighbors = vec![vec![10u32, 11], vec![20u32]];
        layout.write_record(&mut buf, 2, &vector, &neighbors);
        assert_eq!(layout.read_level(&buf), 2);
        assert_eq!(layout.read_vector(&buf), vector);
        assert_eq!(layout.read_neighbors(&buf, 0), vec![10, 11]);
        assert_eq!(layout.read_neighbors(&buf, 1), vec![20]);

        NodeLayout::patch_neighbor(&mut buf, &layout, 0, 1, 99);
        assert_eq!(layout.read_neighbors(&buf, 0)[1], 99);
        assert_eq!(NodeLayout::find_neighbor_slot(&buf, &layout, 0, 99), Some(1));
        assert!(NodeLayout::first_empty_neighbor_slot(&buf, &layout, 1).is_some());
    }
}
