use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) const CODE_BLOCK: usize = 32;
const LEVELS: f64 = 255.0;

fn block_bytes(ncols: usize) -> usize {
    8 * ncols + CODE_BLOCK * ncols
}

pub(crate) fn code_file_bytes(rows: usize, ncols: usize) -> usize {
    let tail = rows % CODE_BLOCK;
    (rows / CODE_BLOCK) * block_bytes(ncols) + if tail > 0 { 8 * ncols + tail * ncols } else { 0 }
}

pub(crate) struct CodeWriter {
    out: BufWriter<File>,
    ncols: usize,
    pending: Vec<f64>,
}

impl CodeWriter {
    pub(crate) fn create(path: &Path, ncols: usize) -> Result<Self, ENNError> {
        let file = File::create(path).map_err(io_error)?;
        Ok(Self { out: BufWriter::new(file), ncols, pending: Vec::with_capacity(CODE_BLOCK * ncols) })
    }

    pub(crate) fn push(&mut self, row: &[f64]) -> Result<(), ENNError> {
        self.pending.extend_from_slice(row);
        if self.pending.len() == CODE_BLOCK * self.ncols {
            self.write_block()?;
        }
        Ok(())
    }

    fn write_block(&mut self) -> Result<(), ENNError> {
        let d = self.ncols;
        let n = self.pending.len() / d.max(1);
        if n == 0 {
            return Ok(());
        }
        let (mut lo, mut hi) = (vec![f64::INFINITY; d], vec![f64::NEG_INFINITY; d]);
        for row in self.pending.chunks_exact(d) {
            for j in 0..d {
                lo[j] = lo[j].min(row[j]);
                hi[j] = hi[j].max(row[j]);
            }
        }
        let lo32: Vec<f32> = lo.iter().map(|&v| v as f32).collect();
        let step: Vec<f32> = (0..d).map(|j| ((hi[j] - f64::from(lo32[j])).max(0.0) / LEVELS) as f32).collect();
        let mut bytes = Vec::with_capacity(8 * d + n * d);
        lo32.iter().chain(&step).for_each(|v| bytes.extend_from_slice(&v.to_le_bytes()));
        for row in self.pending.chunks_exact(d) {
            for j in 0..d {
                let s = f64::from(step[j]);
                let c = if s > 0.0 { ((row[j] - f64::from(lo32[j])) / s).round().clamp(0.0, LEVELS) } else { 0.0 };
                bytes.push(c as u8);
            }
        }
        self.out.write_all(&bytes).map_err(io_error)?;
        self.pending.clear();
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(), ENNError> {
        self.write_block()?;
        self.out.flush().map_err(io_error)?;
        self.out.get_ref().sync_data().map_err(io_error)
    }
}

pub(crate) struct RowCodes {
    map: Mmap,
    ncols: usize,
}

impl RowCodes {
    pub(crate) fn open(path: &Path, ncols: usize, rows: usize) -> Option<Self> {
        let file = File::open(path).ok()?;
        let map = unsafe { Mmap::map(&file) }.ok()?;
        (map.len() == code_file_bytes(rows, ncols) && ncols > 0).then_some(Self { map, ncols })
    }

    fn floats(&self, at: usize, n: usize) -> &[f32] {
        let bytes = &self.map[at..at + 4 * n];
        let (head, mid, _) = unsafe { bytes.align_to::<f32>() };
        if head.is_empty() && mid.len() == n { mid } else { &[] }
    }

    pub(crate) fn dist2(&self, slot: usize, query: &[f32], weights: &[f32]) -> f64 {
        let d = self.ncols;
        let base = (slot / CODE_BLOCK) * block_bytes(d);
        let (lo, step) = (self.floats(base, d), self.floats(base + 4 * d, d));
        let code = &self.map[base + 8 * d + (slot % CODE_BLOCK) * d..][..d];
        if lo.len() != d || step.len() != d {
            return f64::INFINITY;
        }
        let mut acc = [0.0f32; 8];
        let body = d - d % 8;
        for (((((q, w), l), s), c), _) in query[..body].chunks_exact(8).zip(weights[..body].chunks_exact(8)).zip(lo[..body].chunks_exact(8)).zip(step[..body].chunks_exact(8)).zip(code[..body].chunks_exact(8)).zip(0..) {
            for k in 0..8 {
                let e = q[k] - l[k] - s[k] * f32::from(c[k]);
                acc[k] += w[k] * e * e;
            }
        }
        let tail: f32 = (body..d).map(|j| weights[j] * (query[j] - lo[j] - step[j] * f32::from(code[j])).powi(2)).sum();
        f64::from(acc.iter().sum::<f32>() + tail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_codes_track_each_block_range_and_reject_wrong_lengths() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("c.bin");
        let mut w = CodeWriter::create(&path, 2).unwrap();
        let rows: Vec<[f64; 2]> = (0..40).map(|i| if i < 32 { [1e-6 * i as f64, 5.0] } else { [100.0 + i as f64, -3.0] }).collect();
        rows.iter().for_each(|r| w.push(r).unwrap());
        w.finish().unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, code_file_bytes(40, 2));
        assert!(RowCodes::open(&path, 2, 41).is_none());
        let codes = RowCodes::open(&path, 2, 40).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let err = codes.dist2(i, &[r[0] as f32, r[1] as f32], &[1.0, 1.0]).sqrt();
            assert!(err < if i < 32 { 1e-5 } else { 0.02 }, "row {i} err {err}");
        }
        assert!((codes.dist2(3, &[0.0, 0.0], &[0.0, 1.0]) - 25.0).abs() < 1e-5);
        assert!((codes.dist2(35, &[0.0, 0.0], &[0.0, 1.0]) - 9.0).abs() < 1e-3);
    }
}
