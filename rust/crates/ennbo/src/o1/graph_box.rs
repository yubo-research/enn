use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::graph_io::{open_rw, read_at, Items};
use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) fn empty_box(d: usize) -> Vec<f64> {
    let mut b = vec![f64::INFINITY; d];
    b.extend(std::iter::repeat_n(f64::NEG_INFINITY, d));
    b
}

pub(crate) fn union_into(dst: &mut [f64], src: &[f64]) {
    let d = dst.len() / 2;
    for k in 0..d {
        dst[k] = dst[k].min(src[k]);
        dst[d + k] = dst[d + k].max(src[d + k]);
    }
}

pub(crate) fn bbox(it: &Items, d: usize) -> Vec<f64> {
    let mut b = empty_box(d);
    for r in it.x.chunks_exact(d) {
        for k in 0..d {
            b[k] = b[k].min(r[k]);
            b[d + k] = b[d + k].max(r[k]);
        }
    }
    b
}

pub(crate) fn box_lb(q: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let d = q.len();
    let mut acc = 0.0;
    for k in 0..d {
        let gap = (b[k] - q[k]).max(q[k] - b[d + k]).max(0.0);
        acc += w[k] * gap * gap;
    }
    acc * (1.0 - 1e-12)
}

pub(crate) struct BoxFiles {
    dir: PathBuf,
    d: usize,
    files: Vec<File>,
}

impl BoxFiles {
    pub(crate) fn open(dir: &Path, d: usize, height: usize) -> Result<Self, ENNError> {
        let mut b = Self { dir: dir.to_path_buf(), d, files: Vec::new() };
        b.ensure(height)?;
        Ok(b)
    }

    pub(crate) fn ensure(&mut self, height: usize) -> Result<(), ENNError> {
        while self.files.len() < height {
            let l = self.files.len();
            self.files.push(open_rw(&self.dir.join(format!("o1g_box{l}.bin")))?);
        }
        Ok(())
    }

    pub(crate) fn read(&self, level: usize, id: u32) -> Result<Vec<f64>, ENNError> {
        let mut buf = vec![0u8; 16 * self.d];
        read_at(&self.files[level], &mut buf, id as u64)?;
        Ok(buf.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or_default())).collect())
    }

    pub(crate) fn write(&self, level: usize, id: u32, b: &[f64]) -> Result<(), ENNError> {
        let buf: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.files[level].write_all_at(&buf, id as u64 * buf.len() as u64).map_err(io_error)
    }

    pub(crate) fn union_of(&self, level: usize, ids: &[u32]) -> Result<Vec<f64>, ENNError> {
        let mut b = empty_box(self.d);
        for &c in ids {
            union_into(&mut b, &self.read(level, c)?);
        }
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_lower_bound_is_zero_inside_and_below_every_member_distance() {
        let mut it = Items::default();
        it.push(0, &[0.0, 0.0]);
        it.push(1, &[1.0, 2.0]);
        let b = bbox(&it, 2);
        assert_eq!(box_lb(&[0.5, 1.0], &b, &[1.0, 1.0]), 0.0);
        let lb = box_lb(&[3.0, -1.0], &b, &[2.0, 0.5]);
        assert!(lb <= 2.0 * 4.0 + 0.5 * 1.0 && lb > 8.0);
        let mut u = empty_box(2);
        union_into(&mut u, &b);
        assert_eq!(u, b);
    }
}
