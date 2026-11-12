use std::sync::atomic::{AtomicU64, Ordering};

use std::fs::{File, OpenOptions};
use std::path::Path;

use crate::error::BpannError;
use crate::index::tree_store::{TreeStore, CODES_FILE, SLOT_ENTRIES};
use crate::index::tree_store_files::{named_file, scratch_file};

const CODE_PAGE_HEADER: usize = 64;
const ROW_HEADER: usize = 8;
const VALID: u32 = 0xC0DE_0001;

pub const TREE_LEAF_CODES: bool = true;
pub const TREE_CODE_EXTRA: usize = 30;

static LEAF_CODES: AtomicU64 = AtomicU64::new(TREE_LEAF_CODES as u64);
static CODE_EXTRA: AtomicU64 = AtomicU64::new(TREE_CODE_EXTRA as u64);

pub fn leaf_codes() -> bool {
    LEAF_CODES.load(Ordering::Relaxed) != 0
}

pub fn set_leaf_codes(value: bool) {
    LEAF_CODES.store(u64::from(value), Ordering::Relaxed);
}

pub fn code_extra() -> usize {
    CODE_EXTRA.load(Ordering::Relaxed) as usize
}

pub fn set_code_extra(value: usize) {
    CODE_EXTRA.store(value as u64, Ordering::Relaxed);
}

pub fn code_row_bytes(num_dim: usize) -> usize {
    ROW_HEADER + num_dim
}

pub fn code_page_bytes(num_dim: usize) -> usize {
    (CODE_PAGE_HEADER + SLOT_ENTRIES * code_row_bytes(num_dim)).div_ceil(64) * 64
}

pub fn encode_row(v: &[f32], out: &mut [u8]) {
    let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if amax.is_finite() && amax > 0.0 {
        amax / 127.0
    } else {
        0.0
    };
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut norm2 = 0.0f64;
    for (o, &x) in out[ROW_HEADER..].iter_mut().zip(v) {
        let c = (x * inv).round().clamp(-127.0, 127.0) as i8;
        let r = f64::from(scale) * f64::from(c);
        norm2 += r * r;
        *o = c as u8;
    }
    out[..4].copy_from_slice(&scale.to_le_bytes());
    out[4..8].copy_from_slice(&(norm2 as f32).to_le_bytes());
}

pub fn decode_header(row: &[u8]) -> (f32, f32, &[i8]) {
    let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
    let norm2 = f32::from_le_bytes([row[4], row[5], row[6], row[7]]);
    let codes = &row[ROW_HEADER..];
    let codes = unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const i8, codes.len()) };
    (scale, norm2, codes)
}

#[inline]
pub fn coded_sq_l2(query: &[f32], qn2: f32, row: &[u8]) -> (f32, f32) {
    let (scale, norm2, codes) = decode_header(row);
    let dot = dot_f32_i8(query, codes);
    let d2 = (qn2 - 2.0 * scale * dot + norm2).max(0.0);
    (d2, 0.5 * scale * (codes.len() as f32).sqrt())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_i8_avx2(a: &[f32], b: &[i8]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (u, slot) in acc.iter_mut().enumerate() {
            let c8 = _mm_loadl_epi64(pb.add(i + 8 * u) as *const __m128i);
            let c = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c8));
            *slot = _mm256_add_ps(*slot, _mm256_mul_ps(_mm256_loadu_ps(pa.add(i + 8 * u)), c));
        }
        i += 32;
    }
    let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    let mut sum: f32 = lanes.iter().sum();
    for j in i..n {
        sum += a[j] * f32::from(b[j]);
    }
    sum
}

pub fn dot_f32_i8(a: &[f32], b: &[i8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { dot_f32_i8_avx2(a, b) };
    }
    a.iter().zip(b).map(|(&x, &c)| x * f32::from(c)).sum()
}

pub(super) fn open_codes(dir: &Path, bytes: usize) -> Option<(File, bool)> {
    let path = dir.join(CODES_FILE);
    if let Ok(f) = OpenOptions::new().read(true).write(true).open(&path) {
        if f.metadata().ok()?.len() as usize == bytes {
            return Some((f, false));
        }
    }
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .ok()?;
    Some((f, true))
}

pub(super) fn create_codes_file(dir: &Path, named: bool) -> Result<File, BpannError> {
    if named {
        named_file(dir, CODES_FILE)
    } else {
        scratch_file(dir, "tree_codes")
    }
}

impl TreeStore {
    fn code_page_range(&self, id: u32) -> std::ops::Range<usize> {
        let bytes = code_page_bytes(self.num_dim);
        let start = id as usize * bytes;
        start..start + bytes
    }

    fn code_page_mut(&mut self, id: u32) -> &mut [u8] {
        let r = self.code_page_range(id);
        &mut self.codes[r]
    }

    pub fn codes_valid(&self, id: u32) -> bool {
        let r = self.code_page_range(id);
        let h = &self.codes[r.start..r.start + 4];
        u32::from_le_bytes([h[0], h[1], h[2], h[3]]) == VALID
    }

    pub fn code_row(&self, id: u32, slot: usize) -> &[u8] {
        let rb = code_row_bytes(self.num_dim);
        let start = self.code_page_range(id).start + CODE_PAGE_HEADER + slot * rb;
        &self.codes[start..start + rb]
    }

    pub(super) fn set_codes_valid(&mut self, id: u32, valid: bool) {
        let word = if valid { VALID } else { 0 };
        self.code_page_mut(id)[..4].copy_from_slice(&word.to_le_bytes());
    }

    pub(super) fn encode_code_row(&mut self, id: u32, slot: usize, v: &[f32]) {
        let rb = code_row_bytes(self.num_dim);
        let start = CODE_PAGE_HEADER + slot * rb;
        encode_row(v, &mut self.code_page_mut(id)[start..start + rb]);
    }

    pub fn encode_page(&mut self, id: u32) {
        let d = self.num_dim;
        let n = self.len(id);
        let rows = self.block(id)[..n * d].to_vec();
        for (slot, v) in rows.chunks_exact(d).enumerate() {
            self.encode_code_row(id, slot, v);
        }
        self.set_codes_valid(id, true);
    }

    pub fn encode_invalid_pages(&mut self) {
        for id in 0..self.num_pages as u32 {
            if !self.codes_valid(id) {
                self.encode_page(id);
            }
        }
    }

    pub fn advise_codes(&self, id: u32) {
        let r = self.code_page_range(id);
        let used = CODE_PAGE_HEADER + self.len(id) * code_row_bytes(self.num_dim);
        let _ = self
            .codes
            .advise_range(memmap2::Advice::WillNeed, r.start, used.min(r.len()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coded_distance_is_within_its_error_bound() {
        let d = 37;
        let v: Vec<f32> = (0..d)
            .map(|i| ((i * 7919) % 101) as f32 / 13.0 - 3.0)
            .collect();
        let q: Vec<f32> = (0..d)
            .map(|i| ((i * 104_729) % 89) as f32 / 11.0 - 4.0)
            .collect();
        let mut row = vec![0u8; code_row_bytes(d)];
        encode_row(&v, &mut row);
        let qn2: f32 = q.iter().map(|x| x * x).sum();
        let (d2, err) = coded_sq_l2(&q, qn2, &row);
        let exact: f32 = q.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!((d2.sqrt() - exact.sqrt()).abs() <= err * 1.001 + 1e-4);
        assert!(err > 0.0);
    }

    #[test]
    fn zero_row_encodes_to_zero() {
        let mut row = vec![1u8; code_row_bytes(4)];
        encode_row(&[0.0; 4], &mut row);
        let (d2, err) = coded_sq_l2(&[1.0, 0.0, 0.0, 0.0], 1.0, &row);
        assert_eq!((d2, err), (1.0, 0.0));
    }

    #[test]
    fn avx_dot_matches_scalar() {
        let a: Vec<f32> = (0..70).map(|i| i as f32 * 0.5 - 9.0).collect();
        let b: Vec<i8> = (0..70).map(|i| (i * 3 % 255 - 127) as i8).collect();
        let s: f32 = a.iter().zip(&b).map(|(&x, &c)| x * f32::from(c)).sum();
        assert!((dot_f32_i8(&a, &b) - s).abs() < 1e-2);
    }
}
