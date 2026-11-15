use std::sync::atomic::{AtomicU8, Ordering};

use crate::index::tree_codes::{
    code_row_bytes, coded_sq_l2, encode_row, page_bytes_for, CODE_PAGE_HEADER, ROW_HEADER,
};
use crate::index::tree_store::{TreeStore, SLOT_ENTRIES};


static CODE4: AtomicU8 = AtomicU8::new(2);

pub fn code4_enabled() -> bool {
    match CODE4.load(Ordering::Relaxed) {
        2 => {
            let on = std::env::var("ENN_CODE4").is_ok_and(|v| v == "1");
            CODE4.store(u8::from(on), Ordering::Relaxed);
            on
        }
        v => v != 0,
    }
}

pub fn set_code4(value: bool) {
    CODE4.store(u8::from(value), Ordering::Relaxed);
}

pub fn nib_row_bytes(num_dim: usize) -> usize {
    ROW_HEADER + num_dim.div_ceil(2)
}

pub fn encode_nib_row(v: &[f32], out: &mut [u8]) {
    let amax = v.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    let scale = if amax.is_finite() && amax > 0.0 {
        amax / 7.0
    } else {
        0.0
    };
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let h = v.len().div_ceil(2);
    let quant = |x: f32| -> (u8, f64, u32) {
        let c = (x * inv).round().clamp(-7.0, 7.0) as i8;
        let r = f64::from(x) - f64::from(scale) * f64::from(c);
        ((c + 8) as u8, r * r, (i32::from(c) * i32::from(c)) as u32)
    };
    let (mut resid2, mut sumsq) = (0.0f64, 0u32);
    let (lo_half, hi_half) = v.split_at(h);
    for (j, o) in out[ROW_HEADER..ROW_HEADER + h].iter_mut().enumerate() {
        let (lo, r_lo, s_lo) = quant(lo_half[j]);
        let (hi, r_hi, s_hi) = hi_half.get(j).map_or((8, 0.0, 0), |&x| quant(x));
        *o = lo | (hi << 4);
        resid2 += r_lo + r_hi;
        sumsq += s_lo + s_hi;
    }
    let resid = if scale > 0.0 {
        (resid2.sqrt() as f32) * 1.000_01 + 1e-6 * amax
    } else {
        0.0
    };
    out[..4].copy_from_slice(&scale.to_le_bytes());
    out[4..6].copy_from_slice(&bf16_up(resid).to_le_bytes());
    out[6..8].copy_from_slice(&(sumsq.min(u32::from(u16::MAX)) as u16).to_le_bytes());
}

fn bf16_up(x: f32) -> u16 {
    let b = x.max(0.0).to_bits();
    ((b >> 16) + u32::from(b & 0xFFFF != 0)) as u16
}

pub fn decode_nib_row(row: &[u8], num_dim: usize, out: &mut Vec<f32>) {
    let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
    let h = num_dim.div_ceil(2);
    let body = &row[ROW_HEADER..ROW_HEADER + h];
    let at = |nib: u8| scale * f32::from(nib as i8 - 8);
    out.extend(body.iter().map(|&b| at(b & 0x0F)));
    out.extend(body[..num_dim - h].iter().map(|&b| at(b >> 4)));
}

pub fn nib_resid(row: &[u8]) -> f32 {
    f32::from_bits(u32::from(u16::from_le_bytes([row[4], row[5]])) << 16)
}

fn nib_sumsq(row: &[u8]) -> f32 {
    f32::from(u16::from_le_bytes([row[6], row[7]]))
}

pub fn dot_f32_nib(q: &[f32], body: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { dot_f32_nib_avx2(q, body) };
    }
    dot_nib_tail(q, body, 0, 0.0)
}

fn dot_nib_tail(q: &[f32], body: &[u8], from: usize, mut dot: f32) -> f32 {
    let h = body.len();
    for (j, &b) in body.iter().enumerate().skip(from) {
        dot += q[j] * f32::from((b & 0x0F) as i8 - 8);
        if h + j < q.len() {
            dot += q[h + j] * f32::from((b >> 4) as i8 - 8);
        }
    }
    dot
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_nib_avx2(q: &[f32], body: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let h = body.len();
    let (pq, pb) = (q.as_ptr(), body.as_ptr());
    let (mask, eight) = (_mm256_set1_epi32(15), _mm256_set1_epi32(8));
    let (mut a_lo, mut a_hi) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut j = 0;
    while j + 8 <= h && h + j + 8 <= q.len() {
        let w = _mm256_cvtepu8_epi32(_mm_loadl_epi64(pb.add(j) as *const __m128i));
        let lo = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(w, mask), eight));
        let hi = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_srli_epi32(w, 4), eight));
        a_lo = _mm256_add_ps(a_lo, _mm256_mul_ps(_mm256_loadu_ps(pq.add(j)), lo));
        a_hi = _mm256_add_ps(a_hi, _mm256_mul_ps(_mm256_loadu_ps(pq.add(h + j)), hi));
        j += 8;
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(a_lo, a_hi));
    dot_nib_tail(q, body, j, lanes.iter().sum())
}

#[inline]
pub fn nib_sq_l2(query: &[f32], qn2: f32, row: &[u8]) -> (f32, f32) {
    let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
    let dot = dot_f32_nib(query, &row[ROW_HEADER..]);
    let d2 = (qn2 - 2.0 * scale * dot + scale * scale * nib_sumsq(row)).max(0.0);
    (d2, nib_resid(row))
}

#[inline]
pub fn nib_codes(num_dim: usize) -> bool {
    code4_enabled()
        && page_bytes_for(nib_row_bytes(num_dim)) < page_bytes_for(code_row_bytes(num_dim))
}

#[inline]
pub fn page_row_bytes(num_dim: usize) -> usize {
    if nib_codes(num_dim) {
        nib_row_bytes(num_dim)
    } else {
        code_row_bytes(num_dim)
    }
}

#[inline]
pub fn encode_page_row(v: &[f32], out: &mut [u8], nib: bool) {
    if nib {
        encode_nib_row(v, out)
    } else {
        encode_row(v, out)
    }
}

#[inline]
pub fn page_sq_l2(query: &[f32], qn2: f32, row: &[u8], nib: bool) -> (f32, f32) {
    if nib {
        nib_sq_l2(query, qn2, row)
    } else {
        coded_sq_l2(query, qn2, row)
    }
}

impl TreeStore {
    pub fn leaf_code_bytes(&self, id: u32) -> Vec<u8> {
        self.code_rows(id, self.len(id)).to_vec()
    }

    pub fn set_leaf_coded(&mut self, id: u32, ids: &[u32], rows: &[u8]) {
        let rb = self.code_rb();
        assert!(
            self.is_leaf(id) && ids.len() <= SLOT_ENTRIES && rows.len() == ids.len() * rb,
            "bad coded entries for leaf {id}"
        );
        let start = self.code_page_range(id).start + CODE_PAGE_HEADER;
        self.codes[start..start + rows.len()].copy_from_slice(rows);
        self.set_codes_valid(id, true);
        self.set_ids_only(id, ids);
    }

    pub fn nib(&self) -> bool {
        self.nib
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn wave(d: usize, a: usize, m: usize, div: f32, off: f32) -> Vec<f32> {
        (0..d).map(|i| ((i * a) % m) as f32 / div - off).collect()
    }

    #[test]
    fn nib_distance_is_within_its_residual_bound() {
        for d in [1, 7, 37, 100, 117] {
            let v = wave(d, 7919, 101, 13.0, 3.0);
            let q = wave(d, 104_729, 89, 11.0, 4.0);
            let mut row = vec![0u8; nib_row_bytes(d)];
            encode_nib_row(&v, &mut row);
            let qn2: f32 = q.iter().map(|x| x * x).sum();
            let (d2, err) = nib_sq_l2(&q, qn2, &row);
            let exact: f32 = q.iter().zip(&v).map(|(a, b)| (a - b) * (a - b)).sum();
            assert!((d2.sqrt() - exact.sqrt()).abs() <= err * 1.001 + 1e-3, "d={d}");
            let mut back = Vec::new();
            decode_nib_row(&row, d, &mut back);
            let resid: f32 = v.iter().zip(&back).map(|(a, b)| (a - b) * (a - b)).sum();
            assert!(resid.sqrt() <= err && err <= resid.sqrt() * 1.01 + 1e-4, "d={d}");
            let sq: f32 = back.iter().map(|x| (x / f32::from_le_bytes([row[0], row[1], row[2], row[3]])).powi(2)).sum();
            assert!((sq - nib_sumsq(&row)).abs() < 1e-2, "d={d}");
        }
    }

    #[test]
    fn nib_avx_dot_matches_scalar() {
        for d in [16, 33, 100, 117] {
            let q = wave(d, 31, 17, 3.0, 2.0);
            let v = wave(d, 13, 29, 2.0, 7.0);
            let mut row = vec![0u8; nib_row_bytes(d)];
            encode_nib_row(&v, &mut row);
            let body = &row[ROW_HEADER..];
            let a = dot_f32_nib(&q, body);
            let c = dot_nib_tail(&q, body, 0, 0.0);
            assert!((a - c).abs() < 1e-2, "d={d}");
        }
    }

    #[test]
    fn nib_page_only_when_it_saves_pages() {
        assert!(page_bytes_for(nib_row_bytes(100)) < page_bytes_for(code_row_bytes(100)));
        assert_eq!(page_bytes_for(nib_row_bytes(10)), page_bytes_for(code_row_bytes(10)));
        let mut row = vec![1u8; nib_row_bytes(4)];
        encode_nib_row(&[0.0; 4], &mut row);
        assert_eq!(nib_sq_l2(&[1.0, 0.0, 0.0, 0.0], 1.0, &row), (1.0, 0.0));
    }

    #[test]
    fn leaf_rows_keep_their_error_bound_through_splits() {
        for d in [10usize, 100] {
            let dir = tempfile::tempdir().unwrap();
            let v = |r: usize| -> Vec<f32> {
                (0..d)
                    .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
                    .collect()
            };
            let n0 = 200;
            let rows: Vec<u32> = (0..n0 as u32).collect();
            let mut vs: Vec<f32> = (0..n0).flat_map(v).collect();
            let mut t = crate::index::tree_bulk::bulk_build(&rows, &mut vs, d, dir.path().join("i"))
                .unwrap();
            for r in n0..6 * n0 {
                t.insert_row(r as u32, &v(r)).unwrap();
            }
            let q: Vec<f32> = (0..d).map(|j| (j % 5) as f32 * 0.7).collect();
            let qn2: f32 = q.iter().map(|x| x * x).sum();
            let s = &t.store;
            let mut checked = 0;
            for page in 0..s.num_pages() as u32 {
                if s.is_free(page) || !s.is_leaf(page) {
                    continue;
                }
                for (slot, &row) in s.ids(page).iter().enumerate() {
                    let (d2, e) = page_sq_l2(&q, qn2, s.code_row(page, slot), s.nib());
                    let exact: f32 =
                        q.iter().zip(v(row as usize)).map(|(a, b)| (a - b) * (a - b)).sum();
                    let gap = (d2.sqrt() - exact.sqrt()).abs();
                    assert!(gap <= e * 1.001 + 1e-3, "d={d} row={row} gap={gap} err={e}");
                    checked += 1;
                }
            }
            assert_eq!(checked, 6 * n0);
        }
    }
}
