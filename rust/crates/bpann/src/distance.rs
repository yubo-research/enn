use ndarray::ArrayView1;

pub fn row_sq_l2(
    x: ArrayView1<f64>,
    y: ArrayView1<f64>,
    scale_x: bool,
    x_scale: ArrayView1<f64>,
) -> f64 {
    let acc = if scale_x {
        x.iter()
            .zip(y.iter())
            .zip(x_scale.iter())
            .fold(0.0, |acc, ((&xi, &yi), &sc)| {
                let d = xi / sc - yi / sc;
                acc + d * d
            })
    } else {
        x.iter().zip(y.iter()).fold(0.0, |acc, (&xi, &yi)| {
            let d = xi - yi;
            acc + d * d
        })
    };
    acc.max(0.0)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    while i + 32 <= n {
        for (u, slot) in acc.iter_mut().enumerate() {
            let d = _mm256_sub_ps(
                _mm256_loadu_ps(pa.add(i + 8 * u)),
                _mm256_loadu_ps(pb.add(i + 8 * u)),
            );
            *slot = _mm256_add_ps(*slot, _mm256_mul_ps(d, d));
        }
        i += 32;
    }
    let v = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), v);
    let mut sum: f32 = lanes.iter().sum();
    for j in i..n {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}

pub fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { l2_sq_f32_avx2(a, b) };
    }
    l2_sq_f32_body(a, b)
}

#[inline(always)]
fn l2_sq_f32_body(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut acc = [[0.0f32; 8]; 4];
    let mut i = 0;
    while i + 32 <= n {
        for (u, slot) in acc.iter_mut().enumerate() {
            for (l, s) in slot.iter_mut().enumerate() {
                let d = a[i + 8 * u + l] - b[i + 8 * u + l];
                *s += d * d;
            }
        }
        i += 32;
    }
    let mut sum = 0.0f32;
    for (((&p0, &p1), &p2), &p3) in acc[0].iter().zip(&acc[1]).zip(&acc[2]).zip(&acc[3]) {
        sum += (p0 + p1) + (p2 + p3);
    }
    for j in i..n {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq_l2_f64_abandon_avx2(
    query: &[f64],
    point: &[f64],
    scale_x: bool,
    x_scale: &[f64],
    limit: f64,
) -> Option<f64> {
    sq_l2_f64_abandon_body(query, point, scale_x, x_scale, limit)
}

pub fn sq_l2_f64_abandon(
    query: &[f64],
    point: &[f64],
    scale_x: bool,
    x_scale: &[f64],
    limit: f64,
) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { sq_l2_f64_abandon_avx2(query, point, scale_x, x_scale, limit) };
    }
    sq_l2_f64_abandon_body(query, point, scale_x, x_scale, limit)
}

#[inline(always)]
fn sq_l2_f64_abandon_body(
    query: &[f64],
    point: &[f64],
    scale_x: bool,
    x_scale: &[f64],
    limit: f64,
) -> Option<f64> {
    const LANES: usize = 8;
    const ABANDON_BLOCK: usize = 64;
    let mut acc = [0.0f64; LANES];
    let mut start = 0;
    while start < query.len() {
        let end = (start + ABANDON_BLOCK).min(query.len());
        let (q, p) = (&query[start..end], &point[start..end]);
        let (qc, qr) = q.as_chunks::<LANES>();
        let (pc, pr) = p.as_chunks::<LANES>();
        if scale_x {
            let (sc, sr) = x_scale[start..end].as_chunks::<LANES>();
            for ((cq, cp), cs) in qc.iter().zip(pc).zip(sc) {
                for l in 0..LANES {
                    let d = cq[l] - cp[l] / cs[l];
                    acc[l] += d * d;
                }
            }
            for ((&a, &b), &s) in qr.iter().zip(pr).zip(sr) {
                let d = a - b / s;
                acc[0] += d * d;
            }
        } else {
            for (cq, cp) in qc.iter().zip(pc) {
                for l in 0..LANES {
                    let d = cq[l] - cp[l];
                    acc[l] += d * d;
                }
            }
            for (&a, &b) in qr.iter().zip(pr) {
                let d = a - b;
                acc[0] += d * d;
            }
        }
        let total: f64 = acc.iter().sum();
        if total > limit {
            return None;
        }
        start = end;
    }
    Some(acc.iter().sum::<f64>().max(0.0))
}

pub fn bpann_row_to_f32(row: &[f64], scale_x: bool, x_scale: &[f64], out: &mut Vec<f32>) {
    out.clear();
    if scale_x {
        out.extend(
            row.iter()
                .zip(x_scale.iter())
                .map(|(&v, &s)| (v / s) as f32),
        );
    } else {
        out.extend(row.iter().map(|&v| v as f32));
    }
}

pub fn batched_sq_l2_f32(query: &[f32], rows: &[Vec<f32>]) -> Vec<f32> {
    rows.iter().map(|r| l2_sq_f32(query, r)).collect()
}

pub fn batched_sq_l2_f64_rows(
    query: &[f64],
    train_x: &crate::mmap_store::MmapColumnStore,
    row_ids: &[u32],
    scale_x: bool,
    x_scale: &[f64],
) -> Result<Vec<f64>, crate::error::BpannError> {
    let mut out = Vec::with_capacity(row_ids.len());
    for &id in row_ids {
        let row = train_x.mmap_row_slice(id as usize)?;
        out.push(row_sq_l2(
            ndarray::ArrayView1::from(query),
            ndarray::ArrayView1::from(row),
            scale_x,
            ndarray::ArrayView1::from(x_scale),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn distance_units_are_linked() {
        let mut buf = Vec::new();
        crate::distance::bpann_row_to_f32(&[1.0, 2.0], false, &[1.0, 1.0], &mut buf);
        assert_eq!(buf, vec![1.0, 2.0]);
        assert!((crate::distance::l2_sq_f32(&[0.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        let a = array![0.0, 0.0];
        let b = array![1.0, 0.0];
        let s = array![1.0, 1.0];
        let _ = crate::distance::row_sq_l2(a.view(), b.view(), false, s.view());
        let dir = TempDir::new().unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        store
            .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
            .unwrap();
        let _ = crate::distance::batched_sq_l2_f32(&[0.0, 0.0], &[vec![1.0, 0.0]]);
        let _ = crate::distance::batched_sq_l2_f64_rows(
            &[0.0, 0.0],
            &store,
            &[0, 1],
            false,
            &[1.0, 1.0],
        )
        .unwrap();
    }

    #[test]
    fn dispatched_l2_sq_f32_is_bit_identical_to_portable_body() {
        for d in [0usize, 1, 7, 8, 9, 63, 64, 203, 1000] {
            let a: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
            let b: Vec<f32> = (0..d).map(|i| (i as f32 * 0.11).cos() - 0.2).collect();
            assert_eq!(
                crate::distance::l2_sq_f32(&a, &b).to_bits(),
                crate::distance::l2_sq_f32_body(&a, &b).to_bits()
            );
        }
    }

    #[test]
    fn dispatched_f64_abandon_is_bit_identical_to_portable_body() {
        for d in [0usize, 1, 7, 8, 9, 63, 64, 65, 203, 1000] {
            let q: Vec<f64> = (0..d).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
            let p: Vec<f64> = (0..d).map(|i| (i as f64 * 0.11).cos() - 0.2).collect();
            let sc: Vec<f64> = (0..d).map(|i| 0.5 + (i % 7) as f64).collect();
            for scaled in [false, true] {
                for limit in [f64::INFINITY, 1.0, 50.0] {
                    let a = crate::distance::sq_l2_f64_abandon(&q, &p, scaled, &sc, limit);
                    let b = crate::distance::sq_l2_f64_abandon_body(&q, &p, scaled, &sc, limit);
                    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
                }
            }
        }
    }

    #[test]
    fn abandon_kernel_matches_serial_distance_and_respects_limit() {
        let d = 203;
        let q: Vec<f64> = (0..d).map(|i| (i as f64 * 0.37).sin()).collect();
        let p: Vec<f64> = (0..d).map(|i| (i as f64 * 0.11).cos() * 3.0).collect();
        let sc: Vec<f64> = (0..d).map(|i| 0.5 + (i % 7) as f64).collect();
        for scaled in [false, true] {
            let qs: Vec<f64> = if scaled {
                q.iter().zip(&sc).map(|(a, s)| a / s).collect()
            } else {
                q.clone()
            };
            let exact: f64 = qs
                .iter()
                .zip(&p)
                .zip(&sc)
                .map(|((a, b), s)| {
                    let x = if scaled { b / s } else { *b };
                    (a - x) * (a - x)
                })
                .sum();
            let got =
                crate::distance::sq_l2_f64_abandon(&qs, &p, scaled, &sc, f64::INFINITY).unwrap();
            assert!((got - exact).abs() <= 1e-12 * exact.max(1.0));
            assert!(
                crate::distance::sq_l2_f64_abandon(&qs, &p, scaled, &sc, exact * 0.5).is_none()
            );
            assert!(
                crate::distance::sq_l2_f64_abandon(&qs, &p, scaled, &sc, exact * 1.0001).is_some()
            );
        }
    }
}
