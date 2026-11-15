
use crate::index::tree_codes::{code_row_bytes, encode_row, CODE_PAGE_HEADER, ROW_HEADER};
use crate::index::tree_nib::{bf16_up, decode_nib_row, encode_nib_row, nib_resid};
use crate::index::tree_store::{TreeStore, SLOT_ENTRIES};

pub type RowSrc<'a> = Option<&'a dyn Fn(u32, &mut Vec<f32>) -> bool>;

impl TreeStore {
    pub(super) fn exact_moved(&self, ids: &[u32], src: RowSrc) -> Option<Vec<u8>> {
        let f = src.filter(|_| self.resid)?;
        let mut out = Vec::with_capacity(ids.len() * self.moved_rb());
        let mut v = Vec::with_capacity(self.num_dim);
        for &r in ids {
            v.clear();
            if !f(r, &mut v) || v.len() != self.num_dim {
                return None;
            }
            out.extend_from_slice(&0.0f32.to_le_bytes());
            v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        Some(out)
    }

    #[inline]
    pub fn resid(&self) -> bool {
        self.resid
    }

    #[inline]
    pub(super) fn code_hdr(&self) -> usize {
        CODE_PAGE_HEADER
            + if self.resid {
                code_row_bytes(self.num_dim)
            } else {
                0
            }
    }

    fn anchor_start(&self, id: u32) -> usize {
        self.code_page_range(id).start + CODE_PAGE_HEADER
    }

    pub(super) fn set_anchor(&mut self, id: u32, v: &[f32]) {
        let start = self.anchor_start(id);
        let rb = code_row_bytes(self.num_dim);
        encode_row(v, &mut self.codes[start..start + rb]);
    }

    pub fn anchor(&self, id: u32, out: &mut Vec<f32>) {
        let start = self.anchor_start(id);
        let row = &self.codes[start..start + code_row_bytes(self.num_dim)];
        let scale = f32::from_le_bytes([row[0], row[1], row[2], row[3]]);
        out.clear();
        out.extend(row[ROW_HEADER..].iter().map(|&c| scale * f32::from(c as i8)));
    }

    pub(super) fn encode_resid_row(&mut self, id: u32, slot: usize, v: &[f32]) {
        let mut a = Vec::with_capacity(self.num_dim);
        self.anchor(id, &mut a);
        let r: Vec<f32> = v.iter().zip(&a).map(|(x, y)| x - y).collect();
        let (rb, start) = (self.code_rb(), self.code_page_range(id).start + self.code_hdr());
        let at = start + slot * rb;
        encode_nib_row(&r, &mut self.codes[at..at + rb]);
    }

    pub(super) fn decode_leaf_resid(&self, id: u32, out: &mut Vec<f32>) {
        let d = self.num_dim;
        let mut a = Vec::with_capacity(d);
        self.anchor(id, &mut a);
        for slot in 0..self.len(id) {
            let from = out.len();
            decode_nib_row(self.code_row(id, slot), d, out);
            out[from..].iter_mut().zip(&a).for_each(|(x, y)| *x += y);
        }
    }

    pub fn moved_rb(&self) -> usize {
        if self.resid {
            4 + 4 * self.num_dim
        } else {
            self.code_rb()
        }
    }

    pub(super) fn leaf_moved_resid(&self, id: u32) -> Vec<u8> {
        let d = self.num_dim;
        let mut rows = Vec::with_capacity(self.len(id) * d);
        self.decode_leaf_resid(id, &mut rows);
        let mut out = Vec::with_capacity(self.len(id) * self.moved_rb());
        for (slot, v) in rows.chunks_exact(d).enumerate() {
            out.extend_from_slice(&nib_resid(self.code_row(id, slot)).to_le_bytes());
            v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
        }
        out
    }

    pub(super) fn set_leaf_resid(&mut self, id: u32, ids: &[u32], moved: &[u8]) {
        let (d, mrb) = (self.num_dim, self.moved_rb());
        assert!(
            self.is_leaf(id) && ids.len() <= SLOT_ENTRIES && moved.len() == ids.len() * mrb,
            "bad moved entries for leaf {id}"
        );
        let f = |b: &[u8]| f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let rows: Vec<(f32, Vec<f32>)> = moved
            .chunks_exact(mrb)
            .map(|m| (f(m), m[4..].chunks_exact(4).map(f).collect()))
            .collect();
        let mut mean = vec![0.0f32; d];
        for (_, v) in &rows {
            mean.iter_mut().zip(v).for_each(|(m, x)| *m += x);
        }
        let inv = 1.0 / rows.len().max(1) as f32;
        mean.iter_mut().for_each(|m| *m *= inv);
        self.set_anchor(id, &mean);
        let rb = self.code_rb();
        for (slot, (old, v)) in rows.iter().enumerate() {
            self.encode_resid_row(id, slot, v);
            let at = self.code_page_range(id).start + self.code_hdr() + slot * rb;
            let row = &mut self.codes[at..at + rb];
            let total = bf16_up(nib_resid(row) + old);
            row[4..6].copy_from_slice(&total.to_le_bytes());
        }
        self.set_codes_valid(id, true);
        self.set_ids_only(id, ids);
    }

    pub(super) fn shift_query(&self, id: u32, query: &[f32], out: &mut Vec<f32>) -> f32 {
        self.anchor(id, out);
        let mut n2 = 0.0f32;
        for (o, q) in out.iter_mut().zip(query) {
            *o = q - *o;
            n2 += *o * *o;
        }
        n2
    }
}

#[cfg(test)]
mod tests {
    use crate::index::tree_nib::{nib_sq_l2, set_code4_resid};

    #[test]
    fn resid_rows_keep_their_error_bound_through_splits() {
        set_code4_resid(true);
        let d = 100usize;
        let dir = tempfile::tempdir().unwrap();
        let v = |r: usize| -> Vec<f32> {
            (0..d)
                .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
                .collect()
        };
        let n0 = 200;
        let rows: Vec<u32> = (0..n0 as u32).collect();
        let mut vs: Vec<f32> = (0..n0).flat_map(v).collect();
        let mut t =
            crate::index::tree_bulk::bulk_build(&rows, &mut vs, d, dir.path().join("i")).unwrap();
        for r in n0..6 * n0 {
            t.insert_row(r as u32, &v(r)).unwrap();
        }
        set_code4_resid(false);
        let s = &t.store;
        assert!(s.resid());
        let q: Vec<f32> = (0..d).map(|j| (j % 5) as f32 * 0.7).collect();
        let (mut shifted, mut checked) = (Vec::new(), 0);
        for page in 0..s.num_pages() as u32 {
            if s.is_free(page) || !s.is_leaf(page) {
                continue;
            }
            let qn2 = s.shift_query(page, &q, &mut shifted);
            for (slot, &row) in s.ids(page).iter().enumerate() {
                let (d2, e) = nib_sq_l2(&shifted, qn2, s.code_row(page, slot));
                let exact: f32 = q.iter().zip(v(row as usize)).map(|(a, b)| (a - b) * (a - b)).sum();
                let gap = (d2.sqrt() - exact.sqrt()).abs();
                assert!(gap <= e * 1.001 + 1e-3, "row={row} gap={gap} err={e}");
                checked += 1;
            }
        }
        assert_eq!(checked, 6 * n0);
    }
}

#[cfg(test)]
mod profile {
    use rand::{Rng, SeedableRng};

    use crate::index::tree_nib::{encode_nib_row, nib_resid, nib_row_bytes};

    /// Diagnostic (report 0238): leaf slack vs radius on data near a 5-dim subspace, D=100, after
    /// inserting rows one by one. Mode from ENN_CODE4. Run:
    /// ENN_CODE4=k cargo nextest run --run-ignored only -E 'test(slack_profile)' --no-capture
    #[test]
    #[ignore]
    fn slack_profile() {
        let (d, m, n) = (100usize, 5usize, 10_000usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let mut basis: Vec<Vec<f32>> = Vec::new();
        for _ in 0..m {
            let mut b: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
            for o in &basis {
                let dot: f32 = b.iter().zip(o).map(|(x, y)| x * y).sum();
                b.iter_mut().zip(o).for_each(|(x, y)| *x -= dot * y);
            }
            let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            b.iter_mut().for_each(|x| *x /= nb);
            basis.push(b);
        }
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let z: Vec<f32> = (0..m).map(|_| rng.gen()).collect();
                (0..d)
                    .map(|j| {
                        z.iter().zip(&basis).map(|(a, b)| a * b[j]).sum::<f32>()
                            + 0.01 * (rng.gen::<f32>() - 0.5) * 3.46
                    })
                    .collect()
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let mut vs: Vec<f32> = xs[0].clone();
        let mut t =
            crate::index::tree_bulk::bulk_build(&[0], &mut vs, d, dir.path().join("i")).unwrap();
        let exact = |r: u32, out: &mut Vec<f32>| {
            out.extend_from_slice(&xs[r as usize]);
            true
        };
        let use_src = std::env::var("PROFILE_EXACT").is_ok_and(|v| v == "1");
        for (r, x) in xs.iter().enumerate().skip(1) {
            let src: super::RowSrc = if use_src { Some(&exact) } else { None };
            t.insert_row_src(r as u32, x, src).unwrap();
        }
        let s = &t.store;
        let (mut ratio, mut accum) = (Vec::new(), Vec::new());
        let mut a = Vec::new();
        for page in 0..s.num_pages() as u32 {
            if s.is_free(page) || !s.is_leaf(page) || s.len(page) == 0 {
                continue;
            }
            ratio.push(s.leaf_err(page) / s.radius(page).max(1e-9));
            if s.resid() {
                s.anchor(page, &mut a);
                for (slot, &row) in s.ids(page).iter().enumerate() {
                    let r: Vec<f32> = xs[row as usize].iter().zip(&a).map(|(x, y)| x - y).collect();
                    let mut buf = vec![0u8; nib_row_bytes(d)];
                    encode_nib_row(&r, &mut buf);
                    accum.push(nib_resid(s.code_row(page, slot)) / nib_resid(&buf).max(1e-9));
                }
            }
        }
        let med = |v: &mut Vec<f32>, q: f64| {
            v.sort_by(f32::total_cmp);
            v.get(((v.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(f32::NAN)
        };
        println!(
            "PROFILE exact_src={use_src} mode_resid={} nib={} leaves={} slack/radius p50={:.3} p90={:.3}; stored/fresh bound p50={:.2} p90={:.2} p99={:.2}",
            s.resid(), s.nib(), ratio.len(), med(&mut ratio, 0.5), med(&mut ratio, 0.9),
            med(&mut accum, 0.5), med(&mut accum, 0.9), med(&mut accum, 0.99)
        );
    }
}

#[cfg(test)]
mod refine_more_tests {
    use ndarray::Array2;
    use rand::{Rng, SeedableRng};

    use crate::index::tree_codes_search::{set_refine_more, take_coded_retries};
    use crate::index::tree_nib::set_code4_resid;
    use crate::BpannBackend;

    fn near_subspace(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut basis: Vec<Vec<f64>> = Vec::new();
        for _ in 0..5 {
            let mut b: Vec<f64> = (0..d).map(|_| rng.gen::<f64>() - 0.5).collect();
            for o in &basis {
                let dot: f64 = b.iter().zip(o).map(|(x, y)| x * y).sum();
                b.iter_mut().zip(o).for_each(|(x, y)| *x -= dot * y);
            }
            let norm = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            basis.push(b.into_iter().map(|x| x / norm).collect());
        }
        let mut x = Array2::zeros((n, d));
        for mut row in x.rows_mut() {
            let z: Vec<f64> = (0..5).map(|_| rng.gen::<f64>()).collect();
            for (j, v) in row.iter_mut().enumerate() {
                let noise: f64 = (0..12).map(|_| rng.gen::<f64>()).sum::<f64>() - 6.0;
                *v = basis.iter().zip(&z).map(|(b, zi)| b[j] * zi).sum::<f64>() + 0.01 * noise;
            }
        }
        x
    }

    fn brute(b: &BpannBackend, q: &[f64], k: usize) -> Vec<i64> {
        let mut s: Vec<(i64, f64)> = (0..b.len())
            .map(|i| {
                let r = b.mmap_row_slice(i).unwrap();
                (i as i64, q.iter().zip(r).map(|(a, c)| (a - c) * (a - c)).sum())
            })
            .collect();
        s.sort_by(|a, c| a.1.total_cmp(&c.1).then(a.0.cmp(&c.0)));
        s.into_iter().take(k).map(|p| p.0).collect()
    }

    fn retries_and_check(b: &BpannBackend, qs: &Array2<f64>, on: bool) -> u64 {
        set_refine_more(on);
        take_coded_retries();
        for q in qs.rows() {
            let (_, idx) = b.search(&q.insert_axis(ndarray::Axis(0)), 10, false).unwrap();
            let got: Vec<i64> = idx.row(0).to_vec();
            assert_eq!(got, brute(b, q.as_slice().unwrap(), 10), "refine_more={on}");
        }
        take_coded_retries()
    }

    #[test]
    fn refine_more_keeps_exact_neighbors_and_avoids_rewalks() {
        set_code4_resid(true);
        let (d, n) = (100usize, 3000usize);
        let dir = tempfile::tempdir().unwrap();
        let all = near_subspace(n + 20, d, 3);
        let x = all.slice(ndarray::s![..n, ..]).to_owned();
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), d, 1).unwrap();
        b.append_rows(&x.view(), &Array2::zeros((n, 1)).view(), None).unwrap();
        b.ensure_index_sync().unwrap();
        let qs = all.slice(ndarray::s![n.., ..]).to_owned();
        let off = retries_and_check(&b, &qs, false);
        let on = retries_and_check(&b, &qs, true);
        set_code4_resid(false);
        assert!(on < off, "re-walks with refine-more {on}, without {off}");
    }
}
