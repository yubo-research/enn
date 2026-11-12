use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::index::tree_codes::{code_extra, coded_sq_l2};
use crate::index::tree_search::{search_tree, search_tree_scored, F32_WALKS};
use crate::index::tree_store::TreeStore;

static EXACT_ROWS: AtomicU64 = AtomicU64::new(0);
static CODED_RETRIES: AtomicU64 = AtomicU64::new(0);

pub fn take_coded_retries() -> u64 {
    CODED_RETRIES.swap(0, Ordering::Relaxed)
}

pub(super) fn note_coded_retry() {
    CODED_RETRIES.fetch_add(1, Ordering::Relaxed);
}

pub(super) type Candidates = (Vec<(u32, f32)>, f64);

pub(super) fn candidate_walk(
    tree: &Tree,
    scaled: &[f32],
    (k, extra, budget): (usize, usize, usize),
    coded: bool,
    store: &MmapSearchStore<'_>,
) -> Result<Candidates, BpannError> {
    let extra = if coded {
        extra.max(code_extra())
    } else {
        extra
    };
    let pool = k.saturating_add(extra).min(tree.header.indexed_rows);
    if coded {
        return Ok(search_tree_coded(tree, scaled, k, pool, budget));
    }
    let c = search_tree(tree, scaled, pool, budget, store)?;
    let boundary = c.last().map_or(0.0, |p| f64::from(p.1));
    Ok((c, boundary))
}

pub(super) fn refine_rows(
    query: &[f64],
    candidates: &[(u32, f32)],
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f64)>, BpannError> {
    let ids: Vec<u32> = candidates.iter().map(|p| p.0).collect();
    store.train_x.prefetch_rows(&ids);
    let mut refined: Vec<(u32, f64)> = candidates
        .iter()
        .map(|&(row, _)| {
            Ok((
                row,
                crate::distance::row_sq_l2(
                    ndarray::ArrayView1::from(query),
                    ndarray::ArrayView1::from(store.train_x.mmap_row_slice(row as usize)?),
                    store.scale_x,
                    ndarray::ArrayView1::from(store.x_scale),
                ),
            ))
        })
        .collect::<Result<_, BpannError>>()?;
    refined.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    Ok(refined)
}

pub fn take_exact_rows() -> u64 {
    EXACT_ROWS.swap(0, Ordering::Relaxed)
}

#[derive(Default)]
pub struct CodedScan {
    locs: Vec<(u32, u32, u32, f32)>,
    err_max: f32,
}

impl CodedScan {
    pub(super) fn scan_leaf(
        &mut self,
        s: &TreeStore,
        page: u32,
        query: &[f32],
        qn2: f32,
        offer: &mut impl FnMut(u32, f64),
    ) {
        let valid = s.codes_valid(page);
        if valid {
            s.advise_codes(page);
        }
        let d = s.num_dim();
        for (slot, &row) in s.ids(page).iter().enumerate() {
            let (d2, e) = if valid {
                coded_sq_l2(query, qn2, s.code_row(page, slot))
            } else {
                (
                    l2_sq_f32(query, &s.block(page)[slot * d..(slot + 1) * d]),
                    0.0,
                )
            };
            self.err_max = self.err_max.max(e);
            let i = self.locs.len() as u32;
            self.locs.push((row, page, slot as u32, e));
            offer(i, f64::from(d2));
        }
    }
}

pub fn search_tree_coded(
    tree: &Tree,
    query: &[f32],
    k: usize,
    pool: usize,
    budget: usize,
) -> Candidates {
    let mut scan = CodedScan::default();
    let top = search_tree_scored(
        tree,
        query,
        pool,
        budget,
        &F32_WALKS,
        Some(&mut scan),
        |_, _, _| Ok(None),
    )
    .unwrap_or_default();
    let s = &tree.store;
    let bounds: Vec<(f32, f32)> = top
        .iter()
        .map(|&(i, d2)| {
            let (r, e) = ((d2 as f32).sqrt(), scan.locs[i as usize].3);
            ((r - e).max(0.0), r + e)
        })
        .collect();
    let mut highs: Vec<f32> = bounds.iter().map(|b| b.1).collect();
    highs.sort_by(f32::total_cmp);
    let cut = highs
        .get(k.saturating_sub(1))
        .copied()
        .unwrap_or(f32::INFINITY);
    let mut boundary = if top.len() >= pool {
        let last = top.last().map_or(0.0, |p| (p.1 as f32).sqrt());
        (last - scan.err_max).max(0.0)
    } else {
        f32::INFINITY
    };
    let mut need = Vec::new();
    for (&(i, _), &(lo, _)) in top.iter().zip(&bounds) {
        if lo <= cut {
            need.push(scan.locs[i as usize]);
        } else {
            boundary = boundary.min(lo);
        }
    }
    for &(_, page, slot, _) in &need {
        s.prefetch_row(page, slot as usize);
    }
    let d = s.num_dim();
    let orig = query.len().min(d);
    let mut out: Vec<(u32, f32)> = need
        .iter()
        .map(|&(row, page, slot, _)| {
            let x = s.row(page, slot as usize);
            (row, l2_sq_f32(&query[..orig], &x[..orig]))
        })
        .collect();
    EXACT_ROWS.fetch_add(out.len() as u64, Ordering::Relaxed);
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let lower = f64::from(boundary);
    (out, lower * lower)
}

impl TreeStore {
    pub fn prefetch_row(&self, id: u32, slot: usize) {
        let off = id as usize * self.slot_words * 4 + slot * self.num_dim * 4;
        let _ = self
            .slots
            .advise_range(memmap2::Advice::WillNeed, off, self.num_dim * 4);
    }

    pub fn row(&self, id: u32, slot: usize) -> &[f32] {
        let d = self.num_dim;
        assert!(
            (id as usize) < self.num_pages && slot < self.len(id),
            "bad row"
        );
        let off = id as usize * self.slot_words + slot * d;
        unsafe { std::slice::from_raw_parts((self.slots.as_ptr() as *const f32).add(off), d) }
    }
}
