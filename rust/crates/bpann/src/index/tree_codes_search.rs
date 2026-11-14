use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
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
        return search_tree_coded(tree, scaled, k, pool, budget, store);
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
    pub(super) advised: bool,
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
        if valid && !std::mem::take(&mut self.advised) {
            s.advise_codes(page);
        }
        debug_assert!(valid, "leaf {page} has no valid codes");
        for (slot, &row) in s.ids(page).iter().enumerate() {
            let (d2, e) = coded_sq_l2(query, qn2, s.code_row(page, slot));
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
    store: &MmapSearchStore<'_>,
) -> Result<Candidates, BpannError> {
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
    store
        .train_x
        .prefetch_rows(&need.iter().map(|n| n.0).collect::<Vec<_>>());
    let orig = query.len();
    let mut buf = Vec::with_capacity(orig);
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(need.len());
    for &(row, _, _, _) in &need {
        bpann_row_to_f32(
            store.train_x.mmap_row_slice(row as usize)?,
            store.scale_x,
            store.x_scale,
            &mut buf,
        );
        let n = orig.min(buf.len());
        out.push((row, l2_sq_f32(&query[..n], &buf[..n])));
    }
    EXACT_ROWS.fetch_add(out.len() as u64, Ordering::Relaxed);
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let lower = f64::from(boundary);
    Ok((out, lower * lower))
}
