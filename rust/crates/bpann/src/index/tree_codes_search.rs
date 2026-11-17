use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::index::tree_codes::{code_extra, coded_sq_l2};
use crate::index::tree_search::{search_tree, search_tree_scored, F32_WALKS};
use crate::index::tree_store::{TreeStore, SLOT_ENTRIES};

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
    d2s: Vec<f32>,
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
    ) -> usize {
        let valid = s.codes_valid(page);
        if valid && !std::mem::take(&mut self.advised) {
            s.advise_codes(page);
        }
        debug_assert!(valid, "leaf {page} has no valid codes");
        let n = match s.leaf_id_tail(page) {
            Some(tail) => self.scan_rows(
                s,
                (page, query, qn2),
                offer,
                tail.chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .take_while(|&w| w != 0)
                    .map(|w| w - 1),
            ),
            None => self.scan_rows(s, (page, query, qn2), offer, s.ids(page).iter().copied()),
        };
        debug_assert_eq!(n, s.len(page), "leaf {page} id mirror is stale");
        n
    }

    #[inline(always)]
    fn scan_rows(
        &mut self,
        s: &TreeStore,
        at: (u32, &[f32], f32),
        offer: &mut impl FnMut(u32, f64),
        ids: impl Iterator<Item = u32>,
    ) -> usize {
        self.scan_with(s, at, offer, ids, coded_sq_l2)
    }

    #[inline(always)]
    fn scan_with(
        &mut self,
        s: &TreeStore,
        (page, query, qn2): (u32, &[f32], f32),
        offer: &mut impl FnMut(u32, f64),
        ids: impl Iterator<Item = u32>,
        dist: impl Fn(&[f32], f32, &[u8]) -> (f32, f32),
    ) -> usize {
        let rb = s.code_rb();
        let rows = s.code_rows(page, SLOT_ENTRIES);
        let mut n = 0usize;
        for ((slot, row), code) in ids.enumerate().zip(rows.chunks_exact(rb)) {
            let (d2, e) = dist(query, qn2, code);
            self.err_max = self.err_max.max(e);
            let i = self.locs.len() as u32;
            self.locs.push((row, page, slot as u32, e));
            self.d2s.push(d2);
            offer(i, f64::from(d2));
            n += 1;
        }
        n
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
    let stops_before = F32_WALKS.budget_stops.load(Ordering::Relaxed);
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
    let mut out = exact_rows(&need, query, store)?;
    let stopped = F32_WALKS.budget_stops.load(Ordering::Relaxed) > stops_before;
    if !stopped {
        let unvisited = if top.len() >= pool {
            top.last().map_or(0.0, |p| (p.1 as f32).sqrt())
        } else {
            f32::INFINITY
        };
        let done: std::collections::HashSet<u32> = need.iter().map(|n| n.0).collect();
        boundary = refine_more(&scan, &done, (k, unvisited), query, store, &mut out)?;
    }
    EXACT_ROWS.fetch_add(out.len() as u64, Ordering::Relaxed);
    out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let lower = f64::from(boundary);
    Ok((out, lower * lower))
}

fn exact_rows(
    need: &[(u32, u32, u32, f32)],
    query: &[f32],
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    store
        .train_x
        .prefetch_rows(&need.iter().map(|n| n.0).collect::<Vec<_>>());
    let orig = query.len();
    let mut buf = Vec::with_capacity(orig);
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(need.len());
    for &(row, _, _, _) in need {
        bpann_row_to_f32(
            store.train_x.mmap_row_slice(row as usize)?,
            store.scale_x,
            store.x_scale,
            &mut buf,
        );
        let n = orig.min(buf.len());
        out.push((row, l2_sq_f32(&query[..n], &buf[..n])));
    }
    Ok(out)
}

fn refine_more(
    scan: &CodedScan,
    done: &std::collections::HashSet<u32>,
    (k, unvisited): (usize, f32),
    query: &[f32],
    store: &MmapSearchStore<'_>,
    out: &mut Vec<(u32, f32)>,
) -> Result<f32, BpannError> {
    const REFINE_MORE_CAP: usize = 256;
    let mut rest: Vec<(f32, usize)> = (0..scan.locs.len())
        .filter(|&i| !done.contains(&scan.locs[i].0))
        .map(|i| ((scan.d2s[i].sqrt() - scan.locs[i].3).max(0.0), i))
        .filter(|p| p.0 < unvisited)
        .collect();
    rest.sort_by(|a, b| a.0.total_cmp(&b.0));
    let kth = |o: &Vec<(u32, f32)>| {
        let mut d: Vec<f32> = o.iter().map(|p| p.1).collect();
        d.sort_by(f32::total_cmp);
        d.get(k.saturating_sub(1))
            .map_or(f32::INFINITY, |v| v.sqrt())
    };
    let mut at = 0usize;
    while at < rest.len() && at < REFINE_MORE_CAP {
        let cut = kth(out);
        let end = rest[at..]
            .iter()
            .take(REFINE_MORE_CAP - at)
            .take_while(|p| p.0 <= cut)
            .count();
        if end == 0 {
            break;
        }
        let batch: Vec<_> = rest[at..at + end].iter().map(|p| scan.locs[p.1]).collect();
        out.extend(exact_rows(&batch, query, store)?);
        at += end;
    }
    Ok(rest.get(at).map_or(unvisited, |p| p.0.min(unvisited)))
}
