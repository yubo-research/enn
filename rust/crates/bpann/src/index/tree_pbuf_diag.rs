use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::distance::l2_sq_f32;
use crate::index::tree_codes::decode_header;
use crate::index::tree_store::TreeStore;

pub type PbufDiagRow = (u32, u8, u16, u8, u8, u8, u32, u32);

#[derive(Default)]
struct Diag {
    targets: Vec<u32>,
    popped: HashSet<u32>,
    query: Vec<f32>,
    out: Vec<PbufDiagRow>,
}

static DIAG: Mutex<Option<Diag>> = Mutex::new(None);
static ON: AtomicBool = AtomicBool::new(false);

fn with_diag<T>(f: impl FnOnce(&mut Diag) -> T) -> T {
    let mut g: MutexGuard<'_, Option<Diag>> = DIAG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(g.get_or_insert_with(Diag::default))
}

pub fn set_pbuf_diag_targets(rows: Vec<u32>) {
    ON.store(!rows.is_empty(), Ordering::Relaxed);
    with_diag(|d| {
        d.targets = rows;
        d.popped.clear();
    });
}

pub fn take_pbuf_diag() -> Vec<PbufDiagRow> {
    with_diag(|d| std::mem::take(&mut d.out))
}

fn active() -> bool {
    ON.load(Ordering::Relaxed) && with_diag(|d| !d.targets.is_empty())
}

pub(super) fn note_popped(page_id: u32, query: &[f32]) {
    if active() {
        with_diag(|d| {
            d.popped.insert(page_id);
            if d.query.is_empty() {
                d.query = query.to_vec();
            }
        });
    }
}

fn root_of(s: &TreeStore, mut page: u32) -> u32 {
    while let Some(p) = s.parent(page) {
        page = p;
    }
    page
}

#[derive(Clone, Copy)]
enum Place {
    Leaf(u32),
    Buffer(u32, u8, bool),
}

type Located = (HashMap<u32, (Place, u16, Vec<f32>)>, Vec<(u32, Vec<f32>)>);

fn decoded(code: &[u8]) -> Vec<f32> {
    let (scale, _, c) = decode_header(code);
    c.iter().map(|&x| scale * f32::from(x)).collect()
}

fn locate(s: &TreeStore, root: u32, want: &HashSet<u32>) -> Located {
    let d = s.num_dim();
    let mut found = HashMap::new();
    let mut cents = Vec::new();
    let mut stack = vec![(root, 0u16)];
    while let Some((page, depth)) = stack.pop() {
        if s.is_leaf(page) {
            let rows = s.leaf_rows(page);
            for (i, &r) in s.ids(page).iter().enumerate() {
                if want.contains(&r) {
                    let v = rows[i * d..(i + 1) * d].to_vec();
                    found.entry(r).or_insert((Place::Leaf(page), depth, v));
                }
            }
            continue;
        }
        let kids = s.ids(page);
        let leaf_parent = kids.first().is_some_and(|&c| s.is_leaf(c));
        for (r, code, tag) in s.pbuf_tagged_records(page) {
            if want.contains(&r) {
                found.entry(r).or_insert((
                    Place::Buffer(page, tag, leaf_parent),
                    depth,
                    decoded(code),
                ));
            }
        }
        let block = s.block(page);
        for (i, &c) in kids.iter().enumerate() {
            if s.is_leaf(c) {
                cents.push((c, block[i * d..(i + 1) * d].to_vec()));
            }
        }
        stack.extend(kids.iter().map(|&c| (c, depth + 1)));
    }
    (found, cents)
}

fn rank_of(owner: u32, point: &[f32], cents: &[(u32, Vec<f32>)]) -> u32 {
    let Some(own) = cents
        .iter()
        .find(|c| c.0 == owner)
        .map(|c| l2_sq_f32(point, &c.1))
    else {
        return u32::MAX;
    };
    cents
        .iter()
        .filter(|c| l2_sq_f32(point, &c.1) < own)
        .count() as u32
}

pub(super) fn end_walk(s: &TreeStore, done: &HashSet<(u32, usize)>) {
    if !active() {
        return;
    }
    with_diag(|d| {
        let Some(&first) = d.popped.iter().next() else {
            return;
        };
        let root = root_of(s, first);
        let query = std::mem::take(&mut d.query);
        let want: HashSet<u32> = d.targets.iter().copied().collect();
        let (found, cents) = locate(s, root, &want);
        let targets = std::mem::take(&mut d.targets);
        for &r in &targets {
            let Some((place, depth, v)) = found.get(&r) else {
                d.out.push((r, 9, 0, 0, 0, 0, u32::MAX, u32::MAX));
                continue;
            };
            let (kind, owner, tag, scanned) = match *place {
                Place::Leaf(p) => (0u8, Some(p), 0u8, 0u8),
                Place::Buffer(p, tag, lp) => (
                    1 + u8::from(lp) * 2,
                    s.ids(p).get(tag as usize).copied(),
                    tag,
                    u8::from(done.contains(&(p, tag as usize))),
                ),
            };
            let popped = owner.is_some_and(|c| d.popped.contains(&c));
            let (by_row, by_query) = owner.map_or((u32::MAX, u32::MAX), |o| {
                (rank_of(o, v, &cents), rank_of(o, &query, &cents))
            });
            d.out.push((
                r,
                kind,
                *depth,
                tag,
                scanned,
                u8::from(popped),
                by_row,
                by_query,
            ));
        }
        d.popped.clear();
    });
}

pub(super) fn leaf_centroids(s: &TreeStore, root: u32) -> Vec<(u32, Vec<f32>)> {
    locate(s, root, &HashSet::new()).1
}

pub(super) fn audit_rank(owner: u32, point: &[f32], cents: &[(u32, Vec<f32>)]) -> u32 {
    rank_of(owner, point, cents)
}

pub(super) fn note_audit(row: u32, chosen: u32, start: u32) {
    with_diag(|d| d.out.push((row, 7, 0, 0, 0, 0, chosen, start)));
}

#[cfg(test)]
#[path = "tree_pbuf_diag_tests.rs"]
mod tests;
