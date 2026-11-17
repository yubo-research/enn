use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::error::BpannError;
use crate::index::tree::nearest;
use crate::index::tree_codes::decode_header;
use crate::index::tree_codes_search::CodedScan;
use crate::index::tree_counts::dist;
use crate::index::tree_pbuf_all::charged_leaves;
use crate::index::tree_store::TreeStore;
use crate::index::tree_topk::TopK;

pub(super) const STALE_TAG: u8 = u8::MAX;
static TAG: AtomicU8 = AtomicU8::new(u8::MAX);
static SCANNED: AtomicU64 = AtomicU64::new(0);
static VISITED: AtomicU64 = AtomicU64::new(0);
static RETAGS: AtomicU64 = AtomicU64::new(0);
static ORPHANS: AtomicU64 = AtomicU64::new(0);

pub fn pbuf_tag() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    match TAG.load(Ordering::Relaxed) {
        u8::MAX => *ENV.get_or_init(|| std::env::var("ENN_PBUF_TAG").as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn set_pbuf_tag(on: bool) {
    TAG.store(u8::from(on), Ordering::Relaxed);
}

pub fn take_pbuf_tag_stats() -> (u64, u64, u64, u64) {
    (
        SCANNED.swap(0, Ordering::Relaxed),
        VISITED.swap(0, Ordering::Relaxed),
        RETAGS.swap(0, Ordering::Relaxed),
        ORPHANS.swap(0, Ordering::Relaxed),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TagPick<'a> {
    All,
    Stale(usize),
    Rows(&'a [u32]),
}

impl TagPick<'_> {
    pub(super) fn keeps(self, tag: u8) -> bool {
        match self {
            Self::All | Self::Rows(_) => true,
            Self::Stale(n) => tag as usize >= n,
        }
    }
}

#[derive(Default)]
pub(super) struct TagIndex {
    by_parent: HashMap<u32, (Vec<u32>, Vec<u32>)>,
    done: HashSet<(u32, usize)>,
}

impl TagIndex {
    fn rows(&mut self, s: &TreeStore, parent: u32, slot: usize) -> &[u32] {
        let (order, starts) = self
            .by_parent
            .entry(parent)
            .or_insert_with(|| group_by_tag(s, parent));
        if slot + 1 >= starts.len() {
            return &[];
        }
        &order[starts[slot] as usize..starts[slot + 1] as usize]
    }
}

fn group_by_tag(s: &TreeStore, parent: u32) -> (Vec<u32>, Vec<u32>) {
    let slots = s.len(parent);
    let tags: Vec<usize> = s
        .pbuf_tagged_records(parent)
        .map(|r| (r.2 as usize).min(slots))
        .collect();
    note_scan(tags.len(), 0);
    let mut starts = vec![0u32; slots + 2];
    for &t in &tags {
        starts[t + 1] += 1;
    }
    for i in 1..starts.len() {
        starts[i] += starts[i - 1];
    }
    let mut fill = starts.clone();
    let mut order = vec![0u32; tags.len()];
    for (i, &t) in tags.iter().enumerate() {
        order[fill[t] as usize] = i as u32;
        fill[t] += 1;
    }
    (order, starts)
}

pub(super) fn note_scan(visited: usize, scanned: usize) {
    VISITED.fetch_add(visited as u64, Ordering::Relaxed);
    SCANNED.fetch_add(scanned as u64, Ordering::Relaxed);
}

impl TreeStore {
    #[inline]
    pub fn pbuf_tagged(&self) -> bool {
        self.pbuf.tagged && self.pbuf_on()
    }

    pub(super) fn pbuf_tag_row(&mut self, id: u32, v: &[f32], slack: f32) -> u8 {
        if self.is_leaf(id) || self.len(id) == 0 {
            return STALE_TAG;
        }
        let dd = v.len();
        let block = self.block(id);
        let slot = nearest(block, v);
        let r = dist(v, &block[slot * dd..(slot + 1) * dd]) + slack;
        self.widen_route(id, slot, r);
        u8::try_from(slot).unwrap_or(STALE_TAG)
    }

    pub(super) fn pbuf_retag(&mut self, id: u32) {
        RETAGS.fetch_add(1, Ordering::Relaxed);
        let half = 0.5 * (self.num_dim() as f32).sqrt();
        let rows: Vec<(Vec<f32>, f32)> = self
            .pbuf_records(id)
            .map(|(_, code)| {
                let (scale, _, c) = decode_header(code);
                (
                    c.iter().map(|&x| scale * f32::from(x)).collect(),
                    half * scale,
                )
            })
            .collect();
        for (i, (v, slack)) in rows.iter().enumerate() {
            let tag = self.pbuf_tag_row(id, v, *slack);
            self.pbuf.set_tag(id, i, tag);
        }
    }
}

pub(super) struct TagWalk {
    on: bool,
    index: TagIndex,
}

impl TagWalk {
    pub(super) fn new(s: &TreeStore) -> Self {
        Self {
            on: s.pbuf_tagged(),
            index: TagIndex::default(),
        }
    }

    pub(super) fn popped(
        &mut self,
        s: &TreeStore,
        page_id: u32,
        query: (&[f32], f32),
        coded: Option<&mut CodedScan>,
        top: &mut TopK,
        score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
    ) -> Result<usize, BpannError> {
        if !self.on {
            return Ok(0);
        }
        let n = score_parent_pbuf(s, (page_id, &mut self.index), query, coded, top, score)?;
        Ok(charged_leaves(n))
    }

    pub(super) fn expanded(
        &self,
        s: &TreeStore,
        page_id: u32,
        query: (&[f32], f32),
        coded: Option<&mut CodedScan>,
        top: &mut TopK,
        score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
    ) -> Result<usize, BpannError> {
        let pick = if self.on {
            TagPick::Stale(s.len(page_id))
        } else {
            TagPick::All
        };
        let n =
            crate::index::tree_pbuf_scan::score_pbuf(s, page_id, query, coded, top, score, pick)?;
        Ok(charged_leaves(n))
    }
}

fn score_parent_pbuf(
    s: &TreeStore,
    (page_id, index): (u32, &mut TagIndex),
    query: (&[f32], f32),
    coded: Option<&mut CodedScan>,
    top: &mut TopK,
    score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
) -> Result<usize, BpannError> {
    let mut coded = coded;
    let (mut cur, mut n) = (page_id, 0usize);
    while let Some(p) = s.parent(cur).filter(|&p| !s.is_leaf(p)) {
        let Some(slot) = s.ids(p).iter().position(|&c| c == cur) else {
            ORPHANS.fetch_add(1, Ordering::Relaxed);
            break;
        };
        if !index.done.insert((p, slot)) {
            break;
        }
        if s.pbuf_len(p) > 0 {
            let rows = index.rows(s, p, slot);
            if !rows.is_empty() {
                n += crate::index::tree_pbuf_scan::score_pbuf(
                    s,
                    p,
                    query,
                    coded.as_deref_mut(),
                    top,
                    score,
                    TagPick::Rows(rows),
                )?;
            }
        }
        cur = p;
    }
    Ok(n)
}

#[cfg(test)]
#[path = "tree_pbuf_tag_tests.rs"]
mod tests;
