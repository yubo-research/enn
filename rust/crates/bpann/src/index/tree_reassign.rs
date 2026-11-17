use std::sync::atomic::{AtomicUsize, Ordering};

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::split::weighted_mean;
use crate::index::tree::{Tree, TREE_LEAF_CAPACITY};
use crate::index::tree_counts::dist;

pub fn leaf_two_means() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_LEAF_SPLIT").as_deref() == Ok("2"))
}

static REASSIGN_M: AtomicUsize = AtomicUsize::new(usize::MAX);
static REASSIGN_EVERY: AtomicUsize = AtomicUsize::new(usize::MAX);

fn knob(cell: &AtomicUsize, var: &str) -> usize {
    let v = cell.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let e = std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    cell.store(e, Ordering::Relaxed);
    e
}

pub fn reassign_leaves() -> usize {
    if let Some((m, _)) = LOCAL.with(|c| c.get()) {
        return m;
    }
    knob(&REASSIGN_M, "ENN_REASSIGN")
}

pub fn reassign_every() -> u32 {
    if let Some((_, k)) = LOCAL.with(|c| c.get()) {
        return k as u32;
    }
    knob(&REASSIGN_EVERY, "ENN_REASSIGN_EVERY") as u32
}

thread_local! {
    static LOCAL: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
}

pub fn set_reassign(m: usize, every: usize) {
    LOCAL.with(|c| c.set(Some((m, every))));
}

static PROF: [AtomicUsize; 7] = [const { AtomicUsize::new(0) }; 7];

fn prof_on() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var_os("ENN_REASSIGN_PROF").is_some())
}

fn prof(i: usize, t: std::time::Instant) -> std::time::Instant {
    if prof_on() {
        PROF[i].fetch_add(t.elapsed().as_nanos() as usize, Ordering::Relaxed);
        if i == 0 && PROF[4].fetch_add(1, Ordering::Relaxed) % 2000 == 1999 {
            let v: Vec<usize> = PROF.iter().map(|a| a.load(Ordering::Relaxed)).collect();
            eprintln!("reassign_prof calls {} ns/call hood {} gather {} lloyd {} write {} rows/call {} moved/call {}",
                v[4], v[0] / v[4], v[1] / v[4], v[2] / v[4], v[3] / v[4], v[5] / v[4], v[6] / v[4]);
        }
    }
    std::time::Instant::now()
}

fn reassign_full() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_REASSIGN_FULL").as_deref() == Ok("1"))
}

pub(super) struct Hood {
    pub(super) leaves: Vec<u32>,
    pub(super) cents: Vec<Vec<f32>>,
}

impl Tree {
    fn slot_of(&self, child: u32) -> Option<(u32, usize)> {
        let p = self.store.parent(child)?;
        let slot = self.store.ids(p).iter().position(|&c| c == child)?;
        Some((p, slot))
    }

    pub(super) fn centroid_of(&self, leaf: u32) -> Option<Vec<f32>> {
        let (p, slot) = self.slot_of(leaf)?;
        let d = self.store.num_dim();
        Some(self.store.block(p)[slot * d..(slot + 1) * d].to_vec())
    }

    pub(super) fn hood(&self, a: u32, b: u32, m: usize) -> Option<Hood> {
        let s = &self.store;
        let d = s.num_dim();
        let p = s.parent(a)?;
        let top = s.parent(p).unwrap_or(p);
        let ca = self.centroid_of(a)?;
        let cb = self.centroid_of(b)?;
        let mut pages = vec![top];
        pages.extend(s.ids(top).iter().copied().filter(|&c| !s.is_leaf(c)));
        let mut scored: Vec<(f32, u32, u32, usize)> = Vec::new();
        for &pg in &pages {
            let block = s.block(pg);
            for (i, &l) in s.ids(pg).iter().enumerate() {
                if !s.is_leaf(l) {
                    continue;
                }
                let c = &block[i * d..(i + 1) * d];
                let key = if l == a || l == b {
                    -1.0
                } else {
                    l2_sq_f32(c, &ca).min(l2_sq_f32(c, &cb))
                };
                scored.push((key, l, pg, i));
            }
        }
        let k = m.max(2).min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |x, y| x.0.total_cmp(&y.0));
            scored.truncate(k);
        }
        scored.sort_by(|x, y| x.0.total_cmp(&y.0));
        let (leaves, cents) = scored
            .into_iter()
            .map(|(_, l, pg, i)| (l, s.block(pg)[i * d..(i + 1) * d].to_vec()))
            .unzip();
        Some(Hood { leaves, cents })
    }

    fn cover_up_many(&mut self, leaf: u32, rows: &[&[f32]], slack: f32, delta: isize) {
        let d = self.store.num_dim();
        let mut node = leaf;
        while let Some((p, slot)) = self.slot_of(node) {
            if node != leaf && !rows.is_empty() {
                let c = &self.store.block(p)[slot * d..(slot + 1) * d];
                let need = rows.iter().map(|v| dist(v, c)).fold(0.0, f32::max) + slack;
                if need > self.store.radius(node) {
                    self.store.set_radius(node, need);
                }
            }
            if delta != 0 {
                let cnt = self.store.count(p) as isize + delta;
                self.store.set_count(p, cnt.max(0) as usize);
            }
            node = p;
        }
    }

    pub(super) fn gather(&self, h: &Hood) -> Gathered {
        let s = &self.store;
        let mut g = Gathered {
            ids: Vec::new(),
            owner: Vec::new(),
            rows: Vec::new(),
            codes: Vec::new(),
            slack: 0.0,
        };
        for (i, &l) in h.leaves.iter().enumerate() {
            g.ids.extend_from_slice(s.ids(l));
            g.owner.extend(std::iter::repeat_n(i, s.len(l)));
            g.rows.extend_from_slice(&s.leaf_rows(l));
            g.codes.extend_from_slice(&s.leaf_code_bytes(l));
            g.slack = g.slack.max(s.leaf_err(l));
        }
        g
    }

    pub(super) fn reassign_after_split(&mut self, a: u32, b: u32) -> Result<(), BpannError> {
        let m = reassign_leaves();
        let s = &self.store;
        if m == 0 || s.pbuf_on() || s.resid() || s.nib() {
            return Ok(());
        }
        let t = std::time::Instant::now();
        let Some(h) = self.hood(a, b, m) else {
            return Ok(());
        };
        let t = prof(0, t);
        let g = self.gather(&h);
        let t = prof(1, t);
        let sizes: Vec<usize> = h.leaves.iter().map(|&l| self.store.len(l)).collect();
        let (dest, moved) = assign(&h, &g, sizes, self.store.num_dim(), reassign_full());
        if prof_on() {
            PROF[5].fetch_add(g.owner.len(), Ordering::Relaxed);
            PROF[6].fetch_add(moved, Ordering::Relaxed);
        }
        let t = prof(2, t);
        if moved > 0 {
            self.rewrite(&h, &g, &dest);
        }
        prof(3, t);
        Ok(())
    }

    pub(super) fn rewrite(&mut self, h: &Hood, g: &Gathered, dest: &[usize]) {
        let d = self.store.num_dim();
        let rb = self.store.moved_rb();
        let k = h.leaves.len();
        let n = dest.len();
        let ones = vec![1.0f64; n];
        let points: Vec<&[f32]> = g.rows.chunks_exact(d).collect();
        let mut net = vec![0isize; k];
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut changed = vec![false; k];
        for r in 0..n {
            groups[dest[r]].push(r);
            if dest[r] != g.owner[r] {
                net[dest[r]] += 1;
                net[g.owner[r]] -= 1;
                changed[dest[r]] = true;
                changed[g.owner[r]] = true;
            }
        }
        for (j, &l) in h.leaves.iter().enumerate() {
            if !changed[j] {
                continue;
            }
            let group = &groups[j];
            let c = weighted_mean(&points, &ones, group);
            let rad = group
                .iter()
                .map(|&r| dist(points[r], &c))
                .fold(0.0, f32::max)
                + g.slack;
            let gid: Vec<u32> = group.iter().map(|&r| g.ids[r]).collect();
            let bytes: Vec<u8> = group
                .iter()
                .flat_map(|&r| g.codes[r * rb..(r + 1) * rb].iter().copied())
                .collect();
            self.store.set_leaf_coded(l, &gid, &bytes);
            self.store.set_count(l, group.len());
            if let Some((p, slot)) = self.slot_of(l) {
                self.store.entry_mut(p, slot).copy_from_slice(&c);
            }
            self.store.set_radius(l, rad);
            let incoming: Vec<&[f32]> = group
                .iter()
                .filter(|&&r| g.owner[r] != j)
                .map(|&r| points[r])
                .collect();
            self.cover_up_many(l, &incoming, g.slack, net[j]);
        }
    }
}

pub(super) struct Gathered {
    ids: Vec<u32>,
    owner: Vec<usize>,
    rows: Vec<f32>,
    codes: Vec<u8>,
    slack: f32,
}

fn centroid_gaps(cents: &[Vec<f32>]) -> Vec<f32> {
    let k = cents.len();
    let mut cc = vec![0.0f32; k * k];
    for i in 0..k {
        for j in (i + 1)..k {
            let x = l2_sq_f32(&cents[i], &cents[j]);
            cc[i * k + j] = x;
            cc[j * k + i] = x;
        }
    }
    cc
}

pub(super) fn assign(
    h: &Hood,
    g: &Gathered,
    mut size: Vec<usize>,
    d: usize,
    full: bool,
) -> (Vec<usize>, usize) {
    let k = h.cents.len();
    let cc = centroid_gaps(&h.cents);
    let mut dest = g.owner.clone();
    let mut moved = 0usize;
    for (r, v) in g.rows.chunks_exact(d).enumerate() {
        let o = g.owner[r];
        let (mut best, mut bd) = (o, l2_sq_f32(v, &h.cents[o]));
        let wide = full || o < 2;
        for (j, c) in h.cents.iter().enumerate() {
            if j == o
                || (!wide && j >= 2)
                || size[j] >= TREE_LEAF_CAPACITY
                || cc[o * k + j] >= 4.0 * bd
            {
                continue;
            }
            let dj = l2_sq_f32(v, c);
            if dj < bd {
                (best, bd) = (j, dj);
            }
        }
        if best != o && size[o] > 1 {
            size[o] -= 1;
            size[best] += 1;
            dest[r] = best;
            moved += 1;
        }
    }
    (dest, moved)
}

#[cfg(test)]
#[path = "tree_reassign_tests.rs"]
mod tests;
