//! In-place re-partitioning of the incremental tree, a little per added row.
//!
//! After a metric change every stored coordinate is exact, but the partition (which
//! rows share a page) was chosen under the old metric. No row is taken out and
//! inserted again from the root. Instead [`Morph`] sweeps the internal pages and, at
//! each, re-partitions what lies below it: the row ids of its leaves are pooled and
//! dealt out again among those leaves, and the child page references of its internal
//! children are pooled and dealt out again among those pages. Both use recursive
//! balanced bisection at the median of a projection, the rule a bulk build uses
//! ([`crate::index::tree_bulk`]), and a regrouping is applied only if it lowers the
//! scatter of what it regroups, so a tree that already fits the metric is left as it
//! is. Only references move, no page is created or deleted,
//! and the visited page keeps the same rows, so nothing above it changes. The
//! regrouped pages get counts, centroids and radii recomputed from what now lies below
//! them, which also undoes the loosening done by [`TreeCounts::scale_rows`].
//!
//! A visit regroups at most `MORPH_POOL_LEAVES * TREE_LEAF_CAPACITY` rows and
//! `TREE_FANOUT^2` page references, independent of `N`, and each added row pays for
//! [`MORPH_WORK_PER_ADD`] of them, so an add stays `O(log N)` and the tree answers
//! exact queries between any two adds. Rows regroup within a page's leaves and whole
//! subtrees regroup between pages, so rows reach distant leaves over successive laps.

use crate::distance::l2_sq_f32;
use crate::index::split::weighted_mean;
use crate::index::tree::{Tree, TREE_FANOUT, TREE_LEAF_CAPACITY};
use crate::index::tree_counts::dist;

/// Sweeps over the page slots scheduled by one [`Morph::schedule`]. Uniform 12-d with
/// ten dimensions shrunk 1000x: after 16 laps queries are about 2x slower than on a
/// bulk-built tree at 100k and 400k rows (about 100x faster than before the sweep);
/// more laps do not help, fewer leave them slower.
pub const MORPH_LAPS: usize = 16;

/// Rows and page references regrouped per added row while a sweep is in progress.
pub const MORPH_WORK_PER_ADD: usize = 6 * TREE_LEAF_CAPACITY;
/// A visit pools the rows of every leaf below the page when there are at most this
/// many, so pages with few leaf children (a bulk-built tree has many with two) still
/// regroup rows across a wide neighborhood.
pub const MORPH_POOL_LEAVES: usize = 2 * TREE_FANOUT;

/// Progress of the sweep over the page slots.
#[derive(Clone, Debug, Default)]
pub struct Morph {
    cursor: usize,
    slots_left: usize,
    laps_left: usize,
    credit: isize,
}

impl Morph {
    /// Sweep the page slots [`MORPH_LAPS`] times, starting from the cursor.
    pub fn schedule(&mut self, tree: &Tree) {
        self.laps_left = MORPH_LAPS;
        self.slots_left = tree.num_pages();
    }

    pub fn active(&self) -> bool {
        self.slots_left > 0
    }

    /// During a sweep, re-partition below the internal pages under the cursor until
    /// [`MORPH_WORK_PER_ADD`] items (on average) have been regrouped.
    pub fn advance(&mut self, tree: &mut Tree) {
        if !self.active() {
            return;
        }
        self.credit += MORPH_WORK_PER_ADD as isize;
        while self.credit > 0 {
            let Some(page) = self.next_internal(tree) else { break };
            let covered = tree.store.parent(page).is_some_and(|p| small_subtree_leaves(tree, p).is_some());
            let work = if covered {
                0
            } else {
                repartition_leaves(tree, page) + repartition_internal(tree, page)
            };
            self.credit -= work.max(1) as isize;
        }
        if !self.active() {
            self.credit = 0;
        }
    }

    fn next_internal(&mut self, tree: &Tree) -> Option<u32> {
        let num_pages = tree.num_pages();
        while self.slots_left > 0 && num_pages > 0 {
            if self.cursor >= num_pages {
                self.cursor = 0;
            }
            let page_id = self.cursor as u32;
            self.cursor += 1;
            self.slots_left -= 1;
            if self.slots_left == 0 {
                self.laps_left -= 1;
                self.slots_left = if self.laps_left > 0 { num_pages } else { 0 };
            }
            if !tree.store.is_leaf(page_id) {
                return Some(page_id);
            }
        }
        None
    }
}

/// Slots and ids of the children of `page` that are leaves (`leaves`) or not.
fn children_of_kind(tree: &Tree, page: u32, leaves: bool) -> Vec<(usize, u32)> {
    let s = &tree.store;
    s.ids(page).iter().copied().enumerate().filter(|&(_, k)| s.is_leaf(k) == leaves).collect()
}

/// Every leaf below `page`, left to right, or `None` if there are more than
/// [`MORPH_POOL_LEAVES`].
fn small_subtree_leaves(tree: &Tree, page: u32) -> Option<Vec<u32>> {
    let (mut out, mut stack) = (Vec::new(), vec![page]);
    while let Some(p) = stack.pop() {
        if !tree.store.is_leaf(p) {
            stack.extend(tree.store.ids(p).iter().rev());
        } else if out.len() == MORPH_POOL_LEAVES {
            return None;
        } else {
            out.push(p);
        }
    }
    Some(out)
}

/// Recompute the count, radius and centroid (in `parent`, at `slot`) of internal
/// page `page` and of the internal pages below it from their children.
fn refresh(tree: &mut Tree, page: u32, (parent, slot): (u32, usize)) {
    for (s, k) in children_of_kind(tree, page, false) {
        refresh(tree, k, (page, s));
    }
    let st = &tree.store;
    let kids = st.ids(page);
    let cents: Vec<&[f32]> = st.block(page).chunks_exact(st.num_dim()).collect();
    let weights: Vec<f64> = kids.iter().map(|&k| st.count(k) as f64).collect();
    let c = weighted_mean(&cents, &weights, &(0..kids.len()).collect::<Vec<_>>());
    let r = cents.iter().zip(kids).map(|(x, &k)| dist(x, &c) + st.radius(k)).fold(0.0, f32::max);
    let n = kids.iter().map(|&k| st.count(k)).sum();
    let st = &mut tree.store;
    st.set_count(page, n);
    st.set_radius(page, r);
    st.entry_mut(parent, slot).copy_from_slice(&c);
}

fn slot_in_parent(tree: &Tree, page: u32) -> Option<(u32, usize)> {
    let parent = tree.store.parent(page)?;
    Some((parent, tree.store.ids(parent).iter().position(|&k| k == page)?))
}

/// Split `members` (positions into `points`) into the `cut` with the lowest projection
/// onto the line through two far-apart members, and the rest.
fn split_at(points: &[&[f32]], members: &[usize], cut: usize) -> (Vec<usize>, Vec<usize>) {
    let farthest = |anchor: &[f32]| {
        let far = members.iter().map(|&i| (l2_sq_f32(points[i], anchor), i)).max_by(|x, y| x.0.total_cmp(&y.0));
        points[far.expect("members").1]
    };
    let a = farthest(points[members[0]]);
    let b = farthest(a);
    let dir: Vec<f32> = b.iter().zip(a).map(|(&x, &y)| x - y).collect();
    let mut proj: Vec<(f32, usize)> =
        members.iter().map(|&i| (points[i].iter().zip(&dir).map(|(&x, &d)| x * d).sum(), i)).collect();
    proj.select_nth_unstable_by(cut, |x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
    let right = proj.split_off(cut);
    (proj.into_iter().map(|p| p.1).collect(), right.into_iter().map(|p| p.1).collect())
}

/// Split `members` (at least `parts`) into `parts` groups whose sizes differ by at
/// most one, by recursive bisection.
fn bisect(points: &[&[f32]], members: Vec<usize>, parts: usize, out: &mut Vec<Vec<usize>>) {
    if parts <= 1 {
        out.push(members);
        return;
    }
    let left_parts = parts / 2;
    let (left, right) = split_at(points, &members, members.len() * left_parts / parts);
    bisect(points, left, left_parts, out);
    bisect(points, right, parts - left_parts, out);
}

fn groups_of(points: &[&[f32]], parts: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(parts);
    bisect(points, (0..points.len()).collect(), parts, &mut out);
    out
}

/// Weighted sum of squared distances of `points` to the weighted mean of their group.
/// For child centroids weighted by row counts, this is the part of the rows' squared
/// distances to their page mean that regrouping the children can change.
fn scatter(points: &[&[f32]], weights: &[f64], groups: &[Vec<usize>]) -> f64 {
    let group_scatter = |g: &Vec<usize>| {
        let c = weighted_mean(points, weights, g);
        g.iter().map(|&i| weights[i] * f64::from(l2_sq_f32(points[i], &c))).sum::<f64>()
    };
    groups.iter().map(group_scatter).sum()
}

/// Groups of consecutive positions with the given sizes.
fn runs(sizes: impl Iterator<Item = usize>) -> Vec<Vec<usize>> {
    let mut start = 0;
    sizes
        .map(|n| {
            start += n;
            (start - n..start).collect()
        })
        .collect()
}

/// Pool the rows of every leaf below `page` if there are at most
/// [`MORPH_POOL_LEAVES`] (else of its leaf children) and deal them out again among
/// those leaves; returns the number of rows regrouped.
fn repartition_leaves(tree: &mut Tree, page: u32) -> usize {
    let d = tree.store.num_dim();
    let deep = small_subtree_leaves(tree, page);
    let leaves = deep.clone().unwrap_or_else(|| children_of_kind(tree, page, true).iter().map(|l| l.1).collect());
    let rows: Vec<u32> = leaves.iter().flat_map(|&l| tree.store.ids(l).to_vec()).collect();
    let vs: Vec<f32> = leaves.iter().flat_map(|&l| tree.store.block(l).to_vec()).collect();
    if leaves.len() < 2 || rows.len() < leaves.len() || !tree.rows_cached() {
        return 0;
    }
    let points: Vec<&[f32]> = vs.chunks_exact(d).collect();
    let ones = vec![1.0; points.len()];
    let groups = groups_of(&points, leaves.len());
    let now = runs(leaves.iter().map(|&l| tree.store.len(l)));
    if scatter(&points, &ones, &groups) >= scatter(&points, &ones, &now) {
        return rows.len();
    }
    for (&id, group) in leaves.iter().zip(groups) {
        let Some((parent, slot)) = slot_in_parent(tree, id) else { continue };
        let mine: Vec<&[f32]> = group.iter().map(|&i| points[i]).collect();
        let c = weighted_mean(&mine, &vec![1.0; mine.len()], &(0..mine.len()).collect::<Vec<_>>());
        let s = &mut tree.store;
        s.entry_mut(parent, slot).copy_from_slice(&c);
        s.set_count(id, mine.len());
        s.set_radius(id, mine.iter().map(|p| dist(p, &c)).fold(0.0, f32::max));
        s.set_entries(id, &group.iter().map(|&i| rows[i]).collect::<Vec<_>>(), &mine.concat());
    }
    if deep.is_some() {
        for (s, k) in children_of_kind(tree, page, false) {
            refresh(tree, k, (page, s));
        }
    }
    rows.len()
}

/// Pool the children of the internal pages directly below `parent` and deal them
/// out again among those pages, grouped by their centroids; returns the number of
/// page references regrouped.
fn repartition_internal(tree: &mut Tree, parent: u32) -> usize {
    let d = tree.store.num_dim();
    let members = children_of_kind(tree, parent, false);
    let kids: Vec<u32> = members.iter().flat_map(|m| tree.store.ids(m.1).to_vec()).collect();
    let flat: Vec<f32> = members.iter().flat_map(|m| tree.store.block(m.1).to_vec()).collect();
    if members.len() < 2 {
        return 0;
    }
    let cents: Vec<&[f32]> = flat.chunks_exact(d).collect();
    let weights: Vec<f64> = kids.iter().map(|&k| tree.store.count(k) as f64).collect();
    let groups = groups_of(&cents, members.len());
    let now = runs(members.iter().map(|m| tree.store.len(m.1)));
    if scatter(&cents, &weights, &groups) >= scatter(&cents, &weights, &now) {
        return kids.len();
    }
    for (&(slot, id), group) in members.iter().zip(groups) {
        let s = &mut tree.store;
        let c = weighted_mean(&cents, &weights, &group);
        let r = group.iter().map(|&i| dist(cents[i], &c) + s.radius(kids[i])).fold(0.0, f32::max);
        let group_cents: Vec<f32> = group.iter().flat_map(|&i| cents[i].iter().copied()).collect();
        let group_kids: Vec<u32> = group.iter().map(|&i| kids[i]).collect();
        let n: usize = group_kids.iter().map(|&k| s.count(k)).sum();
        s.entry_mut(parent, slot).copy_from_slice(&c);
        s.set_count(id, n);
        s.set_radius(id, r);
        s.set_entries(id, &group_kids, &group_cents);
        group_kids.iter().for_each(|&k| s.set_parent(k, Some(id)));
    }
    kids.len()
}

#[cfg(test)]
#[path = "tree_morph_tests.rs"]
mod tests;
