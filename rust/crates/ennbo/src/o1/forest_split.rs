use std::sync::Arc;

use super::forest_node::{Cut, Dir};
use super::forest_rules::{SplitConfig, SplitRule};
use super::lsh::mix64;

const SPLIT_ATTEMPTS: u32 = 4;
const LEAN_MIN_ROWS: usize = 64;
const SPREAD_SAMPLE: usize = 16;
const SIGN_BITS: usize = 64;

pub(crate) struct Projector<'a> {
    pub tree_seed: u64,
    pub inv_scale: &'a [f64],
}

pub(crate) fn lean_threshold(proj: &[f64], ids: &[u32], lean: u32) -> Option<f64> {
    let n = proj.len();
    if lean == 0 || n < LEAN_MIN_ROWS {
        return None;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| proj[a].total_cmp(&proj[b]));
    let mut rank = vec![0; n];
    order.iter().enumerate().for_each(|(r, &i)| rank[i] = r);
    let mut newest: Vec<usize> = (0..n).collect();
    newest.sort_by_key(|&i| std::cmp::Reverse(ids[i]));
    let ranks = newest[..n.div_ceil(8)].iter().map(|&i| rank[i]);
    let (lo, hi) = ranks.fold((n, 0), |(lo, hi), r| (lo.min(r), hi.max(r)));
    let k = n * (50 - lean as usize) / 100;
    let cut = if hi < k { k } else if lo >= n - k { n - k } else { return None };
    let (a, b) = (proj[order[cut - 1]], proj[order[cut]]);
    let thr = 0.5 * (a + b);
    (a < thr && thr <= b).then_some(thr)
}

impl Projector<'_> {
    pub(crate) fn project(&self, x: &[f64], node_seed: u32) -> f64 {
        let base = mix64(self.tree_seed ^ (u64::from(node_seed) << 24));
        let mut acc = 0.0;
        for (c, (xs, ws)) in x.chunks(SIGN_BITS).zip(self.inv_scale.chunks(SIGN_BITS)).enumerate() {
            let signs = mix64(base ^ c as u64);
            for (j, (v, w)) in xs.iter().zip(ws).enumerate() {
                let term = v * w;
                acc += if (signs >> j) & 1 == 1 { term } else { -term };
            }
        }
        acc / (x.len().max(1) as f64).sqrt()
    }
}

impl Projector<'_> {
    pub(crate) fn project_pivot(&self, x: &[f64], a: &[f64], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0, 0.0);
        for (((xv, av), bv), w) in x.iter().zip(a).zip(b).zip(self.inv_scale) {
            let d = w * (av - bv);
            num += d * w * xv;
            den += d * d;
        }
        if den > 0.0 { num / den.sqrt() } else { 0.0 }
    }

    pub(crate) fn seed_dual_norm(&self, weights: &[f64]) -> f64 {
        let n = self.inv_scale.len().max(1) as f64;
        (self.inv_scale.iter().zip(weights).map(|(i, w)| i * i / w).sum::<f64>() / n).sqrt()
    }

    pub(crate) fn pivot_dual_norm(&self, a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
        self.pivot_direction(a, b).iter().zip(weights).map(|(d, w)| d * d / w).sum::<f64>().sqrt()
    }

    pub(crate) fn pivot_direction(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = a.iter().zip(b).zip(self.inv_scale).map(|((av, bv), w)| w * w * (av - bv)).collect();
        let den: f64 = a.iter().zip(b).zip(self.inv_scale).map(|((av, bv), w)| (w * (av - bv)).powi(2)).sum::<f64>().sqrt();
        d.into_iter().map(|v| if den > 0.0 { v / den } else { 0.0 }).collect()
    }
}

pub(crate) fn split_seed(node_id: u32, attempt: u32, salt: u64) -> u32 {
    mix64(salt ^ (u64::from(node_id) << 8) ^ u64::from(attempt)) as u32
}

pub(crate) fn median_threshold(proj: &[f64]) -> Option<f64> {
    let mut sorted = proj.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    let distinct = |i: usize| sorted[i - 1] < sorted[i];
    let pick = (0..sorted.len())
        .flat_map(|d| [mid.checked_sub(d), mid.checked_add(d)])
        .flatten()
        .find(|&i| i >= 1 && i < sorted.len() && distinct(i))?;
    let thr = 0.5 * (sorted[pick - 1] + sorted[pick]);
    (thr > sorted[pick - 1]).then_some(thr).or(Some(sorted[pick]))
}

impl Projector<'_> {
    fn direction(&self, node_seed: u32) -> Vec<f64> {
        let base = mix64(self.tree_seed ^ (u64::from(node_seed) << 24));
        let norm = (self.inv_scale.len().max(1) as f64).sqrt();
        let mut dir = Vec::with_capacity(self.inv_scale.len());
        for (c, ws) in self.inv_scale.chunks(SIGN_BITS).enumerate() {
            let signs = mix64(base ^ c as u64);
            dir.extend(ws.iter().enumerate().map(|(j, w)| if (signs >> j) & 1 == 1 { w / norm } else { -w / norm }));
        }
        dir
    }
}

fn spread(sample: &[&[f64]], dir: &[f64]) -> f64 {
    let proj: Vec<f64> = sample.iter().map(|x| x.iter().zip(dir).map(|(a, b)| a * b).sum::<f64>()).collect();
    let mean = proj.iter().sum::<f64>() / proj.len().max(1) as f64;
    proj.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>()
}

pub(crate) fn pivot_pair(n: usize, node_id: u32, attempt: u32, salt: u64) -> (usize, usize) {
    let h = mix64(salt ^ 0x5049_564f_5400_0000 ^ (u64::from(node_id) << 8) ^ u64::from(attempt));
    let i = (h % n as u64) as usize;
    let j = (i + 1 + ((h >> 32) % (n as u64 - 1)) as usize) % n;
    (i, j)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalCut {
    Seed(u32),
    Pair(usize, usize),
}

fn candidate_cuts(rule: SplitRule, n: usize, node_id: u32, salt: u64, k: u32) -> Vec<LocalCut> {
    match rule {
        SplitRule::Seeded => (0..k).map(|a| LocalCut::Seed(split_seed(node_id, a, salt))).collect(),
        rule if rule.is_pivot() && n >= 2 => (0..k).map(|a| pivot_pair(n, node_id, a, salt)).map(|(i, j)| LocalCut::Pair(i, j)).collect(),
        _ => Vec::new(),
    }
}

fn local_direction(projector: &Projector<'_>, xs: &[&[f64]], cut: LocalCut) -> Vec<f64> {
    match cut {
        LocalCut::Seed(seed) => projector.direction(seed),
        LocalCut::Pair(i, j) => projector.pivot_direction(xs[i], xs[j]),
    }
}

fn ranked_cuts(projector: &Projector<'_>, xs: &[&[f64]], node_id: u32, salt: u64, split: SplitConfig) -> Vec<LocalCut> {
    let k = split.candidates.max(1);
    let stride = xs.len().div_ceil(SPREAD_SAMPLE).max(1);
    let sample: Vec<&[f64]> = xs.iter().step_by(stride).copied().collect();
    let mut scored: Vec<(f64, LocalCut)> = candidate_cuts(split.rule, xs.len(), node_id, salt, k)
        .into_iter()
        .map(|cut| (if k > 1 { spread(&sample, &local_direction(projector, xs, cut)) } else { 0.0 }, cut))
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let seeded = if split.rule == SplitRule::Seeded { k } else { 0 };
    let fallback = (seeded..seeded.max(SPLIT_ATTEMPTS)).map(|a| LocalCut::Seed(split_seed(node_id, a, salt)));
    scored.into_iter().map(|(_, cut)| cut).chain(fallback).collect()
}

pub(crate) fn choose_split(
    projector: &Projector<'_>,
    xs: &[&[f64]],
    ids: &[u32],
    node_id: u32,
    salt: u64,
    split: SplitConfig,
) -> Option<(Cut, f64, Vec<bool>)> {
    for local in ranked_cuts(projector, xs, node_id, salt, split) {
        let (cut, proj): (Cut, Vec<f64>) = match (local, split.rule.coded()) {
            (LocalCut::Seed(seed), _) => (Cut::Seed(seed), xs.iter().map(|x| projector.project(x, seed)).collect()),
            (LocalCut::Pair(i, j), None) => (Cut::Pivot(ids[i], ids[j]), xs.iter().map(|x| projector.project_pivot(x, xs[i], xs[j])).collect()),
            (LocalCut::Pair(i, j), Some(wide)) => {
                let dir = Dir::quantize(&projector.pivot_direction(xs[i], xs[j]), wide);
                let proj = xs.iter().map(|x| dir.dot(x)).collect();
                (Cut::Dir(Arc::new(dir)), proj)
            }
        };
        if let Some(thr) = lean_threshold(&proj, ids, split.lean).or_else(|| median_threshold(&proj)) {
            let left: Vec<bool> = proj.iter().map(|&p| p < thr).collect();
            if left.iter().any(|&l| l) && left.iter().any(|&l| !l) {
                return Some((cut, thr, left));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_threshold_splits_near_the_middle_and_skips_ties() {
        let thr = median_threshold(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert!(thr > 2.0 && thr < 3.0);
        let thr = median_threshold(&[1.0, 1.0, 1.0, 2.0]).unwrap();
        assert!(thr > 1.0 && thr <= 2.0);
        assert!(median_threshold(&[5.0, 5.0, 5.0]).is_none());
        assert!(median_threshold(&[]).is_none());
        let proj: Vec<f64> = (0..100).map(f64::from).collect();
        let newest_low: Vec<u32> = (0..100).map(|i| 100 - i).collect();
        let newest_high: Vec<u32> = (0..100).collect();
        let spread: Vec<u32> = (0..100).map(|i| (i * 37) % 100).collect();
        assert_eq!(lean_threshold(&proj, &newest_low, 15), Some(34.5));
        assert_eq!(lean_threshold(&proj, &newest_high, 15), Some(64.5));
        assert_eq!(lean_threshold(&proj, &spread, 15), None);
        assert_eq!(lean_threshold(&proj, &newest_high, 0), None);
        assert_eq!(lean_threshold(&proj[..40], &newest_high[..40], 15), None);
        assert_eq!(lean_threshold(&[1.0; 100], &newest_high, 15), None);
    }

    fn seeded(k: u32) -> SplitConfig {
        SplitConfig { rule: SplitRule::Seeded, candidates: k, stage_above: 0, stage_work: 0, lean: 0, par_above: 0, local_kib: 0, pread: 0 }
    }

    fn pivot(k: u32) -> SplitConfig {
        SplitConfig { rule: SplitRule::Pivot, candidates: k, stage_above: 0, stage_work: 0, lean: 0, par_above: 0, local_kib: 0, pread: 0 }
    }

    #[test]
    fn duplicates_cannot_be_split_but_distinct_rows_can() {
        let scale = [1.0; 3];
        let p = Projector { tree_seed: 9, inv_scale: &scale };
        let same = [0.5, 0.5, 0.5];
        let dup: Vec<&[f64]> = (0..40).map(|_| &same[..]).collect();
        let ids: Vec<u32> = (100..140).collect();
        for split in [seeded(1), seeded(16), pivot(1), pivot(16)] {
            assert!(choose_split(&p, &dup, &ids, 3, 1, split).is_none());
        }
        let rows: Vec<[f64; 3]> = (0..40).map(|i| [i as f64, 0.0, 1.0]).collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| &r[..]).collect();
        let (_, _, left) = choose_split(&p, &refs, &ids, 3, 1, seeded(1)).unwrap();
        assert_eq!(left.iter().filter(|&&l| l).count(), 20);
        assert_eq!(choose_split(&p, &refs, &ids, 3, 1, seeded(1)).unwrap().0, Cut::Seed(split_seed(3, 0, 1)));
        let (cut, _, left) = choose_split(&p, &refs, &ids, 3, 1, pivot(1)).unwrap();
        assert_eq!(left.iter().filter(|&&l| l).count(), 20);
        let (i, j) = pivot_pair(40, 3, 0, 1);
        assert_eq!(cut, Cut::Pivot(ids[i], ids[j]));
        let exact = p.pivot_direction(refs[i], refs[j]);
        for (rule, tol) in [(SplitRule::PivotI8, 0.01), (SplitRule::PivotF32, 1e-6)] {
            let (cut, thr, coded_left) = choose_split(&p, &refs, &ids, 3, 1, SplitConfig { rule, candidates: 1, stage_above: 0, stage_work: 0, lean: 0, par_above: 0, local_kib: 0, pread: 0 }).unwrap();
            let Cut::Dir(dir) = cut else { panic!("coded rule must store a direction") };
            assert_eq!(coded_left, left);
            assert!(refs.iter().all(|x| (dir.dot(x) - x.iter().zip(&exact).map(|(a, b)| a * b).sum::<f64>()).abs() <= tol * (1.0 + x[0])));
            assert!(refs.iter().zip(&coded_left).all(|(x, &l)| (dir.dot(x) < thr) == l));
        }
    }

    #[test]
    fn more_candidates_pick_the_direction_with_the_widest_spread() {
        let scale = [1.0; 8];
        let p = Projector { tree_seed: 5, inv_scale: &scale };
        let rows: Vec<[f64; 8]> = (0..33).map(|i| { let mut r = [0.0; 8]; r[0] = i as f64; r[1] = 0.01 * (i % 3) as f64; r }).collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| &r[..]).collect();
        let ids: Vec<u32> = (0..33).collect();
        let dir_of = |cut: Cut| match cut {
            Cut::Seed(seed) => p.direction(seed),
            Cut::Pivot(a, b) => p.pivot_direction(refs[a as usize], refs[b as usize]),
            Cut::Dir(_) => Vec::new(),
        };
        let spread_of = |split| spread(&refs, &dir_of(choose_split(&p, &refs, &ids, 9, 2, split).unwrap().0));
        let best = (1..=16).map(|a| spread(&refs, &p.direction(split_seed(9, a - 1, 2)))).fold(0.0, f64::max);
        assert_eq!(spread_of(seeded(16)), best);
        assert!(spread_of(seeded(16)) >= spread_of(seeded(1)));
        assert!(spread_of(pivot(16)) >= spread_of(pivot(1)));
        assert!(spread_of(pivot(16)) > spread_of(seeded(16)));
    }

    #[test]
    fn pivot_projection_is_the_metric_distance_along_the_pair() {
        let scale = [2.0, 0.5, 1.0];
        let p = Projector { tree_seed: 1, inv_scale: &scale };
        let (a, b, x) = ([1.0, 0.0, 3.0], [0.0, 2.0, 3.0], [0.3, -0.7, 9.0]);
        let z = |v: &[f64]| -> Vec<f64> { v.iter().zip(&scale).map(|(v, w)| v * w).collect() };
        let (za, zb, zx) = (z(&a), z(&b), z(&x));
        let diff: Vec<f64> = za.iter().zip(&zb).map(|(u, v)| u - v).collect();
        let norm = diff.iter().map(|d| d * d).sum::<f64>().sqrt();
        let want = diff.iter().zip(&zx).map(|(d, v)| d * v).sum::<f64>() / norm;
        assert!((p.project_pivot(&x, &a, &b) - want).abs() < 1e-12);
        let dot: f64 = x.iter().zip(p.pivot_direction(&a, &b)).map(|(u, v)| u * v).sum();
        assert!((dot - want).abs() < 1e-12);
        assert_eq!(p.project_pivot(&x, &a, &a), 0.0);
        let (i, j) = pivot_pair(2, 0, 0, 0);
        assert!(i != j && i < 2 && j < 2);
    }

    #[test]
    fn direction_matches_projection() {
        let scale: Vec<f64> = (0..130).map(|j| 1.0 + j as f64 / 7.0).collect();
        let x: Vec<f64> = (0..130).map(|j| (j as f64 * 0.37).sin()).collect();
        let p = Projector { tree_seed: 3, inv_scale: &scale };
        let dot: f64 = x.iter().zip(p.direction(11)).map(|(a, b)| a * b).sum();
        assert!((dot - p.project(&x, 11)).abs() < 1e-9);
    }

    #[test]
    fn projection_is_deterministic_and_scaled() {
        let p = Projector { tree_seed: 1, inv_scale: &[1.0, 1.0] };
        let q = Projector { tree_seed: 1, inv_scale: &[2.0, 2.0] };
        let x = [0.3, -0.7];
        assert_eq!(p.project(&x, 5), p.project(&x, 5));
        assert!((q.project(&x, 5) - 2.0 * p.project(&x, 5)).abs() < 1e-12);
    }
}
