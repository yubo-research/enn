use std::sync::Arc;

use super::forest_node::{Cut, Dir, Node};
use super::forest_rules::{SplitConfig, SplitRule};
use super::lsh::mix64;

const SPLIT_ATTEMPTS: u32 = 4;
const SIGN_BITS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placed {
    Present,
    Appended,
    Split(usize),
}

pub(crate) struct Parted {
    pub node: Node,
    pub left: (u32, Vec<u32>),
    pub right: (u32, Vec<u32>),
}

pub(crate) struct Projector<'a> {
    pub tree_seed: u64,
    pub inv_scale: &'a [f64],
}

impl Projector<'_> {
    pub(crate) fn project(&self, x: &[f64], node_seed: u32) -> f64 {
        let base = mix64(self.tree_seed ^ (u64::from(node_seed) << 24));
        let mut acc = 0.0;
        for (c, (xs, ws)) in x
            .chunks(SIGN_BITS)
            .zip(self.inv_scale.chunks(SIGN_BITS))
            .enumerate()
        {
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
        if den > 0.0 {
            num / den.sqrt()
        } else {
            0.0
        }
    }

    pub(crate) fn seed_dual_norm(&self, weights: &[f64]) -> f64 {
        let n = self.inv_scale.len().max(1) as f64;
        (self
            .inv_scale
            .iter()
            .zip(weights)
            .map(|(i, w)| i * i / w)
            .sum::<f64>()
            / n)
            .sqrt()
    }

    pub(crate) fn pivot_dual_norm(&self, a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
        self.pivot_direction(a, b)
            .iter()
            .zip(weights)
            .map(|(d, w)| d * d / w)
            .sum::<f64>()
            .sqrt()
    }

    pub(crate) fn pivot_direction(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = a
            .iter()
            .zip(b)
            .zip(self.inv_scale)
            .map(|((av, bv), w)| w * w * (av - bv))
            .collect();
        let den: f64 = a
            .iter()
            .zip(b)
            .zip(self.inv_scale)
            .map(|((av, bv), w)| (w * (av - bv)).powi(2))
            .sum::<f64>()
            .sqrt();
        d.into_iter()
            .map(|v| if den > 0.0 { v / den } else { 0.0 })
            .collect()
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
    (thr > sorted[pick - 1])
        .then_some(thr)
        .or(Some(sorted[pick]))
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

fn ranked_cuts(n: usize, node_id: u32, salt: u64) -> Vec<LocalCut> {
    let pair = (n >= 2)
        .then(|| pivot_pair(n, node_id, 0, salt))
        .map(|(i, j)| LocalCut::Pair(i, j));
    let fallback = (0..SPLIT_ATTEMPTS).map(|a| LocalCut::Seed(split_seed(node_id, a, salt)));
    pair.into_iter().chain(fallback).collect()
}

pub(crate) fn choose_split(
    projector: &Projector<'_>,
    xs: &[&[f64]],
    ids: &[u32],
    node_id: u32,
    salt: u64,
    split: SplitConfig,
) -> Option<(Cut, f64, Vec<bool>)> {
    for local in ranked_cuts(xs.len(), node_id, salt) {
        let (cut, proj): (Cut, Vec<f64>) = match (local, split.rule) {
            (LocalCut::Seed(seed), _) => (
                Cut::Seed(seed),
                xs.iter().map(|x| projector.project(x, seed)).collect(),
            ),
            (LocalCut::Pair(i, j), SplitRule::Pivot) => (
                Cut::Pivot(ids[i], ids[j]),
                xs.iter()
                    .map(|x| projector.project_pivot(x, xs[i], xs[j]))
                    .collect(),
            ),
            (LocalCut::Pair(i, j), SplitRule::PivotI8) => {
                let dir = Dir::quantize(&projector.pivot_direction(xs[i], xs[j]));
                let proj = xs.iter().map(|x| dir.dot(x)).collect();
                (Cut::Dir(Arc::new(dir)), proj)
            }
        };
        if let Some(thr) = median_threshold(&proj) {
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
    }

    fn pivot() -> SplitConfig {
        SplitConfig {
            rule: SplitRule::Pivot,
            par_above: 0,
            ..SplitConfig::default()
        }
    }

    #[test]
    fn duplicates_cannot_be_split_but_distinct_rows_can() {
        let scale = [1.0; 3];
        let p = Projector {
            tree_seed: 9,
            inv_scale: &scale,
        };
        let same = [0.5, 0.5, 0.5];
        let dup: Vec<&[f64]> = (0..40).map(|_| &same[..]).collect();
        let ids: Vec<u32> = (100..140).collect();
        assert!(choose_split(&p, &dup, &ids, 3, 1, pivot()).is_none());
        let rows: Vec<[f64; 3]> = (0..40).map(|i| [i as f64, 0.0, 1.0]).collect();
        let refs: Vec<&[f64]> = rows.iter().map(|r| &r[..]).collect();
        let (cut, _, left) = choose_split(&p, &refs, &ids, 3, 1, pivot()).unwrap();
        assert_eq!(left.iter().filter(|&&l| l).count(), 20);
        let (i, j) = pivot_pair(40, 3, 0, 1);
        assert_eq!(cut, Cut::Pivot(ids[i], ids[j]));
        let exact = p.pivot_direction(refs[i], refs[j]);
        {
            let tol = 0.01;
            let (cut, thr, coded_left) = choose_split(
                &p,
                &refs,
                &ids,
                3,
                1,
                SplitConfig {
                    rule: SplitRule::PivotI8,
                    par_above: 0,
                    ..SplitConfig::default()
                },
            )
            .unwrap();
            let Cut::Dir(dir) = cut else {
                panic!("coded rule must store a direction")
            };
            assert_eq!(coded_left, left);
            assert!(refs.iter().all(|x| (dir.dot(x)
                - x.iter().zip(&exact).map(|(a, b)| a * b).sum::<f64>())
            .abs()
                <= tol * (1.0 + x[0])));
            assert!(refs
                .iter()
                .zip(&coded_left)
                .all(|(x, &l)| (dir.dot(x) < thr) == l));
        }
    }

    #[test]
    fn pivot_projection_is_the_metric_distance_along_the_pair() {
        let scale = [2.0, 0.5, 1.0];
        let p = Projector {
            tree_seed: 1,
            inv_scale: &scale,
        };
        let (a, b, x) = ([1.0, 0.0, 3.0], [0.0, 2.0, 3.0], [0.3, -0.7, 9.0]);
        let z = |v: &[f64]| -> Vec<f64> { v.iter().zip(&scale).map(|(v, w)| v * w).collect() };
        let (za, zb, zx) = (z(&a), z(&b), z(&x));
        let diff: Vec<f64> = za.iter().zip(&zb).map(|(u, v)| u - v).collect();
        let norm = diff.iter().map(|d| d * d).sum::<f64>().sqrt();
        let want = diff.iter().zip(&zx).map(|(d, v)| d * v).sum::<f64>() / norm;
        assert!((p.project_pivot(&x, &a, &b) - want).abs() < 1e-12);
        let dot: f64 = x
            .iter()
            .zip(p.pivot_direction(&a, &b))
            .map(|(u, v)| u * v)
            .sum();
        assert!((dot - want).abs() < 1e-12);
        assert_eq!(p.project_pivot(&x, &a, &a), 0.0);
        let (i, j) = pivot_pair(2, 0, 0, 0);
        assert!(i != j && i < 2 && j < 2);
    }

    #[test]
    fn projection_is_deterministic_and_scaled() {
        let p = Projector {
            tree_seed: 1,
            inv_scale: &[1.0, 1.0],
        };
        let q = Projector {
            tree_seed: 1,
            inv_scale: &[2.0, 2.0],
        };
        let x = [0.3, -0.7];
        assert_eq!(p.project(&x, 5), p.project(&x, 5));
        assert!((q.project(&x, 5) - 2.0 * p.project(&x, 5)).abs() < 1e-12);
    }
}
