//! Weighted 2-means split used when a tree page overflows.

use std::cmp::Ordering;

use crate::distance::l2_sq_f32;

const TWO_MEANS_ITERS: usize = 6;

/// Weighted mean of `points[i]` over `members`.
pub fn weighted_mean<P: AsRef<[f32]>>(points: &[P], weights: &[f64], members: &[usize]) -> Vec<f32> {
    let dim = points.first().map_or(0, |p| p.as_ref().len());
    let mut acc = vec![0.0f64; dim];
    let mut total = 0.0f64;
    for &i in members {
        let w = weights[i];
        total += w;
        for (a, &v) in acc.iter_mut().zip(points[i].as_ref()) {
            *a += w * f64::from(v);
        }
    }
    let total = total.max(f64::MIN_POSITIVE);
    acc.iter().map(|&a| (a / total) as f32).collect()
}

fn farthest_from<P: AsRef<[f32]>>(points: &[P], anchor: &[f32]) -> usize {
    let mut best = (0usize, -1.0f32);
    for (i, p) in points.iter().enumerate() {
        let d = l2_sq_f32(p.as_ref(), anchor);
        if d > best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// Split `points` (n >= 2) into two halves (sizes `n/2` and `n - n/2`) at the median
/// of their projections onto the line through two far-apart points: `a`, the point
/// farthest from `points[0]`, and `b`, the point farthest from `a`. Ties (and
/// duplicate points) are broken by position, so the halves are always balanced.
pub fn median_split<P: AsRef<[f32]>>(points: &[P]) -> (Vec<usize>, Vec<usize>) {
    let n = points.len();
    assert!(n >= 2, "median_split needs at least two points");
    let a = points[farthest_from(points, points[0].as_ref())].as_ref();
    let b = points[farthest_from(points, a)].as_ref();
    let dir: Vec<f32> = b.iter().zip(a).map(|(&x, &y)| x - y).collect();
    let mut proj: Vec<(f32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| (p.as_ref().iter().zip(&dir).map(|(&x, &d)| x * d).sum(), i))
        .collect();
    let half = n / 2;
    proj.select_nth_unstable_by(half, |x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
    let left = proj[..half].iter().map(|&(_, i)| i).collect();
    let right = proj[half..].iter().map(|&(_, i)| i).collect();
    (left, right)
}

/// Split `points` (n >= 2) into two non-empty groups of index positions by weighted
/// 2-means. Each group keeps at least a quarter of the points, so repeated splits keep
/// the tree balanced even for skewed or duplicate points.
pub fn two_means_split<P: AsRef<[f32]>>(points: &[P], weights: &[f64]) -> (Vec<usize>, Vec<usize>) {
    let n = points.len();
    assert!(n >= 2, "two_means_split needs at least two points");
    let all: Vec<usize> = (0..n).collect();
    let mean = weighted_mean(points, weights, &all);
    let a0 = farthest_from(points, &mean);
    let b0 = farthest_from(points, points[a0].as_ref());
    let mut ca = points[a0].as_ref().to_vec();
    let mut cb = points[b0].as_ref().to_vec();
    let mut margin: Vec<(f32, usize)> = vec![(0.0, 0); n];
    let mut to_b = vec![false; n];
    let (mut a, mut b) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for iter in 0..=TWO_MEANS_ITERS {
        let mut changed = false;
        for (i, p) in points.iter().enumerate() {
            let m = l2_sq_f32(p.as_ref(), &ca) - l2_sq_f32(p.as_ref(), &cb);
            margin[i] = (m, i);
            changed |= (m > 0.0) != to_b[i] || iter == 0;
            to_b[i] = m > 0.0;
        }
        if !changed || iter == TWO_MEANS_ITERS {
            break;
        }
        a.clear();
        b.clear();
        (0..n).for_each(|i| if to_b[i] { b.push(i) } else { a.push(i) });
        if a.is_empty() || b.is_empty() {
            break;
        }
        ca = weighted_mean(points, weights, &a);
        cb = weighted_mean(points, weights, &b);
    }
    margin.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
    let count_a = margin.iter().filter(|(m, _)| *m <= 0.0).count();
    let min_side = (n / 4).max(1);
    let cut = count_a.clamp(min_side, n - min_side);
    let left = margin[..cut].iter().map(|&(_, i)| i).collect();
    let right = margin[cut..].iter().map(|&(_, i)| i).collect();
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_two_clusters_apart() {
        let mut pts = Vec::new();
        for i in 0..10 {
            pts.push(vec![i as f32 * 0.01, 0.0]);
            pts.push(vec![100.0 + i as f32 * 0.01, 0.0]);
        }
        let w = vec![1.0; pts.len()];
        let (a, b) = two_means_split(&pts, &w);
        assert_eq!(a.len(), 10);
        assert_eq!(b.len(), 10);
        let side = |g: &[usize]| g.iter().all(|&i| pts[i][0] < 50.0) || g.iter().all(|&i| pts[i][0] > 50.0);
        assert!(side(&a) && side(&b));
    }

    #[test]
    fn duplicates_and_outliers_still_split_balanced() {
        let mut pts = vec![vec![1.0f32, 1.0]; 12];
        let w = vec![1.0; 12];
        let (a, b) = two_means_split(&pts, &w);
        assert_eq!(a.len() + b.len(), 12);
        assert!(a.len() >= 3 && b.len() >= 3);
        pts.push(vec![1e6, 1e6]);
        let w = vec![1.0; 13];
        let (a, b) = two_means_split(&pts, &w);
        assert!(a.len() >= 3 && b.len() >= 3);
    }

    #[test]
    fn median_split_separates_clusters_and_balances_duplicates() {
        let mut pts = Vec::new();
        for i in 0..10 {
            pts.push(vec![i as f32 * 0.01, 0.0]);
            pts.push(vec![100.0 + i as f32 * 0.01, 0.0]);
        }
        let (a, b) = median_split(&pts);
        assert_eq!((a.len(), b.len()), (10, 10));
        let side = |g: &[usize]| g.iter().all(|&i| pts[i][0] < 50.0) || g.iter().all(|&i| pts[i][0] > 50.0);
        assert!(side(&a) && side(&b));
        let (a, b) = median_split(&vec![vec![1.0f32, 1.0]; 13]);
        assert_eq!((a.len(), b.len()), (6, 7));
    }

    #[test]
    fn weighted_mean_respects_weights() {
        let pts = vec![vec![0.0f32], vec![10.0]];
        let m = weighted_mean(&pts, &[3.0, 1.0], &[0, 1]);
        assert!((m[0] - 2.5).abs() < 1e-6);
    }
}
