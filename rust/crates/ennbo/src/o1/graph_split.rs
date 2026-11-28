use super::graph_io::Items;
use super::search::weighted_dist2;

pub(crate) fn nearest(q: &[f64], it: &Items, d: usize, w: &[f64]) -> usize {
    let mut best = (f64::INFINITY, 0);
    for i in 0..it.len() {
        let dist = weighted_dist2(q, it.row(i, d), w);
        if dist < best.0 {
            best = (dist, i);
        }
    }
    best.1
}

fn farthest(from: &[f64], it: &Items, d: usize, w: &[f64]) -> usize {
    let mut best = (-1.0, 0);
    for i in 0..it.len() {
        let dist = weighted_dist2(from, it.row(i, d), w);
        if dist > best.0 {
            best = (dist, i);
        }
    }
    best.1
}

fn take(it: &Items, idx: &[usize], d: usize) -> Items {
    let mut out = Items::default();
    idx.iter().for_each(|&i| out.push(it.ids[i], it.row(i, d)));
    out
}

pub(crate) fn halve(it: &Items, d: usize, w: &[f64]) -> (Items, Items) {
    let c = it.mean(d);
    let a = farthest(&c, it, d, w);
    let b = farthest(it.row(a, d), it, d, w);
    let dir: Vec<f64> = (0..d).map(|k| (it.row(b, d)[k] - it.row(a, d)[k]) * w[k]).collect();
    let mut key: Vec<(f64, usize)> = (0..it.len())
        .map(|i| (it.row(i, d).iter().zip(&dir).map(|(x, v)| x * v).sum(), i))
        .collect();
    key.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    let idx: Vec<usize> = key.into_iter().map(|p| p.1).collect();
    let h = idx.len() / 2;
    (take(it, &idx[..h], d), take(it, &idx[h..], d))
}

pub(crate) fn split_to_cap(it: Items, d: usize, w: &[f64], cap: usize) -> Vec<Items> {
    let mut todo = vec![it];
    let mut out = Vec::new();
    while let Some(p) = todo.pop() {
        if p.len() <= cap.max(1) {
            out.push(p);
        } else {
            let (a, b) = halve(&p, d, w);
            todo.push(b);
            todo.push(a);
        }
    }
    out
}

pub(crate) fn pick_links(c: &[f64], cands: &Items, own: u32, d: usize, w: &[f64], m: usize) -> Vec<u32> {
    let mut by: Vec<(f64, u32)> = (0..cands.len())
        .filter(|&i| cands.ids[i] != own)
        .map(|i| (weighted_dist2(c, cands.row(i, d), w), cands.ids[i]))
        .collect();
    by.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
    by.dedup_by_key(|p| p.1);
    by.into_iter().take(m).map(|p| p.1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(n: usize) -> Items {
        let mut it = Items::default();
        (0..n).for_each(|i| it.push(i as u32, &[i as f64, 0.0]));
        it
    }

    #[test]
    fn split_to_cap_keeps_every_item_once_and_respects_cap() {
        let pieces = split_to_cap(line(1000), 2, &[1.0, 1.0], 64);
        assert!(pieces.iter().all(|p| p.len() <= 64 && p.len() >= 32));
        let mut ids: Vec<u32> = pieces.iter().flat_map(|p| p.ids.clone()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..1000).collect::<Vec<u32>>());
    }

    #[test]
    fn halve_separates_along_the_spread_axis() {
        let (a, b) = halve(&line(10), 2, &[1.0, 1.0]);
        let (amax, bmin) = (a.ids.iter().max().copied(), b.ids.iter().min().copied());
        assert!(amax < bmin || b.ids.iter().max() < a.ids.iter().min());
        assert_eq!(nearest(&[7.2, 0.0], &line(10), 2, &[1.0, 1.0]), 7);
    }

    #[test]
    fn pick_links_excludes_self_and_sorts_by_distance() {
        let got = pick_links(&[3.0, 0.0], &line(10), 3, 2, &[1.0, 1.0], 3);
        assert_eq!(got.len(), 3);
        assert!(!got.contains(&3));
        assert!(got.contains(&2) && got.contains(&4));
    }
}
