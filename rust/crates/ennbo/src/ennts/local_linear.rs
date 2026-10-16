pub(super) fn at_query(
    query: &[f64],
    anchors: &[Vec<f64>],
    targets: &[f64],
    weights: &[f64],
) -> Option<f64> {
    let dim = query.len();
    let k = anchors.len();
    if dim == 0 || k == 0 || targets.len() != k || weights.len() != k {
        return None;
    }
    if anchors.iter().any(|row| row.len() != dim) {
        return None;
    }
    let weight_sum: f64 = weights
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return None;
    }
    let mut rms = vec![0.0; dim];
    for (anchor, weight) in anchors.iter().zip(weights) {
        if !(weight.is_finite() && *weight > 0.0) {
            continue;
        }
        let share = weight / weight_sum;
        for d in 0..dim {
            let delta = anchor[d] - query[d];
            rms[d] += share * delta * delta;
        }
    }
    let active: Vec<usize> = (0..dim).filter(|&d| rms[d].sqrt() > 1e-8).collect();
    if active.is_empty() || k < active.len().saturating_mul(4) {
        return None;
    }
    let p = active.len() + 1;
    let mut gram = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    let mut feature = vec![0.0; p];
    for ((anchor, &y), &weight) in anchors.iter().zip(targets).zip(weights) {
        if !(weight.is_finite() && weight > 0.0 && y.is_finite()) {
            continue;
        }
        let share = weight / weight_sum;
        feature[0] = 1.0;
        for (slot, &d) in active.iter().enumerate() {
            feature[slot + 1] = (anchor[d] - query[d]) / rms[d].sqrt();
        }
        for a in 0..p {
            rhs[a] += share * y * feature[a];
            for c in 0..p {
                gram[a * p + c] += share * feature[a] * feature[c];
            }
        }
    }
    let slope_energy: f64 = (1..p).map(|d| gram[d * p + d]).sum();
    if !(slope_energy.is_finite() && slope_energy > 0.0) {
        return None;
    }
    let penalty = 0.005 * slope_energy / active.len() as f64;
    for d in 1..p {
        gram[d * p + d] += penalty;
    }
    solve(&mut gram, &mut rhs, p)
}

fn solve(gram: &mut [f64], rhs: &mut [f64], p: usize) -> Option<f64> {
    for col in 0..p {
        let mut pivot = col;
        let mut best = gram[col * p + col].abs();
        for row in (col + 1)..p {
            let value = gram[row * p + col].abs();
            if value > best {
                best = value;
                pivot = row;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..p {
                gram.swap(col * p + j, pivot * p + j);
            }
            rhs.swap(col, pivot);
        }
        let diag = gram[col * p + col];
        for row in (col + 1)..p {
            let factor = gram[row * p + col] / diag;
            for j in col..p {
                gram[row * p + j] -= factor * gram[col * p + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..p).rev() {
        let mut acc = rhs[row];
        for j in (row + 1)..p {
            acc -= gram[row * p + j] * rhs[j];
        }
        rhs[row] = acc / gram[row * p + row];
    }
    let mu = rhs[0];
    if mu.is_finite() {
        Some(mu)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::at_query;

    #[test]
    fn exact_line_is_recovered_at_the_query() {
        let query = vec![30.0];
        let anchors: Vec<Vec<f64>> = (22..30).map(|x| vec![x as f64]).collect();
        let targets: Vec<f64> = (22..30).map(|x| 2.0 * (x as f64) + 2.0).collect();
        let weights = vec![1.0; 8];
        let mu = at_query(&query, &anchors, &targets, &weights).unwrap();
        assert!((mu - 62.0).abs() < 0.5, "mu={mu}");
    }

    #[test]
    fn identical_anchors_decline_the_fit() {
        let query = vec![1.0, 2.0];
        let anchors = vec![vec![1.0, 2.0], vec![1.0, 2.0]];
        let targets = vec![4.0, 6.0];
        let weights = vec![1.0, 1.0];
        assert!(at_query(&query, &anchors, &targets, &weights).is_none());
    }
}
