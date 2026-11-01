use crate::error::ENNError;
use crate::metric_sobol::{
    group_cell_count, group_sobol_index, null_sd, sobol_index, DEPENDENCE_Z, MIN_DEPENDENCE_ROWS,
};

const MIN_VAR: f64 = 1e-24;
pub const DEPENDENCE_FLOOR: f64 = 1e-6;

pub fn validate_tied_dims(tied: &[Vec<usize>], num_dim: usize) -> Result<(), ENNError> {
    let mut flat = Vec::new();
    for group in tied {
        if group.is_empty() {
            return Err(ENNError::InvalidParameter(
                "tied_dims groups must be non-empty".into(),
            ));
        }
        for &j in group {
            if j >= num_dim {
                return Err(ENNError::InvalidParameter(format!(
                    "tied_dims entries must be in [0, {num_dim}), got {j}"
                )));
            }
            flat.push(j);
        }
    }
    let mut uniq = flat.clone();
    uniq.sort_unstable();
    uniq.dedup();
    if uniq.len() != flat.len() {
        return Err(ENNError::InvalidParameter(
            "tied_dims groups must be disjoint".into(),
        ));
    }
    Ok(())
}

pub fn validate_tied_dims_signed(tied: &[Vec<i64>], num_dim: usize) -> Result<(), ENNError> {
    let mut converted = Vec::with_capacity(tied.len());
    for group in tied {
        let mut row = Vec::with_capacity(group.len());
        for &j in group {
            if j < 0 {
                return Err(ENNError::InvalidParameter(format!(
                    "tied_dims entries must be in [0, {num_dim}), got {j}"
                )));
            }
            row.push(j as usize);
        }
        converted.push(row);
    }
    validate_tied_dims(&converted, num_dim)
}

pub fn smallest_tied_dim_count(tied: &[Vec<i64>]) -> usize {
    let largest = tied.iter().flatten().copied().max().unwrap_or(0);
    usize::try_from(largest).map_or(1, |j| j.saturating_add(1))
}

fn column(y: &[f64], n: usize, m: usize, j: usize) -> Vec<f64> {
    (0..n).map(|i| y[i * m + j]).collect()
}

fn mean_sobol(x: &[f64], n: usize, d: usize, y: &[f64], m: usize) -> Vec<f64> {
    let mut acc = vec![0.0; d];
    for j in 0..m {
        let col = column(y, n, m, j);
        let s = sobol_index(x, n, d, &col, None);
        for dim in 0..d {
            acc[dim] += s[dim];
        }
    }
    for v in &mut acc {
        *v /= m as f64;
    }
    acc
}

fn passing(s: f64, n: usize, cells: Option<usize>) -> f64 {
    if s > DEPENDENCE_Z * null_sd(n, cells) {
        s
    } else {
        0.0
    }
}

fn unit_indices(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    m: usize,
    tied: &[Vec<usize>],
) -> (Vec<f64>, Vec<usize>) {
    let raw = mean_sobol(x, n, d, y, m);
    let s: Vec<f64> = raw.iter().copied().map(|v| passing(v, n, None)).collect();
    if tied.is_empty() {
        return (s, (0..d).collect());
    }
    let mut in_group = vec![false; d];
    for g in tied {
        for &j in g {
            in_group[j] = true;
        }
    }
    let free: Vec<usize> = (0..d).filter(|&j| !in_group[j]).collect();
    let mut unit = vec![0usize; d];
    for (k, &j) in free.iter().enumerate() {
        unit[j] = k;
    }
    let mut joint = Vec::new();
    for (i, g) in tied.iter().enumerate() {
        for &j in g {
            unit[j] = free.len() + i;
        }
        let mut s_g = 0.0;
        for col_j in 0..m {
            let col = column(y, n, m, col_j);
            s_g += group_sobol_index(x, n, d, g, &col);
        }
        s_g /= m as f64;
        let cells = group_cell_count(x, n, d, g);
        joint.push(passing(s_g, n, Some(cells)));
    }
    let mut out = Vec::with_capacity(free.len() + joint.len());
    for &j in &free {
        out.push(s[j]);
    }
    out.extend(joint);
    (out, unit)
}

fn spread(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![1.0; d];
    if n == 0 {
        return out;
    }
    for j in 0..d {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for i in 0..n {
            let v = x[i * d + j];
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        out[j] = if var > MIN_VAR { var } else { 1.0 };
    }
    out
}

pub fn dependence_weights(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    m: usize,
    tied: &[Vec<usize>],
    floor: f64,
) -> Result<Vec<f64>, ENNError> {
    validate_tied_dims(tied, d)?;
    let (mut s, unit) = unit_indices(x, n, d, y, m, tied);
    let s_max = s.iter().copied().fold(0.0_f64, f64::max);
    if s_max > 0.0 {
        for v in &mut s {
            *v = v.max(floor * s_max);
        }
    } else {
        s.fill(1.0);
    }
    let mut spr = spread(x, n, d);
    for g in tied {
        for &j in g {
            spr[j] = 1.0;
        }
    }
    let s_sum = s.iter().sum::<f64>();
    let u = s.len() as f64;
    Ok((0..d).map(|j| (u * s[unit[j]] / s_sum) / spr[j]).collect())
}

pub fn trust_region_sides(weights: &[f64]) -> Vec<f64> {
    let inv: Vec<f64> = weights.iter().map(|w| 1.0 / w).collect();
    let mean = inv.iter().sum::<f64>() / inv.len() as f64;
    let s: Vec<f64> = inv.iter().map(|v| v / mean).collect();
    let geo = (s.iter().map(|v| v.ln()).sum::<f64>() / s.len() as f64).exp();
    s.iter().map(|v| v / geo).collect()
}

pub fn too_few_rows(n: usize) -> bool {
    n < MIN_DEPENDENCE_ROWS
}

pub fn require_weighting_for_ties(
    tied_dims: &[Vec<usize>],
    weighting: bool,
) -> Result<(), ENNError> {
    if !tied_dims.is_empty() && !weighting {
        return Err(ENNError::InvalidParameter(
            "tied_dims requires weighting=True".into(),
        ));
    }
    Ok(())
}

pub fn resolve_tied_groups(
    carried: &[Vec<usize>],
    explicit: Option<&[Vec<usize>]>,
    semantic_only: bool,
    weighting: bool,
) -> Result<Vec<Vec<usize>>, ENNError> {
    require_weighting_for_ties(explicit.unwrap_or_default(), weighting)?;
    let carried = if semantic_only || !weighting {
        &[][..]
    } else {
        carried
    };
    match explicit.filter(|g| !g.is_empty()) {
        None => Ok(carried.to_vec()),
        Some(groups) if !carried.is_empty() && groups != carried => Err(
            ENNError::InvalidParameter("tied_dims do not match the categorical blocks on x".into()),
        ),
        Some(groups) => Ok(groups.to_vec()),
    }
}

#[cfg(test)]
mod tied_feature_tests {
    use super::{resolve_tied_groups, smallest_tied_dim_count};
    #[test]
    fn smallest_tied_dim_count_covers_largest_entry() {
        assert_eq!(smallest_tied_dim_count(&[vec![2, 7], vec![0]]), 8);
        assert_eq!(smallest_tied_dim_count(&[vec![-1]]), 1);
        assert_eq!(smallest_tied_dim_count(&[vec![]]), 1);
        assert_eq!(
            smallest_tied_dim_count(&[vec![i64::MAX]]),
            i64::MAX as usize + 1
        );
    }

    #[test]
    fn projected_ties_do_not_inherit_removed_categorical_columns() {
        let carried = vec![vec![2, 3, 4]];
        let projected = vec![vec![0, 1]];
        assert_eq!(
            resolve_tied_groups(&carried, None, false, true).unwrap(),
            carried
        );
        assert!(resolve_tied_groups(&carried, Some(&projected), false, true).is_err());
        assert!(resolve_tied_groups(&carried, None, true, true)
            .unwrap()
            .is_empty());
        assert_eq!(
            resolve_tied_groups(&carried, Some(&projected), true, true).unwrap(),
            projected
        );
    }

    #[test]
    fn without_weighting_carried_groups_vanish_and_explicit_groups_are_rejected() {
        let groups = vec![vec![0, 1]];
        assert!(resolve_tied_groups(&groups, None, false, false)
            .unwrap()
            .is_empty());
        let err = resolve_tied_groups(&[], Some(&groups), false, false).unwrap_err();
        assert!(err.to_string().contains("requires weighting=True"), "{err}");
        assert!(resolve_tied_groups(&[], Some(&[]), false, false).is_ok());
    }
}
