//! Per-metric output bounds and strictly increasing warps φ(y) → z.
//!
//! Storage / fit / acquisition stay in warped `z`. Public APIs naturalize.

use ndarray::{Array2, ArrayView2, Zip};

use crate::error::ENNError;

/// Default unbounded bounds for `num_metrics` columns: each `(−∞, +∞)`.
pub fn unbounded_bounds(num_metrics: usize) -> Array2<f64> {
    let mut b = Array2::zeros((num_metrics, 2));
    for i in 0..num_metrics {
        b[[i, 0]] = f64::NEG_INFINITY;
        b[[i, 1]] = f64::INFINITY;
    }
    b
}

/// True when every column is `(−∞, +∞)`.
pub fn is_identity_bounds(bounds: &Array2<f64>) -> bool {
    bounds.nrows() == 0
        || bounds
            .rows()
            .into_iter()
            .all(|row| row[0] == f64::NEG_INFINITY && row[1] == f64::INFINITY)
}

/// Validate shape `(num_metrics, 2)` and `a < b` when both finite.
pub fn validate_bounds(bounds: &Array2<f64>, num_metrics: usize) -> Result<(), ENNError> {
    if bounds.nrows() != num_metrics || bounds.ncols() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![num_metrics, 2],
            got: bounds.shape().to_vec(),
        });
    }
    for i in 0..num_metrics {
        let a = bounds[[i, 0]];
        let b = bounds[[i, 1]];
        if a.is_nan() || b.is_nan() {
            return Err(ENNError::InvalidParameter(format!(
                "y_bounds[{i}] contains NaN"
            )));
        }
        if a.is_finite() && b.is_finite() && a >= b {
            return Err(ENNError::InvalidParameter(format!(
                "y_bounds[{i}]: lower bound {a} must be < upper bound {b}"
            )));
        }
        if !a.is_finite() && a != f64::NEG_INFINITY {
            return Err(ENNError::InvalidParameter(format!(
                "y_bounds[{i}]: open lower side must be -inf, got {a}"
            )));
        }
        if !b.is_finite() && b != f64::INFINITY {
            return Err(ENNError::InvalidParameter(format!(
                "y_bounds[{i}]: open upper side must be +inf, got {b}"
            )));
        }
    }
    Ok(())
}

/// Open-interval check: reject `y <= a` or `y >= b` on finite sides.
pub fn validate_y_in_open_interval(
    y: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Result<(), ENNError> {
    if y.ncols() != bounds.nrows() {
        return Err(ENNError::InvalidShape {
            expected: vec![y.nrows(), bounds.nrows()],
            got: y.shape().to_vec(),
        });
    }
    for (i, row) in y.rows().into_iter().enumerate() {
        for j in 0..row.len() {
            let v = row[j];
            let a = bounds[[j, 0]];
            let b = bounds[[j, 1]];
            if !v.is_finite() {
                return Err(ENNError::InvalidParameter(format!(
                    "y[{i},{j}]={v} is not finite"
                )));
            }
            if a.is_finite() && v <= a {
                return Err(ENNError::InvalidParameter(format!(
                    "y[{i},{j}]={v} is not strictly above lower bound {a}"
                )));
            }
            if b.is_finite() && v >= b {
                return Err(ENNError::InvalidParameter(format!(
                    "y[{i},{j}]={v} is not strictly below upper bound {b}"
                )));
            }
        }
    }
    Ok(())
}

#[inline]
fn warp_scalar(y: f64, a: f64, b: f64) -> Result<f64, ENNError> {
    let z = match (a.is_finite(), b.is_finite()) {
        (false, false) => y,
        (true, false) => (y - a).ln(),
        (false, true) => -(b - y).ln(),
        (true, true) => {
            let u = (y - a) / (b - a);
            (u / (1.0 - u)).ln()
        }
    };
    if z.is_finite() {
        Ok(z)
    } else {
        Err(ENNError::InvalidParameter(format!(
            "warped z={z} is not finite for y={y} bounds=({a},{b})"
        )))
    }
}

#[inline]
fn inv_scalar(z: f64, a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => z,
        (true, false) => a + z.exp(),
        (false, true) => b - (-z).exp(),
        (true, true) => {
            let s = 1.0 / (1.0 + (-z).exp());
            a + (b - a) * s
        }
    }
}

/// dφ⁻¹/dz at warped z.
#[inline]
pub fn d_inv_dz(z: f64, a: f64, b: f64) -> f64 {
    match (a.is_finite(), b.is_finite()) {
        (false, false) => 1.0,
        (true, false) => z.exp(),
        (false, true) => (-z).exp(),
        (true, true) => {
            let s = 1.0 / (1.0 + (-z).exp());
            (b - a) * s * (1.0 - s)
        }
    }
}

/// dφ/dy at natural y (for yvar Jacobian).
#[inline]
fn d_warp_dy(y: f64, a: f64, b: f64) -> Result<f64, ENNError> {
    let d = match (a.is_finite(), b.is_finite()) {
        (false, false) => 1.0,
        (true, false) => 1.0 / (y - a),
        (false, true) => 1.0 / (b - y),
        (true, true) => {
            let u = (y - a) / (b - a);
            1.0 / ((b - a) * u * (1.0 - u))
        }
    };
    if d.is_finite() {
        Ok(d)
    } else {
        Err(ENNError::InvalidParameter(format!(
            "dφ/dy={d} is not finite for y={y} bounds=({a},{b})"
        )))
    }
}

/// Warp a batch of natural y → z. Validates open interval and finite z.
pub fn warp_y(y: ArrayView2<f64>, bounds: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    validate_y_in_open_interval(y, bounds)?;
    let mut z = Array2::zeros(y.raw_dim());
    for i in 0..y.nrows() {
        for j in 0..y.ncols() {
            z[[i, j]] = warp_scalar(y[[i, j]], bounds[[j, 0]], bounds[[j, 1]])?;
        }
    }
    Ok(z)
}

/// Inverse-warp a batch of z → natural y.
pub fn inv_y(z: ArrayView2<f64>, bounds: &Array2<f64>) -> Array2<f64> {
    let mut y = Array2::zeros(z.raw_dim());
    for i in 0..z.nrows() {
        for j in 0..z.ncols() {
            y[[i, j]] = inv_scalar(z[[i, j]], bounds[[j, 0]], bounds[[j, 1]]);
        }
    }
    y
}

/// Warp observation noise: `yvar_z = (φ'(y))² · yvar_y`.
pub fn warp_yvar(
    y: ArrayView2<f64>,
    yvar: ArrayView2<f64>,
    bounds: &Array2<f64>,
) -> Result<Array2<f64>, ENNError> {
    if y.shape() != yvar.shape() {
        return Err(ENNError::InvalidShape {
            expected: y.shape().to_vec(),
            got: yvar.shape().to_vec(),
        });
    }
    validate_y_in_open_interval(y, bounds)?;
    let mut out = Array2::zeros(yvar.raw_dim());
    for i in 0..y.nrows() {
        for j in 0..y.ncols() {
            let d = d_warp_dy(y[[i, j]], bounds[[j, 0]], bounds[[j, 1]])?;
            let v = d * d * yvar[[i, j]];
            if !v.is_finite() {
                return Err(ENNError::InvalidParameter(format!(
                    "warped yvar[{i},{j}]={v} is not finite"
                )));
            }
            out[[i, j]] = v;
        }
    }
    Ok(out)
}

/// Naturalize posterior moments: μ_nat = φ⁻¹(μ_z), se_nat = |dφ⁻¹/dz|(μ_z) · se_z.
pub fn naturalize_mu_se(
    mu_z: &mut Array2<f64>,
    se_z: &mut Array2<f64>,
    se_epi_z: &mut Array2<f64>,
    se_ale_z: &mut Array2<f64>,
    bounds: &Array2<f64>,
) {
    if is_identity_bounds(bounds) {
        return;
    }
    Zip::from(mu_z.rows_mut())
        .and(se_z.rows_mut())
        .and(se_epi_z.rows_mut())
        .and(se_ale_z.rows_mut())
        .for_each(|mut mu_row, mut se_row, mut epi_row, mut ale_row| {
            for j in 0..mu_row.len() {
                let a = bounds[[j, 0]];
                let b = bounds[[j, 1]];
                let z = mu_row[j];
                let jac = d_inv_dz(z, a, b).abs();
                mu_row[j] = inv_scalar(z, a, b);
                se_row[j] *= jac;
                epi_row[j] *= jac;
                ale_row[j] *= jac;
            }
        });
}

/// Naturalize draws with shape `(..., metrics)` along the last axis matching bounds columns.
pub fn inv_last_axis(values: &mut Array2<f64>, bounds: &Array2<f64>) {
    if is_identity_bounds(bounds) {
        return;
    }
    for mut row in values.rows_mut() {
        for j in 0..row.len() {
            row[j] = inv_scalar(row[j], bounds[[j, 0]], bounds[[j, 1]]);
        }
    }
}

/// Compare two bound matrices with `null`↔`±∞` semantics (exact f64 match).
pub fn bounds_match(a: &Array2<f64>, b: &Array2<f64>) -> bool {
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| {
        (x.is_nan() && y.is_nan())
            || (*x == *y)
            || (*x == f64::NEG_INFINITY && *y == f64::NEG_INFINITY)
            || (*x == f64::INFINITY && *y == f64::INFINITY)
    })
}

/// Serialize bounds as JSON array of `[lo, hi]` with `null` for open sides.
pub fn bounds_to_json(bounds: &Array2<f64>) -> String {
    let mut parts = Vec::with_capacity(bounds.nrows());
    for row in bounds.rows() {
        let lo = if row[0].is_finite() {
            format!("{}", row[0])
        } else {
            "null".to_string()
        };
        let hi = if row[1].is_finite() {
            format!("{}", row[1])
        } else {
            "null".to_string()
        };
        parts.push(format!("[{lo},{hi}]"));
    }
    format!("[{}]", parts.join(","))
}

/// Parse a `[lo,hi]` pair where `null` means open (−∞ or +∞).
pub fn parse_bound_pair(lo_tok: &str, hi_tok: &str) -> Result<(f64, f64), ENNError> {
    let lo = if lo_tok == "null" {
        f64::NEG_INFINITY
    } else {
        lo_tok
            .parse()
            .map_err(|_| ENNError::InvalidParameter(format!("bad y_bounds lo: {lo_tok}")))?
    };
    let hi = if hi_tok == "null" {
        f64::INFINITY
    } else {
        hi_tok
            .parse()
            .map_err(|_| ENNError::InvalidParameter(format!("bad y_bounds hi: {hi_tok}")))?
    };
    Ok((lo, hi))
}

/// Robust metadata parse using string splitting (avoids null-position ambiguity).
pub fn load_y_bounds_from_metadata_text(
    text: &str,
    num_metrics: usize,
) -> Result<Option<Array2<f64>>, ENNError> {
    let key = "\"y_bounds\":";
    let Some(start) = text.find(key) else {
        return Ok(None);
    };
    let after = &text[start + key.len()..];
    let after = after.trim_start();
    if !after.starts_with('[') {
        return Err(ENNError::InvalidParameter(
            "y_bounds in metadata is not an array".to_string(),
        ));
    }
    // Find matching outer bracket content
    let mut depth = 0i32;
    let mut end = None;
    for (i, c) in after.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or_else(|| {
        ENNError::InvalidParameter("y_bounds array not closed".to_string())
    })?;
    let inner = &after[1..end]; // inside outer []
    let mut bounds = Array2::zeros((num_metrics, 2));
    let mut metric = 0usize;
    let mut rest = inner.trim();
    while !rest.is_empty() {
        rest = rest.trim_start_matches(|c: char| c == ',' || c.is_whitespace());
        if rest.is_empty() {
            break;
        }
        if !rest.starts_with('[') {
            return Err(ENNError::InvalidParameter(
                "y_bounds expected [lo,hi] pair".to_string(),
            ));
        }
        let close = rest.find(']').ok_or_else(|| {
            ENNError::InvalidParameter("y_bounds pair not closed".to_string())
        })?;
        let pair = &rest[1..close];
        let mut parts = pair.splitn(2, ',');
        let lo_tok = parts
            .next()
            .ok_or_else(|| ENNError::InvalidParameter("y_bounds missing lo".to_string()))?
            .trim();
        let hi_tok = parts
            .next()
            .ok_or_else(|| ENNError::InvalidParameter("y_bounds missing hi".to_string()))?
            .trim();
        let (lo, hi) = parse_bound_pair(lo_tok, hi_tok)?;
        if metric >= num_metrics {
            return Err(ENNError::InvalidParameter(format!(
                "y_bounds has more than {num_metrics} metrics"
            )));
        }
        bounds[[metric, 0]] = lo;
        bounds[[metric, 1]] = hi;
        metric += 1;
        rest = &rest[close + 1..];
    }
    if metric != num_metrics {
        return Err(ENNError::InvalidParameter(format!(
            "y_bounds length {metric} != num_metrics {num_metrics}"
        )));
    }
    validate_bounds(&bounds, num_metrics)?;
    Ok(Some(bounds))
}

/// Resolve construction-time bounds: None → unbounded (or load from disk text).
pub fn resolve_y_bounds(
    requested: Option<&Array2<f64>>,
    num_metrics: usize,
    disk_metadata: Option<&str>,
) -> Result<Array2<f64>, ENNError> {
    let loaded = match disk_metadata {
        Some(text) => load_y_bounds_from_metadata_text(text, num_metrics)?,
        None => None,
    };
    match (requested, loaded) {
        (None, None) => Ok(unbounded_bounds(num_metrics)),
        (None, Some(b)) => Ok(b),
        (Some(req), None) => {
            validate_bounds(req, num_metrics)?;
            Ok(req.clone())
        }
        (Some(req), Some(disk)) => {
            validate_bounds(req, num_metrics)?;
            if !bounds_match(req, &disk) {
                return Err(ENNError::InvalidParameter(
                    "y_bounds do not match disk metadata".to_string(),
                ));
            }
            Ok(req.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_roundtrip() {
        let b = unbounded_bounds(2);
        let y = array![[1.0, -2.0], [0.5, 3.0]];
        let z = warp_y(y.view(), &b).unwrap();
        assert_eq!(z, y);
        assert_eq!(inv_y(z.view(), &b), y);
    }

    #[test]
    fn log_and_neglog_and_logit_roundtrip() {
        let b = array![
            [0.0, f64::INFINITY],
            [f64::NEG_INFINITY, 5.0],
            [0.0, 1.0]
        ];
        let y = array![[0.1, 4.0, 0.25], [2.0, 1.0, 0.75]];
        let z = warp_y(y.view(), &b).unwrap();
        let y2 = inv_y(z.view(), &b);
        for (a, c) in y.iter().zip(y2.iter()) {
            assert!((a - c).abs() < 1e-12, "{a} vs {c}");
        }
    }

    #[test]
    fn oob_rejected() {
        let b = array![[0.0, 1.0]];
        let err = warp_y(array![[0.0]].view(), &b).unwrap_err();
        assert!(err.to_string().contains("strictly above"));
        let err = warp_y(array![[1.0]].view(), &b).unwrap_err();
        assert!(err.to_string().contains("strictly below"));
    }

    #[test]
    fn jacobian_yvar_log() {
        let b = array![[0.0, f64::INFINITY]];
        let y = array![[2.0]];
        let yvar = array![[0.25]];
        // φ=log(y-0), φ'=1/y → yvar_z = 0.25 / 4 = 0.0625
        let zv = warp_yvar(y.view(), yvar.view(), &b).unwrap();
        assert!((zv[[0, 0]] - 0.0625).abs() < 1e-15);
    }

    #[test]
    fn metadata_null_roundtrip() {
        let b = array![[0.0, f64::INFINITY], [f64::NEG_INFINITY, 1.0], [0.0, 1.0]];
        let json = bounds_to_json(&b);
        assert_eq!(json, "[[0,null],[null,1],[0,1]]");
        let text = format!("{{\"y_bounds\":{json}}}");
        let loaded = load_y_bounds_from_metadata_text(&text, 3).unwrap().unwrap();
        assert!(bounds_match(&b, &loaded));
    }

    #[test]
    fn resolve_must_match() {
        let disk = array![[0.0, 1.0]];
        let text = format!("{{\"y_bounds\":{}}}", bounds_to_json(&disk));
        let ok = resolve_y_bounds(Some(&disk), 1, Some(&text)).unwrap();
        assert!(bounds_match(&ok, &disk));
        let bad = array![[0.0, 2.0]];
        assert!(resolve_y_bounds(Some(&bad), 1, Some(&text)).is_err());
    }

    #[test]
    fn strict_shape_no_broadcast() {
        let b = array![[0.0, 1.0]];
        assert!(validate_bounds(&b, 2).is_err());
    }

    #[test]
    fn is_identity_bounds_detects_unbounded_and_bounded() {
        assert!(is_identity_bounds(&unbounded_bounds(2)));
        assert!(!is_identity_bounds(&array![[0.0, 1.0]]));
        assert!(!is_identity_bounds(&array![[0.0, f64::INFINITY]]));
    }

    #[test]
    fn d_inv_dz_matches_inverse_of_d_warp_dy() {
        // Identity
        assert!((d_inv_dz(1.5, f64::NEG_INFINITY, f64::INFINITY) - 1.0).abs() < 1e-15);
        // Log / exp: φ=ln(y-a), φ'=1/(y-a), dinv=exp(z)=y-a
        let y = 3.0_f64;
        let a = 1.0_f64;
        let z = (y - a).ln();
        let dinv = d_inv_dz(z, a, f64::INFINITY);
        let dwarp = d_warp_dy(y, a, f64::INFINITY).unwrap();
        assert!((dinv * dwarp - 1.0).abs() < 1e-12, "dinv={dinv} dwarp={dwarp}");
        // Neg-log
        let b = 5.0_f64;
        let y = 2.0_f64;
        let z = -(b - y).ln();
        let dinv = d_inv_dz(z, f64::NEG_INFINITY, b);
        let dwarp = d_warp_dy(y, f64::NEG_INFINITY, b).unwrap();
        assert!((dinv * dwarp - 1.0).abs() < 1e-12);
        // Logit on (0,1)
        let y = 0.25_f64;
        let z = (y / (1.0 - y)).ln();
        let dinv = d_inv_dz(z, 0.0, 1.0);
        let dwarp = d_warp_dy(y, 0.0, 1.0).unwrap();
        assert!((dinv * dwarp - 1.0).abs() < 1e-12);
    }

    #[test]
    fn naturalize_mu_se_applies_inv_and_jacobian() {
        let bounds = array![[0.0, 1.0]];
        let mut mu = array![[0.0]]; // logit(0.5)=0
        let mut se = array![[0.2]];
        let mut se_epi = array![[0.1]];
        let mut se_ale = array![[0.1]];
        naturalize_mu_se(&mut mu, &mut se, &mut se_epi, &mut se_ale, &bounds);
        assert!((mu[[0, 0]] - 0.5).abs() < 1e-12);
        // dinv at z=0 for logit(0,1) is 0.25
        assert!((se[[0, 0]] - 0.05).abs() < 1e-12);
        assert!((se_epi[[0, 0]] - 0.025).abs() < 1e-12);
        assert!((se_ale[[0, 0]] - 0.025).abs() < 1e-12);
        // Identity bounds are a no-op
        let b = unbounded_bounds(1);
        let mut mu2 = array![[1.0]];
        let mut se2 = array![[2.0]];
        let mut e2 = array![[3.0]];
        let mut a2 = array![[4.0]];
        naturalize_mu_se(&mut mu2, &mut se2, &mut e2, &mut a2, &b);
        assert_eq!(mu2[[0, 0]], 1.0);
        assert_eq!(se2[[0, 0]], 2.0);
    }

    #[test]
    fn inv_last_axis_roundtrips_with_warp_y() {
        let bounds = array![[0.0, 1.0], [0.0, f64::INFINITY]];
        let y = array![[0.2, 1.5], [0.8, 3.0]];
        let z = warp_y(y.view(), &bounds).unwrap();
        let mut z_mut = z.clone();
        inv_last_axis(&mut z_mut, &bounds);
        for (a, c) in y.iter().zip(z_mut.iter()) {
            assert!((a - c).abs() < 1e-12, "{a} vs {c}");
        }
        // Identity no-op
        let id = unbounded_bounds(2);
        let mut vals = array![[9.0, -1.0]];
        inv_last_axis(&mut vals, &id);
        assert_eq!(vals[[0, 0]], 9.0);
    }
}
