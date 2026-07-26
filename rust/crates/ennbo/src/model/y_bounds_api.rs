//! y_bounds construction, warp ingress, and naturalized row gathers (kiss split).

use ndarray::{Array1, Array2, ArrayView2};
use std::path::{Path, PathBuf};

use super::EpistemicNearestNeighbors;
use crate::backend::{EnnBackend, EnnStorage, TrainRowsAtResult};
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::y_bounds::{
    inv_y, is_identity_bounds, resolve_y_bounds, warp_y, warp_yvar,
};

impl EpistemicNearestNeighbors {
    /// Empty model with optional `y_bounds` (`None` → unbounded; disk may load later via reopen).
    pub fn new_empty_with_y_bounds(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
        pending_flush_threshold: Option<usize>,
        y_bounds: Option<Array2<f64>>,
    ) -> Result<Self, ENNError> {
        let stored_work_dir = work_dir.clone().or_else(EnnStorage::work_dir_from_env);
        let meta_text = stored_work_dir
            .as_ref()
            .filter(|p| matches!(storage, EnnStorage::Disk) && p.join("metadata.json").exists())
            .and_then(|p| std::fs::read_to_string(p.join("metadata.json")).ok());
        let y_bounds = resolve_y_bounds(y_bounds.as_ref(), num_metrics, meta_text.as_deref())?;
        let backend = EnnBackend::new_empty(
            num_dim,
            num_metrics,
            driver,
            storage,
            work_dir,
            pending_flush_threshold,
        )?;
        let model = Self {
            backend,
            num_obs: 0,
            num_dim,
            num_metrics,
            scale_x: false,
            x_scale: Array1::ones(num_dim),
            y_scale: Array1::ones(num_metrics),
            y_bounds,
            y_sum: Array1::zeros(num_metrics),
            y_sumsq: Array1::zeros(num_metrics),
            x_sum: Array1::zeros(num_dim),
            x_sumsq: Array1::zeros(num_dim),
            work_dir: stored_work_dir,
        };
        model.persist_y_bounds_metadata()?;
        Ok(model)
    }

    pub(crate) fn ingress_warp_owned(
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        y_bounds: &Array2<f64>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
        if is_identity_bounds(y_bounds) {
            return Ok((train_y, train_yvar));
        }
        let z = warp_y(train_y.view(), y_bounds)?;
        let zvar = match train_yvar {
            Some(yv) => Some(warp_yvar(train_y.view(), yv.view(), y_bounds)?),
            None => None,
        };
        Ok((z, zvar))
    }

    /// Warp natural-unit observations for storage / fit (copy; does not mutate inputs).
    pub fn warp_observations(
        &self,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
        if is_identity_bounds(&self.y_bounds) {
            return Ok((y.to_owned(), yvar.map(|v| v.to_owned())));
        }
        let z = warp_y(*y, &self.y_bounds)?;
        let zvar = match yvar {
            Some(yv) => Some(warp_yvar(*y, *yv, &self.y_bounds)?),
            None => None,
        };
        Ok((z, zvar))
    }

    pub(crate) fn persist_y_bounds_metadata(&self) -> Result<(), ENNError> {
        let Some(dir) = self.work_dir.as_ref() else {
            return Ok(());
        };
        patch_metadata_y_bounds(dir, &self.y_bounds)
    }

    /// Per-metric natural-unit y bounds, shape `(num_metrics, 2)`.
    pub fn y_bounds(&self) -> &Array2<f64> {
        &self.y_bounds
    }

    /// Public row gather: y (and yvar) naturalized; x unchanged. Storage stays warped.
    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<TrainRowsAtResult, ENNError> {
        let (x, y_z, yvar_z) = self.rows().train_rows_at(indices)?;
        if is_identity_bounds(&self.y_bounds) {
            return Ok((x, y_z, yvar_z));
        }
        let y = inv_y(y_z.view(), &self.y_bounds);
        let yvar = match yvar_z {
            Some(zv) => {
                let mut yv = Array2::zeros(zv.raw_dim());
                for i in 0..y.nrows() {
                    for j in 0..y.ncols() {
                        let d = crate::y_bounds::d_inv_dz(
                            y_z[[i, j]],
                            self.y_bounds[[j, 0]],
                            self.y_bounds[[j, 1]],
                        )
                        .abs();
                        yv[[i, j]] = zv[[i, j]] * d * d;
                    }
                }
                Some(yv)
            }
            None => None,
        };
        Ok((x, y, yvar))
    }

    /// Public single-row y in natural units.
    pub fn row_y_natural(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        let y_z = self.rows().row_y(i)?;
        if is_identity_bounds(&self.y_bounds) {
            return Ok(y_z);
        }
        let y2 = inv_y(y_z.insert_axis(ndarray::Axis(0)).view(), &self.y_bounds);
        Ok(y2.row(0).to_owned())
    }
}

fn patch_metadata_y_bounds(work_dir: &Path, bounds: &Array2<f64>) -> Result<(), ENNError> {
    use crate::y_bounds::bounds_to_json;
    let meta_path = work_dir.join("metadata.json");
    if !meta_path.exists() {
        return Ok(());
    }
    let text = std::fs::read_to_string(&meta_path)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let yb = format!("\"y_bounds\":{}", bounds_to_json(bounds));
    let new_text = if text.contains("\"y_bounds\":") {
        let key = "\"y_bounds\":";
        let start = text.find(key).unwrap();
        let after = &text[start + key.len()..];
        let after_trim = after.trim_start();
        let mut depth = 0i32;
        let mut end_rel = None;
        for (i, c) in after_trim.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end_rel = Some(i + c.len_utf8());
                        break;
                    }
                }
                _ => {}
            }
        }
        let end_rel = end_rel.ok_or_else(|| {
            ENNError::InvalidParameter("malformed y_bounds in metadata".to_string())
        })?;
        let abs_start = start;
        let abs_end = start + key.len() + (after.len() - after_trim.len()) + end_rel;
        format!("{}{}{}", &text[..abs_start], yb, &text[abs_end..])
    } else if let Some(pos) = text.rfind('}') {
        let insert = if text[..pos].trim_end().ends_with('{') {
            yb
        } else {
            format!(",{yb}")
        };
        format!("{}{}{}", &text[..pos], insert, &text[pos..])
    } else {
        return Err(ENNError::InvalidParameter(
            "metadata.json missing closing brace".to_string(),
        ));
    };
    std::fs::write(&meta_path, new_text).map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::array;

    #[test]
    fn new_empty_with_y_bounds_stores_and_exposes_bounds() {
        let bounds = array![[0.0, 1.0]];
        let model = EpistemicNearestNeighbors::new_empty_with_y_bounds(
            1,
            1,
            IndexDriver::Exact,
            EnnStorage::InMemory,
            None,
            None,
            Some(bounds.clone()),
        )
        .unwrap();
        assert!(crate::y_bounds::bounds_match(model.y_bounds(), &bounds));
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn row_y_natural_inverse_warps_logit_storage() {
        let train_x = array![[0.0], [1.0]];
        let train_y = array![[0.2], [0.8]];
        let bounds = array![[0.0, 1.0]];
        let model = EpistemicNearestNeighbors::new_with_storage(
            train_x,
            train_y.clone(),
            None,
            false,
            IndexDriver::Exact,
            EnnStorage::InMemory,
            None,
            Some(bounds),
        )
        .unwrap();
        let y0 = model.row_y_natural(0).unwrap();
        let y1 = model.row_y_natural(1).unwrap();
        assert!((y0[0] - 0.2).abs() < 1e-12);
        assert!((y1[0] - 0.8).abs() < 1e-12);
        // Storage row differs from natural under logit.
        let stored = model.rows().row_y(0).unwrap();
        assert!((stored[0] - y0[0]).abs() > 1e-6);
    }
}
