//! Incremental hyperparameter fitting tied to a model's `add` calls.
//!
//! `IncrementalFit::add` appends rows to the model and returns an [`AddToken`].
//! The next `ask` must present that exact token, so only the rows of the latest
//! `add` are told to the fitter, once. The first `ask` freezes `k` and the seed.

use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::ArrayView2;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::ENNError;
use crate::fitter::{ENNFitter, DEFAULT_NUM_FIT_SAMPLES};
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

/// Proof that one `add` appended rows to one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddToken {
    owner: u64,
    generation: u64,
}

struct FrozenFitter {
    fitter: ENNFitter,
    rng: StdRng,
    k: i32,
    seed: u64,
}

/// Fit options for one incremental `ask`.
#[derive(Debug, Clone, Copy)]
pub struct IncrementalAsk {
    /// Neighbor count; must equal the one frozen by the first `ask`.
    pub k: i32,
    /// Fitter seed; must equal the one frozen by the first `ask`.
    pub seed: u64,
    /// Random candidates per `ask` (`None` uses the fitter default).
    pub num_fit_candidates: Option<usize>,
    /// Draws per `ask` (`None` uses [`DEFAULT_NUM_FIT_SAMPLES`]).
    pub num_fit_samples: Option<usize>,
    /// Warm-start parameters.
    pub params_warm_start: Option<ENNParams>,
}

/// Incremental-fit state for one model: the frozen fitter and the pending `add`.
pub struct IncrementalFit {
    owner: u64,
    generation: u64,
    pending: Option<(u64, usize, usize)>,
    frozen: Option<FrozenFitter>,
}

impl Default for IncrementalFit {
    fn default() -> Self {
        Self::new()
    }
}

impl IncrementalFit {
    pub fn new() -> Self {
        Self {
            owner: NEXT_OWNER_ID.fetch_add(1, Ordering::Relaxed),
            generation: 0,
            pending: None,
            frozen: None,
        }
    }

    /// Append rows to `model` and make them the only rows the next `ask` may tell.
    pub fn add(
        &mut self,
        model: &mut EpistemicNearestNeighbors,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<AddToken, ENNError> {
        let start = model.len();
        model.add(x, y, yvar)?;
        self.generation += 1;
        self.pending = Some((self.generation, start, model.len()));
        Ok(AddToken {
            owner: self.owner,
            generation: self.generation,
        })
    }

    fn check_frozen(&self, k: i32, seed: u64) -> Result<(), ENNError> {
        match &self.frozen {
            Some(f) if f.k != k || f.seed != seed => Err(ENNError::InvalidParameter(format!(
                "incremental enn_fit freezes k and the fitter seed on the first call; \
                 frozen k={}, frozen seed={}, got k={k}, seed={seed}",
                f.k, f.seed
            ))),
            _ => Ok(()),
        }
    }

    fn take(&mut self, token: AddToken) -> Result<(usize, usize), ENNError> {
        if token.owner != self.owner {
            return Err(ENNError::InvalidParameter(
                "fit token belongs to a different model".to_string(),
            ));
        }
        match self.pending {
            Some((generation, start, end)) if generation == token.generation => {
                self.pending = None;
                Ok((start, end))
            }
            _ => Err(ENNError::InvalidParameter(
                "fit token is stale or already used".to_string(),
            )),
        }
    }

    /// Tell the rows of `token`'s `add`, then fit and return parameters.
    pub fn ask(
        &mut self,
        model: &EpistemicNearestNeighbors,
        token: AddToken,
        opts: &IncrementalAsk,
    ) -> Result<ENNParams, ENNError> {
        self.check_frozen(opts.k, opts.seed)?;
        let (start, end) = self.take(token)?;
        let frozen = self.frozen.get_or_insert_with(|| FrozenFitter {
            fitter: ENNFitter::new(opts.k, true),
            rng: StdRng::seed_from_u64(opts.seed),
            k: opts.k,
            seed: opts.seed,
        });
        let indices: Vec<usize> = (start..end).collect();
        let (x, y, yvar) = model.train_rows_at(&indices)?;
        let yvar_view = yvar.as_ref().map(|v| v.view());
        frozen.fitter.tell(
            &x.view(),
            &y.view(),
            yvar_view.as_ref(),
            Some(model.y_bounds()),
        )?;
        frozen.fitter.ask(
            model,
            opts.num_fit_candidates,
            opts.num_fit_samples.unwrap_or(DEFAULT_NUM_FIT_SAMPLES),
            opts.params_warm_start.as_ref(),
            &mut frozen.rng,
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::layout::EnnLayout;
    use ndarray::Array2;

    fn empty_model() -> EpistemicNearestNeighbors {
        EpistemicNearestNeighbors::new_with_storage(
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            EnnLayout::memory(IndexDriver::Flat, false),
            None,
        )
        .unwrap()
    }

    fn opts(k: i32, seed: u64) -> IncrementalAsk {
        IncrementalAsk {
            k,
            seed,
            num_fit_candidates: Some(1),
            num_fit_samples: Some(4),
            params_warm_start: None,
        }
    }

    fn add_row(fit: &mut IncrementalFit, model: &mut EpistemicNearestNeighbors, v: f64) -> AddToken {
        let x = Array2::from_elem((1, 2), v);
        let y = Array2::from_elem((1, 1), v * v);
        fit.add(model, &x.view(), &y.view(), None).unwrap()
    }

    #[test]
    fn token_is_single_use_and_latest_only() {
        let mut model = empty_model();
        let mut fit = IncrementalFit::new();
        let first = add_row(&mut fit, &mut model, 0.1);
        let second = add_row(&mut fit, &mut model, 0.7);
        let err = fit.ask(&model, first, &opts(2, 1)).unwrap_err();
        assert!(err.to_string().contains("stale or already used"));
        fit.ask(&model, second, &opts(2, 1)).unwrap();
        let err = fit.ask(&model, second, &opts(2, 1)).unwrap_err();
        assert!(err.to_string().contains("stale or already used"));
    }

    #[test]
    fn foreign_token_and_changed_k_are_rejected() {
        let mut model = empty_model();
        let mut fit = IncrementalFit::new();
        let mut other_model = empty_model();
        let mut other = IncrementalFit::new();
        let foreign = add_row(&mut other, &mut other_model, 0.3);
        let err = fit.ask(&model, foreign, &opts(2, 1)).unwrap_err();
        assert!(err.to_string().contains("different model"));
        let first = add_row(&mut fit, &mut model, 0.2);
        fit.ask(&model, first, &opts(2, 1)).unwrap();
        let second = add_row(&mut fit, &mut model, 0.9);
        let err = fit.ask(&model, second, &opts(3, 1)).unwrap_err();
        assert!(err.to_string().contains("freezes k"));
        let params = fit.ask(&model, second, &opts(2, 1)).unwrap();
        assert_eq!(params.k_num_neighbors, 2);
    }
}
