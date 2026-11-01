use std::sync::atomic::{AtomicU64, Ordering};

use ndarray::ArrayView2;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::error::ENNError;
use crate::fit_samples::ScaleSearch;
use crate::fitter::ENNFitter;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

static NEXT_OWNER_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddToken {
    owner: u64,
    generation: u64,
}

struct FrozenFitter {
    fitter: ENNFitter,
    rng: StdRng,
    k: i32,
}

#[derive(Debug, Clone)]
pub struct IncrementalAsk {
    pub k: i32,
    pub seed: u64,
    pub num_fit_candidates: Option<usize>,
    pub num_fit_samples: Option<usize>,
    pub params_warm_start: Option<ENNParams>,
}

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

    pub fn add(
        &mut self,
        model: &mut EpistemicNearestNeighbors,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<AddToken, ENNError> {
        let start = self.pending.map_or(model.len(), |(_, start, _)| start);
        model.add(x, y, yvar)?;
        self.generation += 1;
        self.pending = Some((self.generation, start, model.len()));
        Ok(AddToken {
            owner: self.owner,
            generation: self.generation,
        })
    }

    fn check_frozen(&self, k: i32) -> Result<(), ENNError> {
        match &self.frozen {
            Some(f) if f.k != k => Err(ENNError::InvalidParameter(format!(
                "incremental enn_fit freezes k on the first call; frozen k={}, got k={k}",
                f.k
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

    pub fn ask(
        &mut self,
        model: &EpistemicNearestNeighbors,
        token: AddToken,
        opts: &IncrementalAsk,
    ) -> Result<ENNParams, ENNError> {
        self.check_frozen(opts.k)?;
        let search = ScaleSearch::from_counts(opts.num_fit_samples, opts.num_fit_candidates)?;
        let (start, end) = self.take(token)?;
        let is_first_ask = self.frozen.is_none();
        let frozen = self.frozen.get_or_insert_with(|| FrozenFitter {
            fitter: ENNFitter::new(opts.k),
            rng: StdRng::seed_from_u64(opts.seed),
            k: opts.k,
        });
        if is_first_ask {
            frozen.fitter.seed_y_stats(model);
        } else {
            let indices: Vec<usize> = (start..end).collect();
            let (x, y, yvar) = model.train_rows_at(&indices)?;
            let yvar_view = yvar.as_ref().map(|v| v.view());
            frozen.fitter.tell(
                &x.view(),
                &y.view(),
                yvar_view.as_ref(),
                Some(model.y_bounds()),
            )?;
        }
        frozen.fitter.ask(
            model,
            &search,
            opts.params_warm_start.as_ref(),
            &mut frozen.rng,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn empty_model() -> crate::test_helpers::TestModel {
        crate::test_helpers::test_model(Array2::zeros((0, 2)), Array2::zeros((0, 1)), None, false)
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

    fn add_row(
        fit: &mut IncrementalFit,
        model: &mut EpistemicNearestNeighbors,
        v: f64,
    ) -> AddToken {
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
        let params = fit.ask(&model, second, &opts(2, 9)).unwrap();
        assert_eq!(params.k_num_neighbors, 2);
    }

    #[test]
    fn seeded_y_stats_match_tell_with_y_bounds_and_on_disk_reopen() {
        let dir = tempfile::TempDir::new().unwrap();
        let x = Array2::from_shape_fn((6, 2), |(i, j)| (i * (j + 1)) as f64 * 0.1);
        let y = Array2::from_shape_fn((6, 1), |(i, _)| 0.1 + i as f64 * 0.1);
        let bounds = ndarray::array![[0.0, 1.0]];
        let open = |x, y| {
            EpistemicNearestNeighbors::new_with_storage(
                x,
                y,
                None,
                crate::layout::EnnLayout::new(dir.path().to_path_buf(), false).unwrap(),
                Some(bounds.clone()),
            )
            .unwrap()
        };
        let model = open(x.clone(), y.clone());
        model.persist_index_to_disk().unwrap();
        drop(model);
        let mut reopened = open(Array2::zeros((0, 2)), Array2::zeros((0, 1)));
        let mut fit = IncrementalFit::new();
        let token = add_row(&mut fit, &mut reopened, 0.7);
        fit.ask(&reopened, token, &opts(2, 1)).unwrap();
        let all_x = ndarray::concatenate![ndarray::Axis(0), x, Array2::from_elem((1, 2), 0.7)];
        let all_y = ndarray::concatenate![ndarray::Axis(0), y, ndarray::array![[0.49]]];
        let mut told = ENNFitter::new(2);
        told.tell(&all_x.view(), &all_y.view(), None, Some(&bounds))
            .unwrap();
        let got = fit.frozen.as_ref().unwrap().fitter.y_std()[0];
        assert!((got - told.y_std()[0]).abs() < 1e-9);
    }

    #[test]
    fn rows_from_skipped_tokens_still_reach_the_fitter() {
        let mut model = empty_model();
        let mut fit = IncrementalFit::new();
        let token = add_row(&mut fit, &mut model, 0.1);
        fit.ask(&model, token, &opts(2, 1)).unwrap();
        add_row(&mut fit, &mut model, 0.5);
        let token = add_row(&mut fit, &mut model, 0.7);
        fit.ask(&model, token, &opts(2, 1)).unwrap();
        let want = ndarray::array![0.01f64, 0.25, 0.49].std(0.0);
        let got = fit.frozen.as_ref().unwrap().fitter.y_std()[0];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn first_ask_seeds_y_stats_from_constructor_rows_and_later_adds_accumulate() {
        let x = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((4, 1), |(i, _)| (i * i) as f64);
        let dir = tempfile::tempdir().unwrap();
        let mut model = EpistemicNearestNeighbors::new_with_storage(
            x,
            y.clone(),
            None,
            crate::layout::EnnLayout::new(dir.path(), false).unwrap(),
            None,
        )
        .unwrap();
        let mut fit = IncrementalFit::new();
        let token = add_row(&mut fit, &mut model, 2.0);
        fit.ask(&model, token, &opts(2, 1)).unwrap();
        let token = add_row(&mut fit, &mut model, 3.0);
        fit.ask(&model, token, &opts(2, 1)).unwrap();
        let all = ndarray::concatenate![ndarray::Axis(0), y, ndarray::array![[4.0], [9.0]]];
        let want = all.std_axis(ndarray::Axis(0), 0.0)[0];
        let got = fit.frozen.as_ref().unwrap().fitter.y_std()[0];
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
