mod add;
mod lsh;
mod predict;
mod search;
mod store;

use std::collections::HashMap;

use ndarray::ArrayView2;

use crate::error::ENNError;

pub use predict::PosteriorArrays;
pub use search::SearchBatch;

use lsh::LshFamily;
use store::{Posting, SeriesBuf};

#[derive(Clone, Debug)]
pub struct EnnTsConfig {
    pub lookback: usize,
    pub horizon: i64,
    pub cadence: i64,
    pub num_tables: usize,
    pub num_projections: usize,
    pub bucket_width: f64,
    pub k: usize,
    pub probe: i32,
    pub seed: u64,
    pub epistemic_variance_scale: f64,
    pub aleatoric_variance_scale: f64,
}

impl EnnTsConfig {
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback,
            horizon: 1,
            cadence: 1,
            num_tables: 8,
            num_projections: 4,
            bucket_width: 4.0,
            k: 8,
            probe: 1,
            seed: 0,
            epistemic_variance_scale: 1.0,
            aleatoric_variance_scale: 0.1,
        }
    }

    fn invalid_reason(&self) -> Option<&'static str> {
        let bad = [
            (self.lookback == 0, "lookback must be >= 1"),
            (self.horizon <= 0, "horizon must be >= 1"),
            (self.cadence <= 0, "cadence must be >= 1"),
            (self.num_tables == 0, "num_tables must be >= 1"),
            (self.num_projections == 0, "num_projections must be >= 1"),
            (
                !(self.bucket_width.is_finite() && self.bucket_width > 0.0),
                "bucket_width must be finite and > 0",
            ),
            (self.k == 0, "k must be >= 1"),
            (self.probe < 0, "probe must be >= 0"),
            (
                !(self.epistemic_variance_scale.is_finite()
                    && self.epistemic_variance_scale >= 0.0),
                "epistemic_variance_scale must be finite and >= 0",
            ),
            (
                !(self.aleatoric_variance_scale.is_finite()
                    && self.aleatoric_variance_scale >= 0.0),
                "aleatoric_variance_scale must be finite and >= 0",
            ),
        ];
        bad.iter().find(|(flag, _)| *flag).map(|(_, msg)| *msg)
    }
}

pub struct QueryOptions {
    pub cutoff: i64,
    pub k: usize,
    pub require_target: bool,
}

pub struct EnnTs {
    lookback: usize,
    config: EnnTsConfig,
    dim: Option<usize>,
    metrics: usize,
    lsh: Option<LshFamily>,
    series: HashMap<u64, SeriesBuf>,
    buckets: Vec<HashMap<Vec<i32>, Vec<Posting>>>,
    windows: Vec<Posting>,
    n_obs: usize,
    n_y: usize,
    y_sum: Vec<f64>,
    y_sumsq: Vec<f64>,
}

impl EnnTs {
    pub fn new(config: EnnTsConfig) -> Result<Self, ENNError> {
        if let Some(msg) = config.invalid_reason() {
            return Err(ENNError::InvalidParameter(msg.into()));
        }
        let lookback = config.lookback;
        Ok(Self {
            lookback,
            config,
            dim: None,
            metrics: 0,
            lsh: None,
            series: HashMap::new(),
            buckets: Vec::new(),
            windows: Vec::new(),
            n_obs: 0,
            n_y: 0,
            y_sum: Vec::new(),
            y_sumsq: Vec::new(),
        })
    }

    pub fn add(
        &mut self,
        series_id: u64,
        timestamps: &[i64],
        x: ArrayView2<f64>,
        y: Option<ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        add::add(self, series_id, timestamps, x, y)
    }

    pub fn add_y(
        &mut self,
        series_id: u64,
        timestamps: &[i64],
        y: ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        add::add_y(self, series_id, timestamps, y)
    }

    pub fn auto_timestamps(&self, series_id: u64, n: usize) -> Result<Vec<i64>, ENNError> {
        add::auto_timestamps(self, series_id, n)
    }

    pub fn search(
        &self,
        queries: ArrayView2<f64>,
        options: &QueryOptions,
    ) -> Result<SearchBatch, ENNError> {
        search::search(self, queries, options, false)
    }

    pub fn search_exact(
        &self,
        queries: ArrayView2<f64>,
        options: &QueryOptions,
    ) -> Result<SearchBatch, ENNError> {
        search::search(self, queries, options, true)
    }

    pub fn posterior(
        &self,
        queries: ArrayView2<f64>,
        cutoff: i64,
        k: Option<usize>,
        observation_noise: bool,
    ) -> Result<PosteriorArrays, ENNError> {
        predict::posterior(
            self,
            queries,
            cutoff,
            k.unwrap_or(self.config.k),
            observation_noise,
        )
    }

    pub fn set_variance_scales(&mut self, epistemic: f64, aleatoric: f64) -> Result<(), ENNError> {
        let mut trial = self.config.clone();
        trial.epistemic_variance_scale = epistemic;
        trial.aleatoric_variance_scale = aleatoric;
        if let Some(msg) = trial.invalid_reason() {
            return Err(ENNError::InvalidParameter(msg.into()));
        }
        trial.lookback = self.lookback;
        self.config = trial;
        Ok(())
    }

    pub fn num_obs(&self) -> usize {
        self.n_obs
    }

    pub fn num_windows(&self) -> usize {
        self.windows.len()
    }

    pub fn num_dim(&self) -> Option<usize> {
        self.dim
    }

    pub fn lookback(&self) -> usize {
        self.lookback
    }

    pub fn neighbor_k(&self) -> usize {
        self.config.k
    }
}

#[cfg(test)]
mod tests;
