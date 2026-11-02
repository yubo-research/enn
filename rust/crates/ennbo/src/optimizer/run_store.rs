use std::path::{Path, PathBuf};

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use ndarray::{concatenate, Array1, Array2, ArrayView2, Axis};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use super::durable_sync::{sync_dir, BackgroundSync};
use super::Optimizer;
use crate::config::ConfigOverrides;
use crate::error::ENNError;
use crate::incumbent_tracker::IncrementalIncumbentTracker;
use crate::input_history_disk::DiskInputs;
use crate::layout::EnnLayout;
use crate::strategy::Strategy;
use crate::surrogate_state::SurrogateState;

const RUN_FILE: &str = "optimizer_run.json";
const STATE_FILE: &str = "optimizer_state.json";
const OBSERVATIONS_FILE: &str = "optimizer_observations.bin";
const FORMAT_VERSION: u32 = 1;

fn file_error(e: ennbo_bpann::BpannError) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunKind {
    Enn,
    Zero,
    Lhd,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecipe {
    pub kind: RunKind,
    #[serde(with = "crate::exact")]
    pub bounds: Array2<f64>,
    pub k: Option<i32>,
    pub num_init: Option<usize>,
    pub overrides: Option<ConfigOverrides>,
    version: u32,
}

impl RunRecipe {
    pub fn new(
        kind: RunKind,
        bounds: Array2<f64>,
        k: Option<i32>,
        num_init: Option<usize>,
        overrides: Option<ConfigOverrides>,
    ) -> Self {
        Self {
            kind,
            bounds,
            k,
            num_init,
            overrides,
            version: FORMAT_VERSION,
        }
    }

    fn build(&self, work_dir: &Path, seed: u64) -> Result<Optimizer, ENNError> {
        use crate::optimizer_factory::{
            create_optimizer_enn_with_overrides, create_optimizer_lhd_with_overrides,
            create_optimizer_zero_with_overrides,
        };
        let bounds = self.bounds.clone();
        let overrides = self.overrides.as_ref();
        match self.kind {
            RunKind::Enn => create_optimizer_enn_with_overrides(
                bounds,
                EnnLayout::new(work_dir, false)?,
                self.k,
                self.num_init,
                seed,
                overrides,
            ),
            RunKind::Zero => {
                create_optimizer_zero_with_overrides(bounds, self.num_init, seed, overrides)
            }
            RunKind::Lhd => {
                create_optimizer_lhd_with_overrides(bounds, self.num_init, seed, overrides)
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
pub(super) struct OptimizerState {
    version: u32,
    restart_pending: bool,
    obs_count: usize,
    rng_seed: [u8; 32],
    rng_word_pos: String,
    rng_stream: u64,
    strategy: Strategy,
    tr_state: super::TrustRegionState,
    incumbent_idx: Option<usize>,
    #[serde(with = "crate::exact")]
    incumbent_x_unit: Option<Array1<f64>>,
    #[serde(with = "crate::exact")]
    incumbent_y_scalar: Option<Array1<f64>>,
    #[serde(default)]
    best_observation: Option<super::incumbent::BestObservation>,
    restart_generation: usize,
    yvar_on_tell: Option<bool>,
    reinit_left: Option<usize>,
    sobol_seed_base: u64,
    sobol_position: u64,
    #[serde(with = "crate::exact")]
    sobol_shift: Option<Vec<f64>>,
    incumbent_tracker: IncrementalIncumbentTracker,
    surrogate: Option<SurrogateState>,
}

pub(super) struct RunStore {
    work_dir: PathBuf,
    observations: Option<DiskInputs>,
    sync: BackgroundSync,
}

impl RunStore {
    fn new(work_dir: PathBuf) -> Self {
        Self {
            sync: BackgroundSync::for_work_dir(work_dir.clone(), STATE_FILE),
            work_dir,
            observations: None,
        }
    }

    pub(super) fn append_observation(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        let rows = concatenate(Axis(1), &[x.view(), y.view()])
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        match &mut self.observations {
            Some(log) => log.append(&rows.view()),
            None => {
                let log = DiskInputs::create(&self.work_dir, OBSERVATIONS_FILE, &rows.view())?;
                self.observations = Some(log);
                Ok(())
            }
        }
    }

    pub(super) fn clear_observations(&mut self) -> Result<(), ENNError> {
        self.observations = None;
        match std::fs::remove_file(self.work_dir.join(OBSERVATIONS_FILE)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                Err(ENNError::InvalidParameter(e.to_string()))
            }
            _ => Ok(()),
        }
    }

    fn load_observations(&mut self) -> Result<Option<Array2<f64>>, ENNError> {
        let Some((log, rows)) = DiskInputs::reopen(&self.work_dir, OBSERVATIONS_FILE)? else {
            return Ok(None);
        };
        let all: Vec<usize> = (0..rows).collect();
        let stored = log.gather(&all)?;
        self.observations = Some(log);
        Ok(Some(stored))
    }

    fn write_state(&self, state: &OptimizerState) -> Result<(), ENNError> {
        bpann_write_json(&self.work_dir.join(STATE_FILE), state).map_err(file_error)?;
        self.sync.request()?;
        let rows_will_be_deleted_next = state.restart_pending;
        if rows_will_be_deleted_next {
            self.sync.wait_until_on_disk()?;
        }
        Ok(())
    }

    fn sync_new_run(&self) -> Result<(), ENNError> {
        self.sync.wait_until_on_disk()?;
        let Some(parent) = self.work_dir.parent().filter(|p| !p.as_os_str().is_empty()) else {
            return Ok(());
        };
        sync_dir(parent).map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }
}

pub fn create_run(recipe: RunRecipe, seed: u64, work_dir: PathBuf) -> Result<Optimizer, ENNError> {
    for name in [RUN_FILE, STATE_FILE] {
        if work_dir.join(name).exists() {
            return Err(ENNError::InvalidParameter(format!(
                "work_dir {} already holds an optimizer run; reopen it with Optimizer.open",
                work_dir.display()
            )));
        }
    }
    if recipe.kind == RunKind::Enn {
        crate::store_identity::claim_store_dir(&work_dir)?;
    }
    let mut optimizer = recipe.build(&work_dir, seed)?;
    std::fs::create_dir_all(&work_dir).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    bpann_write_json(&work_dir.join(RUN_FILE), &recipe).map_err(file_error)?;
    optimizer.run_store = Some(RunStore::new(work_dir));
    optimizer.write_snapshot(false)?;
    if let Some(store) = optimizer.run_store.as_ref() {
        store.sync_new_run()?;
    }
    Ok(optimizer)
}

fn missing_run(work_dir: &Path) -> ENNError {
    ENNError::InvalidParameter(format!(
        "work_dir {} holds no optimizer run; start one with create_optimizer",
        work_dir.display()
    ))
}

impl Optimizer {
    pub fn open(work_dir: impl Into<PathBuf>) -> Result<Self, ENNError> {
        let work_dir = work_dir.into();
        let recipe = bpann_read_json::<RunRecipe>(&work_dir.join(RUN_FILE))
            .map_err(file_error)?
            .ok_or_else(|| missing_run(&work_dir))?;
        let state = bpann_read_json::<OptimizerState>(&work_dir.join(STATE_FILE))
            .map_err(file_error)?
            .ok_or_else(|| missing_run(&work_dir))?;
        if recipe.version != FORMAT_VERSION || state.version != FORMAT_VERSION {
            return Err(ENNError::InvalidParameter(format!(
                "work_dir {} holds an unsupported optimizer format",
                work_dir.display()
            )));
        }
        let mut optimizer = recipe.build(&work_dir, 0)?;
        optimizer.run_store = Some(RunStore::new(work_dir));
        optimizer.restore(state)?;
        Ok(optimizer)
    }

    pub(super) fn write_snapshot(&self, restart_pending: bool) -> Result<(), ENNError> {
        let Some(store) = self.run_store.as_ref() else {
            return Ok(());
        };
        let surrogate = match self.surrogate.as_ref() {
            Some(surrogate) => surrogate.durable_state()?,
            None => None,
        };
        store.write_state(&OptimizerState {
            version: FORMAT_VERSION,
            restart_pending,
            obs_count: self.obs_count(),
            rng_seed: self.rng.get_seed(),
            rng_word_pos: self.rng.get_word_pos().to_string(),
            rng_stream: self.rng.get_stream(),
            strategy: self.strategy.clone(),
            tr_state: self.tr_state.clone(),
            incumbent_idx: self.incumbent_idx,
            incumbent_x_unit: self.incumbent_x_unit.clone(),
            incumbent_y_scalar: self.incumbent_y_scalar.clone(),
            best_observation: self.best_observation.clone(),
            restart_generation: self.restart_generation,
            yvar_on_tell: self.yvar_on_tell,
            reinit_left: self.reinit_left,
            sobol_seed_base: self.sobol_seed_base,
            sobol_position: self.sobol_engine.as_ref().map_or(0, |e| e.position()),
            sobol_shift: self
                .sobol_engine
                .as_ref()
                .and_then(|e| e.shift().map(<[f64]>::to_vec)),
            incumbent_tracker: self.incumbent_tracker.clone(),
            surrogate,
        })
    }

    pub(super) fn flush_rows(&self) -> Result<(), ENNError> {
        if self.run_store.is_none() {
            return Ok(());
        }
        match self.surrogate.as_ref() {
            Some(surrogate) => surrogate.flush_rows(),
            None => Ok(()),
        }
    }

    fn restore(&mut self, state: OptimizerState) -> Result<(), ENNError> {
        let mut rng = ChaCha12Rng::from_seed(state.rng_seed);
        rng.set_stream(state.rng_stream);
        rng.set_word_pos(
            state
                .rng_word_pos
                .parse()
                .map_err(|_| ENNError::InvalidParameter("bad generator position".into()))?,
        );
        self.rng = rng;
        self.strategy = state.strategy;
        self.tr_state = state.tr_state;
        self.incumbent_idx = state.incumbent_idx;
        self.incumbent_x_unit = state.incumbent_x_unit;
        self.incumbent_y_scalar = state.incumbent_y_scalar;
        self.best_observation = state.best_observation;
        self.restart_generation = state.restart_generation;
        self.yvar_on_tell = state.yvar_on_tell;
        self.reinit_left = state.reinit_left;
        self.sobol_seed_base = state.sobol_seed_base;
        if self.sobol_engine.is_some() {
            self.sobol_engine = Some(crate::candidates::SobolEngine::restored(
                self.num_dim,
                state.sobol_position,
                state.sobol_shift,
            )?);
        }
        self.incumbent_tracker = state.incumbent_tracker;
        if state.restart_pending {
            return self.finish_pending_restart();
        }
        self.restore_rows(state.surrogate)?;
        self.backfill_best_observation();
        self.reconcile_rows(state.obs_count)
    }

    fn backfill_best_observation(&mut self) {
        if self.best_observation.is_some() {
            return;
        }
        if let (Some(x), Some(y)) = (self.x_obs_unit(), self.y_obs()) {
            self.record_best_observation(&x.view(), &y.view());
        }
    }

    fn finish_pending_restart(&mut self) -> Result<(), ENNError> {
        if let Some(store) = self.run_store.as_mut() {
            crate::input_history::remove_store(&store.work_dir)?;
            store.clear_observations()?;
        }
        self.begin_local_restart()?;
        self.write_snapshot(false)
    }

    fn restore_rows(&mut self, saved: Option<SurrogateState>) -> Result<(), ENNError> {
        if let Some(surrogate) = self.surrogate.as_mut() {
            return surrogate.restore_state(saved);
        }
        let num_dim = self.num_dim;
        let Some(store) = self.run_store.as_mut() else {
            return Ok(());
        };
        if let Some(rows) = store.load_observations()? {
            for row in rows.rows() {
                self.fallback_x.push(row.slice(ndarray::s![..num_dim]).to_owned());
                self.fallback_y.push(row.slice(ndarray::s![num_dim..]).to_owned());
            }
        }
        Ok(())
    }

    fn reconcile_rows(&mut self, recorded: usize) -> Result<(), ENNError> {
        let stored = self.obs_count();
        if stored < recorded {
            return Err(ENNError::InvalidParameter(format!(
                "the store holds {stored} rows but the optimizer state records {recorded}"
            )));
        }
        if stored > recorded {
            self.replay_rows(recorded, stored)?;
            self.flush_rows()?;
            self.write_snapshot(false)?;
        }
        Ok(())
    }

    fn stored_rows(&self, start: usize, end: usize) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let access = super::obs_access::ObsAccess::new(self);
        let first_y = access.obs_row_y(start)?;
        let mut x = Array2::zeros((end - start, self.num_dim));
        let mut y = Array2::zeros((end - start, first_y.len()));
        for i in start..end {
            x.row_mut(i - start).assign(&access.obs_row_x(i)?);
            y.row_mut(i - start).assign(&access.obs_row_y(i)?);
        }
        Ok((x, y))
    }

    fn replay_rows(&mut self, start: usize, end: usize) -> Result<(), ENNError> {
        let (x, y) = self.stored_rows(start, end)?;
        self.fallback_x.truncate(start);
        self.fallback_y.truncate(start);
        self.replay_base = Some(start);
        if let Some(surrogate) = self.surrogate.as_mut() {
            surrogate.set_replaying(true);
        }
        let result = self.apply_tell(&x.view(), &y.view(), None);
        if let Some(surrogate) = self.surrogate.as_mut() {
            surrogate.set_replaying(false);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn recipe(kind: RunKind) -> RunRecipe {
        RunRecipe::new(kind, array![[0.0, 1.0], [0.0, 2.0]], Some(3), Some(3), None)
    }

    fn objective(x: &Array2<f64>) -> Array2<f64> {
        x.map_axis(Axis(1), |row| -row.sum()).insert_axis(Axis(1))
    }

    fn run_with_reopen(kind: RunKind) {
        let dir = tempfile::tempdir().unwrap();
        let mut kept = create_run(recipe(kind), 7, dir.path().join("kept")).unwrap();
        let resumed_dir = dir.path().join("resumed");
        let mut resumed = create_run(recipe(kind), 7, resumed_dir.clone()).unwrap();
        for _ in 0..6 {
            let x = kept.ask(2).unwrap();
            assert_eq!(resumed.ask(2).unwrap(), x);
            let y = objective(&x);
            kept.tell(&x.view(), &y.view(), None).unwrap();
            resumed.tell(&x.view(), &y.view(), None).unwrap();
            drop(resumed);
            resumed = Optimizer::open(&resumed_dir).unwrap();
            assert_eq!(resumed.x_obs(), kept.x_obs());
            assert_eq!(resumed.incumbent_x(), kept.incumbent_x());
            assert_eq!(resumed.tr_length(), kept.tr_length());
        }
    }

    #[test]
    fn a_reopened_run_matches_a_single_instance_in_every_mode() {
        for kind in [RunKind::Zero, RunKind::Lhd] {
            run_with_reopen(kind);
        }
    }

    fn shrinking_recipe() -> RunRecipe {
        let overrides = ConfigOverrides {
            length_init: Some(0.015),
            length_min: Some(0.01),
            length_max: Some(0.02),
            ..Default::default()
        };
        RunRecipe::new(
            RunKind::Zero,
            array![[0.0, 1.0], [0.0, 2.0]],
            None,
            Some(3),
            Some(overrides),
        )
    }

    fn told_best_survives_restarts_and_reopen(work_dir: PathBuf, recipe: RunRecipe) {
        let mut run = create_run(recipe, 3, work_dir.clone()).unwrap();
        let mut best = f64::NEG_INFINITY;
        let mut best_x = None;
        for _ in 0..200 {
            let x = run.ask(1).unwrap();
            let y = objective(&x);
            if y[[0, 0]] > best {
                best = y[[0, 0]];
                best_x = Some(x.row(0).to_owned());
            }
            run.tell(&x.view(), &y.view(), None).unwrap();
        }
        assert!(run.restart_generation() > 0, "the run never restarted");
        assert_eq!(run.best_y().unwrap()[0], best);
        drop(run);
        let reopened = Optimizer::open(&work_dir).unwrap();
        assert_eq!(reopened.best_y().unwrap()[0], best);
        let told_x = best_x.unwrap();
        let kept_x = reopened.best_x().unwrap();
        assert!((kept_x - told_x).iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn the_best_observation_survives_restarts_and_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        told_best_survives_restarts_and_reopen(dir.path().join("w"), shrinking_recipe());
    }

    #[test]
    fn a_snapshot_written_before_the_best_record_existed_recovers_it_from_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w");
        let mut run = create_run(recipe(RunKind::Zero), 7, path.clone()).unwrap();
        let x = run.ask(3).unwrap();
        let y = objective(&x);
        run.tell(&x.view(), &y.view(), None).unwrap();
        let best = run.best_y().unwrap()[0];
        run.best_observation = None;
        run.write_snapshot(false).unwrap();
        drop(run);
        assert_eq!(Optimizer::open(&path).unwrap().best_y().unwrap()[0], best);
    }

    #[test]
    fn a_used_work_dir_and_an_empty_work_dir_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("w");
        let run = create_run(recipe(RunKind::Zero), 1, path.clone()).unwrap();
        drop(run);
        let used = create_run(recipe(RunKind::Zero), 1, path).err().unwrap();
        assert!(used.to_string().contains("Optimizer.open"), "{used}");
        let empty = Optimizer::open(dir.path()).err().unwrap();
        assert!(empty.to_string().contains("create_optimizer"), "{empty}");
    }

    #[test]
    fn a_recipe_round_trips_through_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("recipe.json");
        bpann_write_json(&path, &recipe(RunKind::Enn)).unwrap();
        let loaded: RunRecipe = bpann_read_json(&path).unwrap().unwrap();
        assert_eq!(loaded.kind, RunKind::Enn);
        assert_eq!(loaded.k, Some(3));
        assert_eq!(loaded.bounds, recipe(RunKind::Enn).bounds);
    }
}
