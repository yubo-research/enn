use super::Optimizer;
use crate::config::InitStrategy;
use crate::error::ENNError;

impl Optimizer {
    pub(crate) fn reinit_left(&self) -> Option<usize> {
        self.reinit_left
    }

    pub(crate) fn local_init_kind(&self) -> InitStrategy {
        self.local_init_kind
    }

    pub(crate) fn consume_reinit(&mut self, n: usize) {
        if let Some(left) = self.reinit_left {
            let left = left.saturating_sub(n);
            self.reinit_left = if left == 0 { None } else { Some(left) };
        }
    }

    pub(crate) fn begin_local_restart(&mut self) -> Result<(), ENNError> {
        if self.tr_state.is_morbo() {
            return Ok(());
        }
        if let Some(surrogate) = self.surrogate.as_mut() {
            surrogate.clear_observations()?;
        }
        self.fallback_x.clear();
        self.fallback_y.clear();
        self.incumbent_tracker.reset();
        self.incumbent_idx = None;
        self.incumbent_x_unit = None;
        self.incumbent_y_scalar = None;
        self.tr_state.restart_local();
        self.restart_generation += 1;
        self.reinit_left = if self.local_init_budget > 0 {
            Some(self.local_init_budget)
        } else {
            None
        };
        Ok(())
    }

    pub fn increment_restart_generation(&mut self) {
        self.restart_generation += 1;
    }

    pub fn restart_generation(&self) -> usize {
        self.restart_generation
    }
}
