use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::trust_region_config::TrustRegionConfig;
use crate::error::ENNError;
use crate::morbo_trust_region::{MorboTrustRegion, Rescalarize};
use crate::trust_region::{TrustRegionError, TurboTrustRegion};

#[derive(Debug)]
pub enum TrustRegionState {
    Turbo(TurboTrustRegion),
    Morbo(Box<MorboTrustRegion>),
}

impl TrustRegionState {
    pub fn from_config(
        num_dim: usize,
        config: &TrustRegionConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, TrustRegionError> {
        match config {
            TrustRegionConfig::Turbo(cfg) => {
                Ok(TrustRegionState::Turbo(TurboTrustRegion::new(num_dim, *cfg)))
            }
            TrustRegionConfig::Morbo(settings) => Ok(TrustRegionState::Morbo(Box::new(
                MorboTrustRegion::new(num_dim, settings.clone(), rng)?,
            ))),
        }
    }

    pub fn is_morbo(&self) -> bool {
        matches!(self, TrustRegionState::Morbo(_))
    }

    pub fn morbo(&self) -> Option<&MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_ref()),
            _ => None,
        }
    }

    pub fn morbo_mut(&mut self) -> Option<&mut MorboTrustRegion> {
        match self {
            TrustRegionState::Morbo(m) => Some(m.as_mut()),
            _ => None,
        }
    }

    pub fn length(&self) -> f64 {
        match self {
            TrustRegionState::Turbo(t) => t.length(),
            TrustRegionState::Morbo(m) => m.as_ref().length(),
        }
    }

    pub fn set_num_arms(&mut self, num_arms: usize) {
        match self {
            TrustRegionState::Turbo(t) => t.set_num_arms(num_arms),
            TrustRegionState::Morbo(m) => m.as_mut().set_num_arms(num_arms),
        }
    }

    pub fn compute_bounds_1d(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        match self {
            TrustRegionState::Turbo(t) => t.compute_bounds_1d(x_center, lengthscales),
            TrustRegionState::Morbo(m) => m.as_ref().compute_bounds_1d(x_center, lengthscales),
        }
    }

    pub fn needs_restart(&self) -> bool {
        match self {
            TrustRegionState::Turbo(t) => t.needs_restart(),
            TrustRegionState::Morbo(m) => m.as_ref().needs_restart(),
        }
    }

    pub fn restart(&mut self, rng: Option<&mut dyn RngCore>) {
        match self {
            TrustRegionState::Turbo(t) => t.restart(),
            TrustRegionState::Morbo(m) => m.as_mut().restart(rng),
        }
    }

    pub fn resample_on_propose(&mut self, rng: &mut dyn RngCore) {
        if let TrustRegionState::Morbo(m) = self {
            if m.as_ref().rescalarize() == Rescalarize::OnPropose {
                m.as_mut().resample_weights(rng);
            }
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            TrustRegionState::Turbo(_) => 1,
            TrustRegionState::Morbo(m) => m.as_ref().num_metrics(),
        }
    }

    pub fn morbo_scalarize(
        &self,
        y: &ArrayView2<f64>,
        clip: bool,
    ) -> Result<Array1<f64>, TrustRegionError> {
        match self {
            TrustRegionState::Morbo(m) => m.as_ref().scalarize(y, clip),
            _ => Err(TrustRegionError::InvalidState(
                "scalarize requires Morbo trust region".to_string(),
            )),
        }
    }

    pub fn tell_update(
        &mut self,
        y_all: &ArrayView2<f64>,
        y_incumbent: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), ENNError> {
        match self {
            TrustRegionState::Turbo(t) => {
                if y_all.ncols() != 1 {
                    return Err(ENNError::InvalidParameter(format!(
                        "Turbo TR expects 1 objective column, got {}",
                        y_all.ncols()
                    )));
                }
                let y_1d = y_all.column(0).to_owned();
                t.update(&y_1d.view(), num_obs)
                    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
            }
            TrustRegionState::Morbo(m) => m
                .as_mut()
                .update(&y_all.view(), y_incumbent)
                .map_err(|e| ENNError::InvalidParameter(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::trust_region_config::TrustRegionConfig;
    use crate::morbo_trust_region::{MorboTRSettings, Rescalarize};
    use crate::trust_region::TRLengthConfig;

    #[test]
    fn trust_region_state_morbo_from_config() {
        let mut rng = StdRng::seed_from_u64(99);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnPropose,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(3, &cfg, &mut rng).unwrap();
        assert!(tr.is_morbo());
        assert_eq!(tr.num_metrics(), 2);
        tr.resample_on_propose(&mut rng);
    }

    #[test]
    fn morbo_on_propose_rescalarize_changes_weights_each_propose() {
        let mut rng = StdRng::seed_from_u64(11);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnPropose,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(2, &cfg, &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_on_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_on_propose(&mut rng);
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert!(!approx::relative_eq!(w0, w1, epsilon = 1e-12));
        assert!(!approx::relative_eq!(w1, w2, epsilon = 1e-12));
    }

    #[test]
    fn morbo_on_restart_rescalarize_unchanged_until_restart() {
        let mut rng = StdRng::seed_from_u64(12);
        let cfg = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: 2,
            alpha: 0.05,
            length: TRLengthConfig::default(),
            rescalarize: Rescalarize::OnRestart,
            noise_aware: false,
        });
        let mut tr = TrustRegionState::from_config(2, &cfg, &mut rng).unwrap();
        let w0 = tr.morbo_mut().expect("morbo").weights().to_owned();
        tr.resample_on_propose(&mut rng);
        let w1 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert!(approx::relative_eq!(w0, w1, epsilon = 1e-12));
        tr.restart(Some(&mut rng));
        let w2 = tr.morbo_mut().expect("morbo").weights().to_owned();
        assert!(!approx::relative_eq!(w1, w2, epsilon = 1e-12));
    }
}
