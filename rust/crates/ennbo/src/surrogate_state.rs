
use crate::calibration::AffineCalibrator;
use crate::fit_samples::ScaleSearch;
use crate::fitter::ENNFitter;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

pub(crate) const BULK_DISK_TELL_SKIP_FIT_ROWS: usize = 4_096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct AppendInput {
    pub(crate) on_disk: bool,
    pub(crate) rows: usize,
    pub(crate) frozen: bool,
    pub(crate) has_params: bool,
    pub(crate) initial: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AppendPlan {
    pub(crate) sync: bool,
    pub(crate) search: bool,
    pub(crate) release: bool,
}

impl AppendPlan {
    pub(crate) fn new(input: AppendInput) -> Self {
        let AppendInput {
            on_disk,
            rows,
            frozen,
            has_params,
            initial,
        } = input;
        let bulk = on_disk && rows >= BULK_DISK_TELL_SKIP_FIT_ROWS;
        if initial {
            let act = frozen || !bulk;
            return Self {
                sync: act && on_disk,
                search: act && !frozen,
                release: act && on_disk,
            };
        }
        let skip_fit = bulk || (on_disk && has_params);
        Self {
            sync: !skip_fit || bulk,
            search: !frozen && !skip_fit,
            release: !skip_fit || bulk,
        }
    }
}

pub(crate) struct ScaleFit {
    pub(crate) search: ScaleSearch,
    pub(crate) fitter: ENNFitter,
    pub(crate) params: Option<ENNParams>,
    pub(crate) calibrator: Option<AffineCalibrator>,
}

pub(crate) enum FitState {
    Unfitted,
    Frozen {
        model: EpistemicNearestNeighbors,
        params: ENNParams,
    },
    Searched {
        model: EpistemicNearestNeighbors,
        fit: Box<ScaleFit>,
    },
}

impl FitState {
    pub(crate) fn model(&self) -> Option<&EpistemicNearestNeighbors> {
        match self {
            Self::Unfitted => None,
            Self::Frozen { model, .. } | Self::Searched { model, .. } => Some(model),
        }
    }

    pub(crate) fn params(&self) -> Option<&ENNParams> {
        match self {
            Self::Unfitted => None,
            Self::Frozen { params, .. } => Some(params),
            Self::Searched { fit, .. } => fit.params.as_ref(),
        }
    }

    pub(crate) fn calibrator(&self) -> Option<&AffineCalibrator> {
        match self {
            Self::Searched { fit, .. } => fit.calibrator.as_ref(),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(rows: usize, flags: u8) -> AppendInput {
        AppendInput {
            on_disk: flags & 1 != 0,
            rows,
            frozen: flags & 2 != 0,
            has_params: flags & 4 != 0,
            initial: flags & 8 != 0,
        }
    }

    #[test]
    fn append_plan_truth_table() {
        let plan = |sync, search, release| AppendPlan { sync, search, release };
        let bulk = BULK_DISK_TELL_SKIP_FIT_ROWS;
        const DISK: u8 = 1;
        const FROZEN: u8 = 2;
        const PARAMS: u8 = 4;
        const INITIAL: u8 = 8;
        for flags in [0, PARAMS, FROZEN, FROZEN | PARAMS] {
            let search = flags & FROZEN == 0;
            assert_eq!(AppendPlan::new(input(1, flags)), plan(true, search, true));
            assert_eq!(AppendPlan::new(input(bulk, flags)), plan(true, search, true));
            assert_eq!(
                AppendPlan::new(input(1, flags | INITIAL)),
                plan(false, search, false)
            );
        }
        for params in [0, PARAMS] {
            assert_eq!(
                AppendPlan::new(input(bulk, DISK | params | INITIAL)),
                plan(false, false, false)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, DISK | params)),
                plan(true, false, true)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, DISK | FROZEN | params | INITIAL)),
                plan(true, false, true)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, DISK | FROZEN | params)),
                plan(true, false, true)
            );
        }
        assert_eq!(
            AppendPlan::new(input(bulk - 1, DISK)),
            plan(true, true, true)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, DISK | PARAMS)),
            plan(false, false, false)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, DISK | INITIAL)),
            plan(true, true, true)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, DISK | FROZEN | PARAMS | INITIAL)),
            plan(true, false, true)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, DISK | FROZEN | PARAMS)),
            plan(false, false, false)
        );
    }
}
