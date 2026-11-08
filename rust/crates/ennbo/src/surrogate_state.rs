use crate::fit_samples::ScaleSearch;
use crate::fitter::ENNFitter;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

pub(crate) const BULK_DISK_TELL_SKIP_FIT_ROWS: usize = 4_096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct AppendInput {
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
            rows,
            frozen,
            has_params,
            initial,
        } = input;
        let bulk = rows >= BULK_DISK_TELL_SKIP_FIT_ROWS;
        if initial {
            let act = frozen || !bulk;
            return Self {
                sync: act,
                search: act && !frozen,
                release: act,
            };
        }
        let skip_fit = bulk || has_params;
        Self {
            sync: !skip_fit || bulk,
            search: !frozen && !skip_fit,
            release: !skip_fit || bulk,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SurrogateState {
    pub(crate) params: Option<ENNParams>,
    pub(crate) fitter_params: Option<ENNParams>,
    #[serde(with = "crate::exact")]
    pub(crate) y_sum: ndarray::Array1<f64>,
    #[serde(with = "crate::exact")]
    pub(crate) y_sumsq: ndarray::Array1<f64>,
    pub(crate) num_obs: usize,
}

pub(crate) struct ScaleFit {
    pub(crate) search: ScaleSearch,
    pub(crate) fitter: ENNFitter,
    pub(crate) params: Option<ENNParams>,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(rows: usize, flags: u8) -> AppendInput {
        AppendInput {
            rows,
            frozen: flags & 2 != 0,
            has_params: flags & 4 != 0,
            initial: flags & 8 != 0,
        }
    }

    #[test]
    fn append_plan_truth_table() {
        let plan = |sync, search, release| AppendPlan {
            sync,
            search,
            release,
        };
        let bulk = BULK_DISK_TELL_SKIP_FIT_ROWS;
        const FROZEN: u8 = 2;
        const PARAMS: u8 = 4;
        const INITIAL: u8 = 8;
        for params in [0, PARAMS] {
            assert_eq!(
                AppendPlan::new(input(bulk, params | INITIAL)),
                plan(false, false, false)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, params)),
                plan(true, false, true)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, FROZEN | params | INITIAL)),
                plan(true, false, true)
            );
            assert_eq!(
                AppendPlan::new(input(bulk, FROZEN | params)),
                plan(true, false, true)
            );
        }
        assert_eq!(AppendPlan::new(input(bulk - 1, 0)), plan(true, true, true));
        assert_eq!(
            AppendPlan::new(input(bulk - 1, PARAMS)),
            plan(false, false, false)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, INITIAL)),
            plan(true, true, true)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, FROZEN | PARAMS | INITIAL)),
            plan(true, false, true)
        );
        assert_eq!(
            AppendPlan::new(input(bulk - 1, FROZEN | PARAMS)),
            plan(false, false, false)
        );
    }
}
