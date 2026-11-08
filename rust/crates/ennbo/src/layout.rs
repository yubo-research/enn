use crate::error::ENNError;
use crate::metric_auto::MetricLearning;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct EnnLayout {
    work_dir: PathBuf,
    weighting: bool,
    aggregate_duplicates: bool,
}

impl EnnLayout {
    pub fn new(work_dir: impl Into<PathBuf>, weighting: bool) -> Result<Self, ENNError> {
        let work_dir = work_dir.into();
        if work_dir.as_os_str().is_empty() {
            return Err(ENNError::InvalidParameter(
                "work_dir must not be empty".into(),
            ));
        }
        Ok(Self {
            work_dir,
            weighting,
            aggregate_duplicates: false,
        })
    }

    pub fn with_weighting(mut self, weighting: bool) -> Self {
        self.weighting = weighting;
        self
    }

    pub fn with_aggregate_duplicates(mut self, aggregate_duplicates: bool) -> Self {
        self.aggregate_duplicates = aggregate_duplicates;
        self
    }

    pub fn aggregate_duplicates(&self) -> bool {
        self.aggregate_duplicates
    }

    pub fn weighting(&self) -> bool {
        self.weighting
    }

    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    pub fn metric_learning(&self) -> MetricLearning {
        if self.weighting {
            MetricLearning::Auto
        } else {
            MetricLearning::None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighting_selects_metric_learning() {
        for weighting in [false, true] {
            let layout = EnnLayout::new("rows", weighting).unwrap();
            assert_eq!(layout.weighting(), weighting);
            assert_eq!(layout.metric_learning() == MetricLearning::Auto, weighting);
            assert_eq!(layout.work_dir(), Path::new("rows"));
        }
    }

    #[test]
    fn with_weighting_keeps_the_work_dir() {
        let layout = EnnLayout::new("rows", false).unwrap().with_weighting(true);
        assert!(layout.weighting());
        assert_eq!(layout.work_dir(), Path::new("rows"));
    }

    #[test]
    fn aggregate_duplicates_is_off_by_default_and_keeps_the_other_settings() {
        let layout = EnnLayout::new("rows", true).unwrap();
        assert!(!layout.aggregate_duplicates());
        let on = layout.with_aggregate_duplicates(true);
        assert!(on.aggregate_duplicates() && on.weighting());
        assert_eq!(on.work_dir(), Path::new("rows"));
    }

    #[test]
    fn empty_work_dir_is_rejected() {
        let err = EnnLayout::new("", false).unwrap_err();
        assert!(err.to_string().contains("work_dir"));
    }
}
