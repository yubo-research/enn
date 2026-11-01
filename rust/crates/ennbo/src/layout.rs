use crate::error::ENNError;
use crate::metric_auto::MetricLearning;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct EnnLayout {
    work_dir: PathBuf,
    weighting: bool,
    input_fingerprint: Option<String>,
    semantic_only: bool,
    output: Option<Box<OutputIdentity>>,
}

#[derive(Debug, Clone, PartialEq)]
struct OutputIdentity {
    fingerprint: Option<String>,
    alpha: Option<f64>,
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
            input_fingerprint: None,
            semantic_only: false,
            output: None,
        })
    }

    pub fn with_weighting(mut self, weighting: bool) -> Self {
        self.weighting = weighting;
        self
    }

    pub fn with_input_fingerprint(mut self, input_fingerprint: Option<String>) -> Self {
        self.input_fingerprint = input_fingerprint;
        self
    }

    pub fn with_semantic_only(mut self, semantic_only: bool) -> Self {
        self.semantic_only = semantic_only;
        self
    }

    pub fn with_output_identity(
        mut self,
        output_fingerprint: Option<String>,
        output_alpha: Option<f64>,
    ) -> Self {
        self.output = (output_fingerprint.is_some() || output_alpha.is_some()).then(|| {
            Box::new(OutputIdentity {
                fingerprint: output_fingerprint,
                alpha: output_alpha,
            })
        });
        self
    }

    pub fn semantic_only(&self) -> bool {
        self.semantic_only
    }

    pub fn output_fingerprint(&self) -> Option<&str> {
        self.output.as_ref()?.fingerprint.as_deref()
    }

    pub fn output_alpha(&self) -> Option<f64> {
        self.output.as_ref()?.alpha
    }

    pub fn weighting(&self) -> bool {
        self.weighting
    }

    pub fn input_fingerprint(&self) -> Option<&str> {
        self.input_fingerprint.as_deref()
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
    fn input_fingerprint_defaults_to_none_and_survives_with_weighting() {
        let layout = EnnLayout::new("rows", false).unwrap();
        assert_eq!(layout.input_fingerprint(), None);
        let layout = layout
            .with_input_fingerprint(Some("abc".into()))
            .with_weighting(true);
        assert_eq!(layout.input_fingerprint(), Some("abc"));
    }

    #[test]
    fn empty_work_dir_is_rejected() {
        let err = EnnLayout::new("", false).unwrap_err();
        assert!(err.to_string().contains("work_dir"));
    }
}
