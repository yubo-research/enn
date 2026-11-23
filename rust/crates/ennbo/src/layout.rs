use crate::error::ENNError;
use crate::metric_auto::MetricLearning;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StoreType {
    #[default]
    #[serde(rename = "OLOGN")]
    OLogN,
    #[serde(rename = "O1")]
    O1,
}

impl StoreType {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "OLOGN" | "ologn" | "OLogN" => Some(Self::OLogN),
            "O1" | "o1" => Some(Self::O1),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::OLogN => "OLOGN",
            Self::O1 => "O1",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnnLayout {
    work_dir: PathBuf,
    weighting: bool,
    store: StoreType,
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
            store: StoreType::OLogN,
        })
    }

    pub fn with_weighting(mut self, weighting: bool) -> Self {
        self.weighting = weighting;
        self
    }

    pub fn with_store(mut self, store: StoreType) -> Self {
        self.store = store;
        self
    }

    pub fn store(&self) -> StoreType {
        self.store
    }

    pub fn weighting(&self) -> bool {
        self.weighting || self.store == StoreType::O1
    }

    pub fn work_dir(&self) -> &Path {
        &self.work_dir
    }

    pub fn metric_learning(&self) -> MetricLearning {
        if self.weighting() {
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
    fn o1_store_always_learns_its_metric() {
        let layout = EnnLayout::new("rows", false).unwrap().with_store(StoreType::O1);
        assert_eq!(layout.store(), StoreType::O1);
        assert!(layout.weighting());
        assert_eq!(layout.metric_learning(), MetricLearning::Auto);
        for t in [StoreType::OLogN, StoreType::O1] {
            assert_eq!(StoreType::parse(t.name()), Some(t));
        }
        assert_eq!(StoreType::parse("nope"), None);
    }

    #[test]
    fn empty_work_dir_is_rejected() {
        let err = EnnLayout::new("", false).unwrap_err();
        assert!(err.to_string().contains("work_dir"));
    }
}
