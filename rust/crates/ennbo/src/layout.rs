use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::metric_auto::MetricLearning;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub enum EnnLayout {
    Memory { scale_x: bool },
    Disk { work_dir: PathBuf },
    DiskAuto { work_dir: PathBuf },
}

impl EnnLayout {
    pub fn flat(weighting: bool) -> Self {
        Self::Memory { scale_x: weighting }
    }

    pub fn disk(work_dir: PathBuf, weighting: bool) -> Self {
        if weighting {
            Self::DiskAuto { work_dir }
        } else {
            Self::Disk { work_dir }
        }
    }

    pub fn new(
        driver: IndexDriver,
        work_dir: Option<PathBuf>,
        weighting: bool,
    ) -> Result<Self, ENNError> {
        match (driver, work_dir) {
            (IndexDriver::Flat, None) => Ok(Self::flat(weighting)),
            (IndexDriver::Flat, Some(_)) => Err(ENNError::InvalidParameter(
                "FLAT stores in memory; work_dir requires BPANN_DISK".into(),
            )),
            (IndexDriver::BpAnnDisk, Some(dir)) if !dir.as_os_str().is_empty() => {
                Ok(Self::disk(dir, weighting))
            }
            (IndexDriver::BpAnnDisk, _) => Err(ENNError::InvalidParameter(
                "BPANN_DISK requires an explicit work_dir".into(),
            )),
        }
    }

    pub fn weighting(&self) -> bool {
        self.scale_x() || self.metric_learning() == MetricLearning::Auto
    }
    pub fn scale_x(&self) -> bool {
        matches!(self, Self::Memory { scale_x: true })
    }
    pub fn index_driver(&self) -> IndexDriver {
        match self {
            Self::Memory { .. } => IndexDriver::Flat,
            _ => IndexDriver::BpAnnDisk,
        }
    }
    pub fn storage(&self) -> EnnStorage {
        match self {
            Self::Memory { .. } => EnnStorage::InMemory,
            _ => EnnStorage::Disk,
        }
    }
    pub fn work_dir(&self) -> Option<&Path> {
        match self {
            Self::Memory { .. } => None,
            Self::Disk { work_dir } | Self::DiskAuto { work_dir } => Some(work_dir),
        }
    }
    pub fn metric_learning(&self) -> MetricLearning {
        match self {
            Self::DiskAuto { .. } => MetricLearning::Auto,
            _ => MetricLearning::None,
        }
    }


}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn placement_and_weighting_follow_driver() {
        for weighting in [false, true] {
            let flat = EnnLayout::new(IndexDriver::Flat, None, weighting).unwrap();
            let disk =
                EnnLayout::new(IndexDriver::BpAnnDisk, Some("rows".into()), weighting).unwrap();
            assert_eq!(flat.scale_x(), weighting);
            assert!(!disk.scale_x());
            assert_eq!(disk.metric_learning() == MetricLearning::Auto, weighting);
            assert_eq!(flat.storage(), EnnStorage::InMemory);
            assert_eq!(disk.storage(), EnnStorage::Disk);
            assert_eq!(flat.weighting(), weighting);
            assert_eq!(disk.weighting(), weighting);
        }
        assert!(EnnLayout::new(IndexDriver::Flat, Some("rows".into()), false).is_err());
        assert!(EnnLayout::new(IndexDriver::BpAnnDisk, None, false).is_err());
        assert!(EnnLayout::new(IndexDriver::BpAnnDisk, Some("".into()), false).is_err());
    }
}
