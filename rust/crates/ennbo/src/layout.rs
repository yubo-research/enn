//! Index, storage, and metric combinations that can be constructed.

use std::path::{Path, PathBuf};

use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::metric_auto::MetricLearning;

/// Legal placement of an ENN model.
///
/// Disk storage exists only with `BpAnnDisk`. `Auto` exists only on disk with
/// `scale_x` false. A flat index on disk, and Auto on any other layout, have
/// no variant.
#[derive(Debug, Clone, PartialEq)]
pub enum EnnLayout {
    /// In-memory index. `driver` may be flat or B+ANN.
    Memory {
        /// Neighbor index.
        driver: IndexDriver,
        /// Divide `x` by per-dimension scales.
        scale_x: bool,
    },
    /// On-disk B+ANN with a fixed metric.
    Disk {
        /// Directory for rows and the index.
        work_dir: PathBuf,
        /// Divide `x` by per-dimension scales.
        scale_x: bool,
    },
    /// On-disk B+ANN with Auto metric learning. `scale_x` is false.
    DiskAuto {
        /// Directory for rows and the index.
        work_dir: PathBuf,
    },
}

/// Fields the model constructor reads off a layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenedLayout {
    /// Divide `x` by per-dimension scales.
    pub scale_x: bool,
    /// Neighbor index.
    pub driver: IndexDriver,
    /// Where rows are stored.
    pub storage: EnnStorage,
    /// Disk directory, when storage is disk.
    pub work_dir: Option<PathBuf>,
}

fn auto_rejected() -> ENNError {
    ENNError::InvalidParameter(
        "metric_learning=Auto requires IndexDriver::BpAnnDisk, disk storage, and scale_x=false"
            .into(),
    )
}

fn memory_layout(
    driver: IndexDriver,
    scale_x: bool,
    metric: MetricLearning,
) -> Result<EnnLayout, ENNError> {
    if metric == MetricLearning::Auto {
        return Err(auto_rejected());
    }
    Ok(EnnLayout::Memory { driver, scale_x })
}

fn require_disk_dir(work_dir: Option<PathBuf>) -> Result<PathBuf, ENNError> {
    work_dir.or_else(EnnStorage::work_dir_from_env).ok_or_else(|| {
        ENNError::InvalidParameter(
            "Disk storage requires work_dir or ENN_WORK_DIR".into(),
        )
    })
}

fn disk_layout(
    driver: IndexDriver,
    work_dir: Option<PathBuf>,
    scale_x: bool,
    metric: MetricLearning,
) -> Result<EnnLayout, ENNError> {
    if driver != IndexDriver::BpAnnDisk {
        return Err(ENNError::InvalidParameter(
            "Disk storage requires IndexDriver::BpAnnDisk".into(),
        ));
    }
    let work_dir = require_disk_dir(work_dir)?;
    if metric != MetricLearning::Auto {
        return Ok(EnnLayout::Disk { work_dir, scale_x });
    }
    if scale_x {
        return Err(auto_rejected());
    }
    Ok(EnnLayout::DiskAuto { work_dir })
}

impl EnnLayout {
    /// In-memory layout. Metric learning is off.
    pub fn memory(driver: IndexDriver, scale_x: bool) -> Self {
        Self::Memory { driver, scale_x }
    }

    /// On-disk B+ANN with a fixed metric.
    pub fn disk(work_dir: PathBuf, scale_x: bool) -> Self {
        Self::Disk { work_dir, scale_x }
    }

    /// Whether `x` is divided by per-dimension scales.
    pub fn scale_x(&self) -> bool {
        match self {
            Self::Memory { scale_x, .. } | Self::Disk { scale_x, .. } => *scale_x,
            Self::DiskAuto { .. } => false,
        }
    }

    /// Neighbor index used by this layout.
    pub fn index_driver(&self) -> IndexDriver {
        match self {
            Self::Memory { driver, .. } => *driver,
            Self::Disk { .. } | Self::DiskAuto { .. } => IndexDriver::BpAnnDisk,
        }
    }

    /// Row storage used by this layout.
    pub fn storage(&self) -> EnnStorage {
        match self {
            Self::Memory { .. } => EnnStorage::InMemory,
            Self::Disk { .. } | Self::DiskAuto { .. } => EnnStorage::Disk,
        }
    }

    /// Disk directory, if this layout stores rows on disk.
    pub fn work_dir(&self) -> Option<&Path> {
        match self {
            Self::Memory { .. } => None,
            Self::Disk { work_dir, .. } | Self::DiskAuto { work_dir } => Some(work_dir),
        }
    }

    /// Metric policy implied by this layout.
    pub fn metric_learning(&self) -> MetricLearning {
        match self {
            Self::DiskAuto { .. } => MetricLearning::Auto,
            Self::Memory { .. } | Self::Disk { .. } => MetricLearning::None,
        }
    }

    /// Split the layout into the fields the model backend expects.
    pub fn open(&self) -> OpenedLayout {
        OpenedLayout {
            scale_x: self.scale_x(),
            driver: self.index_driver(),
            storage: self.storage(),
            work_dir: self.work_dir().map(Path::to_path_buf),
        }
    }

    /// Build a layout from the pieces a caller can set independently.
    ///
    /// `work_dir` without `Some(Disk)` is an error. It does not select disk.
    pub fn try_from_parts(
        driver: IndexDriver,
        storage: Option<EnnStorage>,
        work_dir: Option<PathBuf>,
        scale_x: bool,
        metric: MetricLearning,
    ) -> Result<Self, ENNError> {
        if work_dir.is_some() && !matches!(storage, Some(EnnStorage::Disk)) {
            return Err(ENNError::InvalidParameter(
                "work_dir does not select storage; pass disk storage explicitly".into(),
            ));
        }
        match storage {
            Some(EnnStorage::Disk) => disk_layout(driver, work_dir, scale_x, metric),
            Some(EnnStorage::InMemory) | None => memory_layout(driver, scale_x, metric),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_disk_and_auto_on_flat_are_rejected() {
        let dir = PathBuf::from("/tmp/enn_layout");
        let flat_disk = EnnLayout::try_from_parts(
            IndexDriver::Flat,
            Some(EnnStorage::Disk),
            Some(dir.clone()),
            false,
            MetricLearning::None,
        );
        assert!(flat_disk.unwrap_err().to_string().contains("BpAnnDisk"));
        let auto_flat = EnnLayout::try_from_parts(
            IndexDriver::Flat,
            None,
            None,
            false,
            MetricLearning::Auto,
        );
        assert!(auto_flat.unwrap_err().to_string().contains("scale_x=false"));
        let auto_scaled = EnnLayout::try_from_parts(
            IndexDriver::BpAnnDisk,
            Some(EnnStorage::Disk),
            Some(dir),
            true,
            MetricLearning::Auto,
        );
        assert!(auto_scaled.unwrap_err().to_string().contains("scale_x=false"));
    }

    #[test]
    fn work_dir_alone_does_not_select_disk() {
        let err = EnnLayout::try_from_parts(
            IndexDriver::BpAnnDisk,
            None,
            Some(PathBuf::from("/tmp/enn_layout")),
            false,
            MetricLearning::None,
        );
        assert!(err.unwrap_err().to_string().contains("does not select storage"));
    }

    #[test]
    fn disk_auto_opens_as_bpann_disk() {
        let layout = EnnLayout::try_from_parts(
            IndexDriver::BpAnnDisk,
            Some(EnnStorage::Disk),
            Some(PathBuf::from("/tmp/enn_layout")),
            false,
            MetricLearning::Auto,
        )
        .unwrap();
        let opened = layout.open();
        assert!(!opened.scale_x);
        assert_eq!(opened.driver, IndexDriver::BpAnnDisk);
        assert_eq!(opened.storage, EnnStorage::Disk);
        assert_eq!(layout.metric_learning(), MetricLearning::Auto);
        let fixed = EnnLayout::disk(PathBuf::from("/tmp/enn_layout"), true);
        assert!(fixed.scale_x());
        assert!(EnnLayout::memory(IndexDriver::Flat, false).work_dir().is_none());
    }
}
