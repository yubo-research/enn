use super::access::EnnIndexAccess;
use crate::error::ENNError;
use crate::o1::{MaintStats, OrderParams};

impl EnnIndexAccess<'_> {
    pub fn maint_stats(&self) -> Result<Option<MaintStats>, ENNError> {
        self.model.backend.with_o1(|s| Ok(s.maint_stats())).map(Option::flatten)
    }

    pub fn set_pread_pivots(&self, on: bool) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_pread_pivots(on))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no pivot reads".into()))
    }

    pub fn set_lean(&self, lean: u32) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_lean(lean))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no split lean".into()))
    }

    pub fn set_staging(&self, above: u32, work: u32) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_staging(above, work))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no staged rebuilds".into()))
    }

    pub fn set_row_order(&self, params: OrderParams) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_row_order(params))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no row order".into()))
    }

    pub fn set_par_rebuild(&self, above: u32) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_par_rebuild(above))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no subtree rebuilds".into()))
    }

    pub fn set_trees(&self, n: usize) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_trees(n))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no trees".into()))
    }

    pub fn set_local_rebuild(&self, kib: u32, pread: u8) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_local_rebuild(kib, pread))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no subtree rebuilds".into()))
    }

    pub fn set_rerank(&self, rerank: u32) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_rerank(rerank))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no row codes".into()))
    }

    pub fn set_alpha(&self, alpha: f64) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_alpha(alpha))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no subtree rebuilds".into()))
    }

    pub fn set_segments(&self, params: crate::o1::SegParams) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_segments(params))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no segments".into()))
    }

    pub fn order_rows(&self) -> Result<usize, ENNError> {
        self.model
            .backend
            .with_o1(|s| s.order_rows())?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no row order".into()))
    }
}
