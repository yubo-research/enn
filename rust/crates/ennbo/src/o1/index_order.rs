use super::index_kind::O1Index;
use super::row_order::{OrderParams, Rows};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

impl O1Index {
    pub(crate) fn rows<'a>(&'a self, x: &'a MmapColumnStore) -> Rows<'a> {
        match self {
            Self::Lsh(_) => Rows::plain(x),
            Self::Forest(f) => Rows { x, order: f.order.as_ref() },
        }
    }

    pub(crate) fn set_row_order(&mut self, params: OrderParams) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no row order".into())),
            Self::Forest(f) => f.set_row_order(params),
        }
    }

    pub(crate) fn set_par_rebuild(&mut self, above: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no subtree rebuilds".into())),
            Self::Forest(f) => {
                f.set_par_rebuild(above);
                Ok(())
            }
        }
    }

    pub(crate) fn set_alpha(&mut self, alpha: f64) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no subtree rebuilds".into())),
            Self::Forest(f) => {
                f.set_alpha(alpha);
                Ok(())
            }
        }
    }

    pub(crate) fn set_rerank(&mut self, rerank: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no row codes".into())),
            Self::Forest(f) => {
                f.set_rerank(rerank);
                Ok(())
            }
        }
    }

    pub(crate) fn set_local_rebuild(&mut self, kib: u32, pread: u8) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no subtree rebuilds".into())),
            Self::Forest(f) => {
                f.set_local_rebuild(kib, pread);
                Ok(())
            }
        }
    }

    pub(crate) fn set_segments(&mut self, params: super::SegParams) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no segments".into())),
            Self::Forest(f) => {
                f.set_segments(params);
                Ok(())
            }
        }
    }

    pub(crate) fn order_now(&mut self, rows: &MmapColumnStore) -> Result<usize, ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no row order".into())),
            Self::Forest(f) => f.order_now(rows),
        }
    }
}
