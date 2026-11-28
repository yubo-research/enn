use super::index_kind::O1Index;
use super::row_order::{OrderParams, Rows};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

impl O1Index {
    pub(crate) fn rows<'a>(&'a self, x: &'a MmapColumnStore) -> Rows<'a> {
        match self {
            Self::Lsh(_) | Self::Lsm(_) | Self::Hlsm(_) | Self::Graph(_) => Rows::plain(x),
            Self::Forest(f) => Rows {
                x,
                order: f.order.as_ref(),
                near: None,
            },
        }
    }

    pub(crate) fn set_row_order(&mut self, params: OrderParams) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) | Self::Lsm(_) | Self::Hlsm(_) | Self::Graph(_) => Err(ENNError::InvalidParameter(
                "this O1 index has no row order".into(),
            )),
            Self::Forest(f) => f.set_row_order(params),
        }
    }

    pub(crate) fn set_par_rebuild(&mut self, above: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) | Self::Lsm(_) | Self::Hlsm(_) | Self::Graph(_) => Err(ENNError::InvalidParameter(
                "this O1 index has no subtree rebuilds".into(),
            )),
            Self::Forest(f) => {
                f.set_par_rebuild(above);
                Ok(())
            }
        }
    }

    pub(crate) fn set_rerank(&mut self, rerank: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) | Self::Lsm(_) | Self::Hlsm(_) | Self::Graph(_) => Err(ENNError::InvalidParameter(
                "this O1 index has no row codes".into(),
            )),
            Self::Forest(f) => {
                f.set_rerank(rerank);
                Ok(())
            }
        }
    }

    pub(crate) fn order_now(&mut self, rows: &MmapColumnStore) -> Result<usize, ENNError> {
        match self {
            Self::Lsh(_) | Self::Lsm(_) | Self::Hlsm(_) | Self::Graph(_) => Err(ENNError::InvalidParameter(
                "this O1 index has no row order".into(),
            )),
            Self::Forest(f) => f.order_now(rows),
        }
    }
}
