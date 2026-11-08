use crate::categorical;
use crate::categorical_plan::{self, invalid, CatBlock, CatOptions};
use crate::categorical_values::CatValue;
pub use crate::categorical_values::InputValue;
use crate::error::ENNError;
use ndarray::{s, Array2, ArrayView2};

#[derive(Debug, Clone)]
pub struct CatEncoder {
    pub kinds: Vec<bool>,
    pub blocks: Vec<CatBlock>,
    pub options: CatOptions,
}

impl CatEncoder {
    pub fn new(
        kinds: Vec<bool>,
        cards: Option<Vec<Option<usize>>>,
        options: CatOptions,
    ) -> Result<Self, ENNError> {
        let blocks = categorical_plan::blocks(&kinds, cards.as_deref(), &options)?;
        Ok(Self {
            kinds,
            blocks,
            options,
        })
    }

    pub fn num_dim(&self) -> usize {
        self.blocks.iter().map(|b| b.width).sum::<usize>() + self.options.hash_dim.unwrap_or(0)
    }

    pub fn tied_dims(&self) -> Vec<Vec<usize>> {
        let mut cursor = 0;
        let mut groups = Vec::new();
        for block in &self.blocks {
            if block.style != "continuous" {
                groups.push((cursor..cursor + block.width).collect());
            }
            cursor += block.width;
        }
        if let Some(width) = self.options.hash_dim {
            groups.push((cursor..cursor + width).collect());
        }
        groups
    }

    pub fn encode(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        self.encode_values(x)
    }

    pub fn encode_i64(&self, x: &ArrayView2<i64>) -> Result<Array2<f64>, ENNError> {
        self.encode_values(x)
    }

    pub fn encode_mixed(&self, x: &ArrayView2<InputValue>) -> Result<Array2<f64>, ENNError> {
        self.encode_values(x)
    }

    fn encode_values<T: CatValue>(&self, x: &ArrayView2<T>) -> Result<Array2<f64>, ENNError> {
        if x.ncols() != self.kinds.len() {
            return Err(invalid("x shape does not match kinds"));
        }
        let mut out = Array2::zeros((x.nrows(), self.num_dim()));
        let mut cursor = 0;
        for block in &self.blocks {
            let values = x.column(block.source);
            if block.style == "continuous" {
                out.column_mut(cursor)
                    .assign(&values.mapv(CatValue::number));
            } else {
                let ids = ids(values.iter().copied())?;
                let encoded = if block.style == "onehot" {
                    categorical::one_hot(&ids, block.width)? * self.options.separation
                } else {
                    categorical::rademacher(
                        &ids,
                        block.variable,
                        block.width,
                        self.options.seed,
                        self.options.separation,
                    )?
                };
                out.slice_mut(s![.., cursor..cursor + block.width])
                    .assign(&encoded);
            }
            cursor += block.width;
        }
        if let Some(width) = self.options.hash_dim {
            let columns: Vec<_> = self
                .kinds
                .iter()
                .enumerate()
                .filter_map(|(j, c)| c.then_some(j))
                .collect();
            let mut categories = Array2::zeros((x.nrows(), columns.len()));
            for (j, &source) in columns.iter().enumerate() {
                let values = ids(x.column(source).iter().copied())?;
                for (i, v) in values.into_iter().enumerate() {
                    categories[[i, j]] = v;
                }
            }
            out.slice_mut(s![.., cursor..])
                .assign(&categorical::hash_columns(
                    &categories.view(),
                    width,
                    self.options.seed,
                )?);
        }
        Ok(out)
    }

    pub fn snap(&self, x: &ArrayView2<f64>, domains: &[Vec<i64>]) -> Result<Array2<i64>, ENNError> {
        if self.options.hash_dim.is_some() {
            return Err(invalid("hashed codes are not separated by variable"));
        }
        if x.ncols() != self.num_dim() {
            return Err(invalid("encoded shape does not match layout"));
        }
        if domains.len()
            != self
                .blocks
                .iter()
                .filter(|b| b.style != "continuous")
                .count()
        {
            return Err(invalid("domains has wrong length"));
        }
        let mut out = Array2::zeros((x.nrows(), domains.len()));
        let mut cursor = 0;
        let mut cat = 0;
        for block in &self.blocks {
            if block.style != "continuous" {
                let labels = categorical::snap(
                    &x.slice(s![.., cursor..cursor + block.width]),
                    &domains[cat],
                    block.variable,
                    self.options.seed,
                    self.options.separation,
                    block.style == "onehot",
                )?;
                for (i, label) in labels.into_iter().enumerate() {
                    out[[i, cat]] = label;
                }
                cat += 1;
            }
            cursor += block.width;
        }
        Ok(out)
    }
}

fn ids<T: CatValue>(values: impl Iterator<Item = T>) -> Result<Vec<i64>, ENNError> {
    values.map(CatValue::category).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::categorical_plan::CatEncoding;
    use ndarray::array;
    #[test]
    fn mixed_encoders_roundtrip_and_keep_tied_blocks() {
        let x = array![[0.3, 1.], [0.7, 0.]];
        for encoding in [
            CatEncoding::OneHot,
            CatEncoding::Rademacher,
            CatEncoding::Auto,
        ] {
            let encoder = CatEncoder::new(
                vec![false, true],
                Some(vec![None, Some(3)]),
                CatOptions {
                    encoding,
                    ..Default::default()
                },
            )
            .unwrap();
            let encoded = encoder.encode(&x.view()).unwrap();
            assert_eq!(encoded.column(0), x.column(0));
            assert_eq!(
                encoder.snap(&encoded.view(), &[vec![0, 1, 2]]).unwrap(),
                array![[1], [0]]
            );
            assert_eq!(encoder.tied_dims().len(), 1);
        }
        assert!(CatEncoder::new(vec![], None, CatOptions::default()).is_err());
    }
}
