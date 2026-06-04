//! Dense row-major storage for ENN observation columns.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::error::ENNError;

#[derive(Debug)]
pub(crate) struct RowStorage {
    buf: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

impl RowStorage {
    pub(crate) fn from_array2(a: Array2<f64>) -> Self {
        let (nrows, ncols) = a.dim();
        let a = a.as_standard_layout().into_owned();
        let mut buf = Vec::with_capacity(nrows.saturating_mul(ncols));
        buf.extend(a.iter());
        let cur_elems = nrows.saturating_mul(ncols);
        buf.reserve(cur_elems.max(ncols.saturating_mul(4096)));
        Self { buf, nrows, ncols }
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn view(&self) -> ndarray::ArrayView2<'_, f64> {
        ndarray::ArrayView2::from_shape((self.nrows, self.ncols), &self.buf[..self.nrows * self.ncols])
            .expect("row-major view")
    }

    pub(crate) fn gather_rows(&self, indices: &[usize]) -> Array2<f64> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
            out.row_mut(new_i).assign(&self.view().row(old_i));
        }
        out
    }

    pub(crate) fn row_vec(&self, i: usize) -> Array1<f64> {
        self.view().row(i).to_owned()
    }

    pub(crate) fn push_rows(&mut self, extra: &ArrayView2<f64>) -> Result<(), ENNError> {
        if extra.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![extra.nrows(), extra.ncols()],
            });
        }
        let n1 = extra.nrows();
        if n1 == 0 {
            return Ok(());
        }
        let cur_elems = self.nrows * self.ncols;
        let add_elems = n1 * self.ncols;
        let new_elems = cur_elems + add_elems;
        if self.buf.capacity() < new_elems {
            let growth = new_elems.saturating_sub(self.buf.capacity());
            let slack = cur_elems.max(self.ncols.saturating_mul(4096));
            self.buf.reserve(growth + slack);
        }
        for row in extra.axis_iter(Axis(0)) {
            self.buf.extend(row.iter().copied());
        }
        self.nrows += n1;
        Ok(())
    }
}
