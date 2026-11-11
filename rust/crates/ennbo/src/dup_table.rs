use std::sync::Arc;

use ennbo_bpann::row_groups::RowGroups;

use crate::backend::EnnBackend;

pub(crate) struct DupTable {
    groups: Arc<RowGroups>,
    num_metrics: usize,
    count: Vec<u32>,
    mean: Vec<f64>,
    sum_yvar: Vec<f64>,
}

impl DupTable {
    pub(crate) fn new(groups: Arc<RowGroups>, num_metrics: usize) -> Self {
        Self {
            groups,
            num_metrics,
            count: Vec::new(),
            mean: Vec::new(),
            sum_yvar: Vec::new(),
        }
    }

    pub(crate) fn attach(backend: &EnnBackend, num_metrics: usize) -> Option<Self> {
        backend
            .row_groups()
            .map(|groups| Self::new(groups, num_metrics))
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.count.len()
    }

    pub(crate) fn observe(&mut self, obs: usize, y: &[f64], yvar: Option<&[f64]>) {
        let row = self.groups.row_of(obs).expect("observation is registered");
        let m = self.num_metrics;
        if row == self.count.len() {
            self.count.push(1);
            self.mean.extend_from_slice(y);
            if let Some(yvar) = yvar {
                self.sum_yvar.extend_from_slice(yvar);
            }
            return;
        }
        self.count[row] += 1;
        let n = f64::from(self.count[row]);
        for (mean, &y) in self.mean[row * m..(row + 1) * m].iter_mut().zip(y) {
            *mean += (y - *mean) / n;
        }
        if let Some(yvar) = yvar {
            for (sum, &v) in self.sum_yvar[row * m..(row + 1) * m].iter_mut().zip(yvar) {
                *sum += v;
            }
        }
    }

    fn row(&self, rep: usize) -> usize {
        self.groups
            .row_of(rep)
            .expect("representative is registered")
    }

    pub(crate) fn count_at(&self, rep: usize) -> f64 {
        f64::from(self.count[self.row(rep)])
    }

    pub(crate) fn mean_at(&self, rep: usize) -> &[f64] {
        let (row, m) = (self.row(rep), self.num_metrics);
        &self.mean[row * m..(row + 1) * m]
    }

    pub(crate) fn mean_yvar_at(&self, rep: usize) -> Option<Vec<f64>> {
        let (row, m) = (self.row(rep), self.num_metrics);
        let sums = self.sum_yvar.get(row * m..(row + 1) * m)?;
        let n = f64::from(self.count[row]);
        Some(sums.iter().map(|s| s / n).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ennbo_bpann::mmap_store::MmapColumnStore;
    use ndarray::array;

    fn table_for(x: &ndarray::Array2<f64>, dir: &tempfile::TempDir) -> DupTable {
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), x.ncols(), None)
                .unwrap();
        store.mmap_append(&x.view()).unwrap();
        let groups = Arc::new(RowGroups::default());
        groups.register(&store, x.nrows()).unwrap();
        DupTable::new(groups, 2)
    }

    #[test]
    fn a_row_keeps_count_mean_and_yvar_sum_of_its_observations() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = table_for(&array![[1.0], [2.0], [1.0], [1.0]], &dir);
        let ys = [[1.0, 10.0], [5.0, 50.0], [2.0, 20.0], [6.0, 30.0]];
        let vs = [[0.1, 1.0], [0.5, 5.0], [0.2, 2.0], [0.3, 3.0]];
        for i in 0..4 {
            t.observe(i, &ys[i], Some(&vs[i]));
        }
        assert_eq!(t.num_rows(), 2);
        assert_eq!(t.count_at(0), 3.0);
        assert_eq!(t.count_at(1), 1.0);
        let mean = t.mean_at(0);
        assert!((mean[0] - 3.0).abs() < 1e-12 && (mean[1] - 20.0).abs() < 1e-12);
        let yvar = t.mean_yvar_at(0).unwrap();
        assert!((yvar[0] - 0.2).abs() < 1e-12 && (yvar[1] - 2.0).abs() < 1e-12);
        assert_eq!(t.mean_at(1), &[5.0, 50.0]);
        assert_eq!(t.mean_yvar_at(1).unwrap(), vec![0.5, 5.0]);
    }

    #[test]
    fn a_row_of_one_observation_reproduces_its_values_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = table_for(&array![[1.0]], &dir);
        t.observe(0, &[0.1 + 0.2, 1.0 / 3.0], Some(&[0.7, 1e-9]));
        assert_eq!(t.mean_at(0), &[0.1 + 0.2, 1.0 / 3.0]);
        assert_eq!(t.mean_yvar_at(0).unwrap(), vec![0.7, 1e-9]);
        assert_eq!(t.count_at(0), 1.0);
    }

    #[test]
    fn without_yvar_no_yvar_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut t = table_for(&array![[1.0], [1.0]], &dir);
        t.observe(0, &[1.0, 2.0], None);
        t.observe(1, &[3.0, 4.0], None);
        assert!(t.mean_yvar_at(0).is_none());
        assert_eq!(t.mean_at(0), &[2.0, 3.0]);
    }
}
