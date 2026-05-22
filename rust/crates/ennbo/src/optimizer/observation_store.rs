use ndarray::{Array1, Array2};
use std::cell::RefCell;

pub(crate) struct ObservationStore {
    x_obs: Vec<Array1<f64>>,
    y_obs: Vec<Array1<f64>>,
    cached_x: RefCell<Option<Array2<f64>>>,
    cached_y: RefCell<Option<Array2<f64>>>,
}

impl ObservationStore {
    pub(crate) fn new() -> Self {
        Self {
            x_obs: Vec::new(),
            y_obs: Vec::new(),
            cached_x: RefCell::new(None),
            cached_y: RefCell::new(None),
        }
    }

    pub(crate) fn invalidate_cache(&self) {
        *self.cached_x.borrow_mut() = None;
        *self.cached_y.borrow_mut() = None;
    }

    pub(crate) fn push(&mut self, x: Array1<f64>, y: Array1<f64>) {
        self.invalidate_cache();
        self.x_obs.push(x);
        self.y_obs.push(y);
    }

    pub(crate) fn len(&self) -> usize {
        self.x_obs.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.x_obs.is_empty()
    }

    pub(crate) fn x_obs_array(&self) -> Option<Array2<f64>> {
        if self.x_obs.is_empty() {
            return None;
        }
        let mut cache = self.cached_x.borrow_mut();
        if let Some(ref cached) = *cache {
            return Some(cached.clone());
        }
        let arr = Self::build_array2(&self.x_obs);
        *cache = Some(arr.clone());
        Some(arr)
    }

    pub(crate) fn y_obs_array(&self) -> Option<Array2<f64>> {
        if self.y_obs.is_empty() {
            return None;
        }
        let mut cache = self.cached_y.borrow_mut();
        if let Some(ref cached) = *cache {
            return Some(cached.clone());
        }
        let arr = Self::build_array2(&self.y_obs);
        *cache = Some(arr.clone());
        Some(arr)
    }

    pub(crate) fn build_array2(vecs: &[Array1<f64>]) -> Array2<f64> {
        let n = vecs.len();
        let d = vecs[0].len();
        let mut result = Array2::zeros((n, d));
        for (i, v) in vecs.iter().enumerate() {
            for j in 0..d {
                result[[i, j]] = v[j];
            }
        }
        result
    }

    #[allow(dead_code)]
    pub(crate) fn replace(&mut self, new_x: Vec<Array1<f64>>, new_y: Vec<Array1<f64>>) {
        self.invalidate_cache();
        self.x_obs = new_x;
        self.y_obs = new_y;
    }

    pub(crate) fn x_at(&self, idx: usize) -> &Array1<f64> {
        &self.x_obs[idx]
    }

    pub(crate) fn y_at(&self, idx: usize) -> &Array1<f64> {
        &self.y_obs[idx]
    }

    pub(crate) fn rows_as_array2(&self, start: usize, end: usize) -> Option<(Array2<f64>, Array2<f64>)> {
        if start >= end || end > self.x_obs.len() {
            return None;
        }
        let n = end - start;
        let d = self.x_obs[start].len();
        let m = self.y_obs[start].len();
        let mut x = Array2::zeros((n, d));
        let mut y = Array2::zeros((n, m));
        for i in start..end {
            let r = i - start;
            for j in 0..d {
                x[[r, j]] = self.x_obs[i][j];
            }
            for j in 0..m {
                y[[r, j]] = self.y_obs[i][j];
            }
        }
        Some((x, y))
    }
}
