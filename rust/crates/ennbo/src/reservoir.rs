//! Algorithm R reservoir shared by the AUTO metric and the Python sampler.

use crate::error::ENNError;
use crate::numpy_pcg::NumpyPcg64;

pub struct RowReservoir {
    num_dim: usize,
    num_outputs: usize,
    capacity: usize,
    seed: u64,
    rng: NumpyPcg64,
    xs: Vec<f64>,
    ys: Vec<f64>,
    len: usize,
    num_seen: usize,
}

impl RowReservoir {
    pub fn new(
        capacity: usize,
        num_dim: usize,
        num_outputs: usize,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if capacity < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity must be >= 1, got {capacity}"
            )));
        }
        Ok(Self {
            num_dim,
            num_outputs,
            capacity,
            seed,
            rng: NumpyPcg64::from_seed(seed),
            xs: vec![0.0; capacity * num_dim],
            ys: vec![0.0; capacity * num_outputs],
            len: 0,
            num_seen: 0,
        })
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn num_seen(&self) -> usize {
        self.num_seen
    }

    pub fn x(&self) -> &[f64] {
        &self.xs[..self.len * self.num_dim]
    }

    pub fn y(&self) -> &[f64] {
        &self.ys[..self.len * self.num_outputs]
    }

    pub fn configure(&mut self, seed: u64, capacity: usize) -> Result<(), ENNError> {
        if capacity < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity must be >= 1, got {capacity}"
            )));
        }
        if capacity < self.len {
            return Err(ENNError::InvalidParameter(format!(
                "reservoir_capacity {capacity} is below the {} rows already stored",
                self.len
            )));
        }
        if seed != self.seed && self.num_seen > self.capacity {
            return Err(ENNError::InvalidParameter(
                "seed is fixed once the reservoir starts replacing rows".into(),
            ));
        }
        if seed != self.seed {
            self.seed = seed;
            self.rng = NumpyPcg64::from_seed(seed);
        }
        if capacity != self.capacity {
            let mut xs = vec![0.0; capacity * self.num_dim];
            let mut ys = vec![0.0; capacity * self.num_outputs];
            let n = self.len;
            xs[..n * self.num_dim].copy_from_slice(&self.xs[..n * self.num_dim]);
            ys[..n * self.num_outputs].copy_from_slice(&self.ys[..n * self.num_outputs]);
            self.xs = xs;
            self.ys = ys;
            self.capacity = capacity;
        }
        Ok(())
    }

    /// Algorithm R: fill the next slot until `capacity`, otherwise replace a
    /// uniform index in `0..num_seen` when that index still lies in the buffer.
    pub fn push_row(&mut self, x: &[f64], y: &[f64]) {
        if self.len < self.capacity {
            let i = self.len;
            self.xs[i * self.num_dim..(i + 1) * self.num_dim].copy_from_slice(x);
            self.ys[i * self.num_outputs..(i + 1) * self.num_outputs].copy_from_slice(y);
            self.len += 1;
        } else {
            let slot = self.rng.integers_high(self.num_seen as u64 + 1) as usize;
            if slot < self.capacity {
                self.xs[slot * self.num_dim..(slot + 1) * self.num_dim].copy_from_slice(x);
                self.ys[slot * self.num_outputs..(slot + 1) * self.num_outputs].copy_from_slice(y);
            }
        }
        self.num_seen += 1;
    }
}
