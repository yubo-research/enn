use super::lsm_key::{Frame, Proj, MAX_AXES};

#[derive(Debug, Clone)]
pub(crate) struct Keying {
    pub proj: Proj,
    pub raw: bool,
}

pub(crate) fn axis_keys(d: usize) -> bool {
    d <= MAX_AXES && std::env::var("ENN_LSM_PROJ").is_ok_and(|v| v == "axis")
}

impl Keying {
    pub(crate) fn open(d: usize, seed: u64, axis: bool, raw: bool) -> Self {
        let proj = match axis {
            true => Proj::axes(d),
            false => Proj::new(d, seed),
        };
        Self { proj, raw }
    }

    pub(crate) fn klen(&self) -> usize {
        self.proj.m
    }

    pub(crate) fn bdim(&self) -> usize {
        match self.raw {
            true => self.proj.d,
            false => self.proj.m,
        }
    }

    pub(crate) fn key(&self, x: &[f64], frame: &Frame, z: &mut [f64], key: &mut [u64]) {
        self.proj.key(x, frame, z, key);
    }

    pub(crate) fn box_coords<'a>(&self, x: &'a [f64], z: &'a [f64]) -> &'a [f64] {
        match self.raw {
            true => x,
            false => z,
        }
    }
}
