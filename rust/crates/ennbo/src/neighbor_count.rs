
use crate::error::ENNError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborCount(i32);

impl NeighborCount {
    pub fn new(k: i32) -> Result<Self, ENNError> {
        if k < 1 {
            return Err(ENNError::InvalidParameter(format!(
                "k (number of neighbors) must be > 0, got {k}"
            )));
        }
        Ok(Self(k))
    }

    pub fn get(self) -> i32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Default for NeighborCount {
    fn default() -> Self {
        Self(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_rejects_zero_and_negative() {
        assert!(NeighborCount::new(0).is_err());
        assert!(NeighborCount::new(-3).is_err());
        let k = NeighborCount::new(4).unwrap();
        assert_eq!(k.get(), 4);
        assert_eq!(k.as_usize(), 4);
        assert_eq!(NeighborCount::default().get(), 10);
    }
}
