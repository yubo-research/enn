use std::sync::Arc;

use ndarray::{Array1, Array2, ShapeBuilder};
use serde::de::{DeserializeOwned, Error};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub(crate) trait Exact: Sized {
    type Repr: Serialize + DeserializeOwned;
    fn to_repr(&self) -> Self::Repr;
    fn from_repr(repr: Self::Repr) -> Result<Self, String>;
}

impl Exact for f64 {
    type Repr = u64;
    fn to_repr(&self) -> u64 {
        self.to_bits()
    }
    fn from_repr(repr: u64) -> Result<Self, String> {
        Ok(f64::from_bits(repr))
    }
}

impl Exact for Array1<f64> {
    type Repr = Vec<u64>;
    fn to_repr(&self) -> Vec<u64> {
        self.iter().map(|v| v.to_bits()).collect()
    }
    fn from_repr(repr: Vec<u64>) -> Result<Self, String> {
        Ok(repr.into_iter().map(f64::from_bits).collect())
    }
}

impl Exact for Array2<f64> {
    type Repr = (usize, usize, bool, Vec<u64>);
    fn to_repr(&self) -> Self::Repr {
        let column_major = !self.is_standard_layout() && self.t().is_standard_layout();
        let values = if column_major {
            self.t().iter().map(|v| v.to_bits()).collect()
        } else {
            self.iter().map(|v| v.to_bits()).collect()
        };
        (self.nrows(), self.ncols(), column_major, values)
    }
    fn from_repr((rows, cols, column_major, values): Self::Repr) -> Result<Self, String> {
        let values = values.into_iter().map(f64::from_bits).collect();
        let shape = (rows, cols).set_f(column_major);
        Array2::from_shape_vec(shape, values).map_err(|e| e.to_string())
    }
}

impl<T: Exact> Exact for Option<T> {
    type Repr = Option<T::Repr>;
    fn to_repr(&self) -> Self::Repr {
        self.as_ref().map(Exact::to_repr)
    }
    fn from_repr(repr: Self::Repr) -> Result<Self, String> {
        repr.map(T::from_repr).transpose()
    }
}

impl<T: Exact> Exact for Vec<T> {
    type Repr = Vec<T::Repr>;
    fn to_repr(&self) -> Self::Repr {
        self.iter().map(Exact::to_repr).collect()
    }
    fn from_repr(repr: Self::Repr) -> Result<Self, String> {
        repr.into_iter().map(T::from_repr).collect()
    }
}

impl Exact for (usize, f64) {
    type Repr = (usize, u64);
    fn to_repr(&self) -> Self::Repr {
        (self.0, self.1.to_bits())
    }
    fn from_repr(repr: Self::Repr) -> Result<Self, String> {
        Ok((repr.0, f64::from_bits(repr.1)))
    }
}

impl<T: Serialize + DeserializeOwned + Clone> Exact for Arc<T> {
    type Repr = T;
    fn to_repr(&self) -> T {
        self.as_ref().clone()
    }
    fn from_repr(repr: T) -> Result<Self, String> {
        Ok(Arc::new(repr))
    }
}

pub(crate) fn serialize<T: Exact, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    value.to_repr().serialize(serializer)
}

pub(crate) fn deserialize<'de, T: Exact, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    T::from_repr(T::Repr::deserialize(deserializer)?).map_err(D::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn round_trip<T: Exact>(value: &T) -> T {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("value.json");
        ennbo_bpann::bpann_write_json(&path, &value.to_repr()).unwrap();
        let repr = ennbo_bpann::bpann_read_json(&path).unwrap().unwrap();
        T::from_repr(repr).unwrap()
    }

    #[test]
    fn floats_survive_json_bit_for_bit_including_non_finite_values() {
        for value in [0.1 + 0.2, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1e-320] {
            assert_eq!(round_trip(&value).to_bits(), value.to_bits());
        }
        assert!(round_trip(&f64::NAN).is_nan());
    }

    #[test]
    fn containers_round_trip() {
        let matrix = array![[1.0, f64::INFINITY], [0.1, -2.0]];
        assert_eq!(round_trip(&matrix), matrix);
        let vector = Some(array![0.25, 0.5]);
        assert_eq!(round_trip(&vector), vector);
        let entries = vec![vec![(3_usize, 0.75)]];
        assert_eq!(round_trip(&entries), entries);
        let wrapped = Arc::new(5_u32);
        assert_eq!(round_trip(&wrapped), wrapped);
    }

    #[test]
    fn matrices_keep_their_memory_layout() {
        let column_major =
            Array2::from_shape_vec((2, 3).f(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let restored = round_trip(&column_major);
        assert_eq!(restored, column_major);
        assert!(!restored.is_standard_layout());
        assert!(round_trip(&array![[1.0, 2.0]]).is_standard_layout());
    }

    #[test]
    fn a_matrix_with_the_wrong_length_is_an_error() {
        assert!(<Array2<f64>>::from_repr((2, 2, false, vec![0])).is_err());
    }
}
