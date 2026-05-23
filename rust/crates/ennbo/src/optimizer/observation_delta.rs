use ndarray::{Array2, ArrayView2};

use super::observation_store::ObservationStore;
use crate::error::ENNError;

/// Result of an append-only observation batch.
#[derive(Debug, Clone)]
pub struct ObservationDelta {
    pub old_n: usize,
    pub new_n: usize,
    pub x_new: Array2<f64>,
    pub y_new: Array2<f64>,
}

impl ObservationDelta {
    pub fn x_new_view(&self) -> ArrayView2<'_, f64> {
        self.x_new.view()
    }

    pub fn y_new_view(&self) -> ArrayView2<'_, f64> {
        self.y_new.view()
    }
}

pub(crate) fn observation_delta_from_store(
    store: &ObservationStore,
    old_n: usize,
) -> Result<ObservationDelta, ENNError> {
    let new_n = store.len();
    if new_n <= old_n {
        return Err(ENNError::InvalidParameter(
            "observation delta requires appended rows".to_string(),
        ));
    }
    let (x_new, y_new) = store
        .rows_as_array2(old_n, new_n)
        .ok_or_else(|| ENNError::InvalidParameter("missing appended rows".to_string()))?;
    Ok(ObservationDelta {
        old_n,
        new_n,
        x_new,
        y_new,
    })
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;
    use crate::optimizer::observation_store::ObservationStore;
    use ndarray::array;

    #[test]
    fn observation_delta_from_store_appends() {
        let mut store = ObservationStore::new();
        store.push(array![0.0, 0.0], array![1.0]);
        let old_n = store.len();
        store.push(array![1.0, 0.0], array![2.0]);
        let delta = observation_delta_from_store(&store, old_n).unwrap();
        assert_eq!(delta.old_n, 1);
        assert_eq!(delta.new_n, 2);
        assert_eq!(delta.x_new.nrows(), 1);
        assert_eq!(delta.y_new.nrows(), 1);
    }
}
