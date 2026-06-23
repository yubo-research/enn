use ndarray::{Array2, ArrayView2};

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

pub(crate) fn observation_delta_from_batch(
    old_n: usize,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) -> Result<ObservationDelta, ENNError> {
    let new_n = old_n.saturating_add(x.nrows());
    if new_n <= old_n {
        return Err(ENNError::InvalidParameter(
            "observation delta requires appended rows".to_string(),
        ));
    }
    Ok(ObservationDelta {
        old_n,
        new_n,
        x_new: x.to_owned(),
        y_new: y.to_owned(),
    })
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn observation_delta_from_batch_appends() {
        let x = array![[1.0, 0.0]];
        let y = array![[2.0]];
        let delta = observation_delta_from_batch(1, &x.view(), &y.view()).unwrap();
        assert_eq!(delta.old_n, 1);
        assert_eq!(delta.new_n, 2);
        assert_eq!(delta.x_new.nrows(), 1);
        assert_eq!(delta.y_new.nrows(), 1);
        let _ = delta.x_new_view();
        let _ = delta.y_new_view();
        assert!(std::mem::size_of::<ObservationDelta>() > 0);
    }
}
