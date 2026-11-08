use crate::layout::EnnLayout;
use crate::model::EpistemicNearestNeighbors;
use ndarray::{array, Array2};
use std::ops::{Deref, DerefMut};
use tempfile::TempDir;

/// An ENN model and the temporary work_dir that outlives it.
pub(crate) struct TestModel {
    model: EpistemicNearestNeighbors,
    _work_dir: TempDir,
}

impl Deref for TestModel {
    type Target = EpistemicNearestNeighbors;
    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl DerefMut for TestModel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}

pub(crate) fn test_model_with(
    weighting: bool,
    build: impl FnOnce(EnnLayout) -> EpistemicNearestNeighbors,
) -> TestModel {
    let work_dir = TempDir::new().unwrap();
    let layout = EnnLayout::new(work_dir.path(), weighting).unwrap();
    TestModel {
        model: build(layout),
        _work_dir: work_dir,
    }
}

pub(crate) fn test_model(
    train_x: Array2<f64>,
    train_y: Array2<f64>,
    train_yvar: Option<Array2<f64>>,
    weighting: bool,
) -> TestModel {
    test_model_with(weighting, |layout| {
        EpistemicNearestNeighbors::new(train_x, train_y, train_yvar, layout).unwrap()
    })
}

pub(crate) fn test_epistemic_model_exact_unit_square() -> TestModel {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];

    test_model(train_x, train_y, None, false)
}
