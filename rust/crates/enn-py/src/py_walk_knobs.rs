use pyo3::prelude::*;

#[pyfunction(name = "set_radius_weight")]
pub fn set_radius_weight_py(value: f32) {
    ennbo::set_radius_weight(value);
}

#[pyfunction(name = "set_patience_coefficient")]
pub fn set_patience_coefficient_py(value: f64) {
    ennbo::set_patience_coefficient(value);
}

#[pyfunction(name = "set_walk_trace")]
pub fn set_walk_trace_py(on: bool) {
    ennbo::set_walk_trace(on);
}

#[pyfunction(name = "take_walk_trace")]
pub fn take_walk_trace_py() -> ennbo::WalkTrace {
    ennbo::take_walk_trace()
}
