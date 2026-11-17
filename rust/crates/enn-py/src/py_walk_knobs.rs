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

#[pyfunction(name = "take_pbuf_tag_stats")]
pub fn take_pbuf_tag_stats_py() -> (u64, u64, u64, u64) {
    ennbo::take_pbuf_tag_stats()
}

#[pyfunction(name = "set_pbuf_diag_targets")]
pub fn set_pbuf_diag_targets_py(rows: Vec<u32>) {
    ennbo::set_pbuf_diag_targets(rows);
}

#[pyfunction(name = "take_pbuf_diag")]
pub fn take_pbuf_diag_py() -> Vec<ennbo::PbufDiagRow> {
    ennbo::take_pbuf_diag()
}

#[pyfunction(name = "take_reparent_stats")]
pub fn take_reparent_stats_py() -> (u64, u64, u64, u64) {
    ennbo::take_reparent_stats()
}

#[pyfunction(name = "take_pbuf_rehome_stats")]
pub fn take_pbuf_rehome_stats_py() -> (u64, u64) {
    ennbo::take_pbuf_rehome_stats()
}
