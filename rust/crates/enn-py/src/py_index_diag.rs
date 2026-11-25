use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyReadonlyArrayDyn;

use crate::py_args::matrix_view;
use crate::py_model::PyEpistemicNearestNeighbors;

fn occupancy_dict<'py>(py: Python<'py>, o: ennbo::o1::O1Occupancy) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("num_rows", o.num_rows)?;
    d.set_item("hash_bits", o.hash_bits)?;
    d.set_item("bucket_cap", o.bucket_cap)?;
    d.set_item("reachable_rows", o.reachable_rows)?;
    d.set_item("table_rows", o.table_rows)?;
    d.set_item("keys_used", o.keys_used)?;
    d.set_item("overflow_keys", o.overflow_keys)?;
    d.set_item("max_arrivals", o.max_arrivals)?;
    d.set_item("mean_depth", o.mean_depth)?;
    d.set_item("disk_nodes", o.disk_nodes)?;
    d.set_item("free_nodes", o.free_nodes)?;
    d.set_item("node_bytes", o.node_bytes)?;
    d.set_item("node_reads", o.node_reads)?;
    d.set_item("node_writes", o.node_writes)?;
    d.set_item("splits", o.splits)?;
    d.set_item("rebuilds", o.rebuilds)?;
    d.set_item("rebuilt_rows", o.rebuilt_rows)?;
    d.set_item("max_rebuild_rows", o.max_rebuild_rows)?;
    d.set_item("global_rebuilds", o.global_rebuilds)?;
    d.set_item("stale_trees", o.stale_trees)?;
    d.set_item("max_drift", o.max_drift)?;
    d.set_item("repairs_started", o.repairs_started)?;
    d.set_item("repairs_finished", o.repairs_finished)?;
    d.set_item("repairs_abandoned", o.repairs_abandoned)?;
    d.set_item("repair_rows", o.repair_rows)?;
    d.set_item("repair_cursor", o.repair_cursor)?;
    Ok(d)
}

#[pyfunction]
pub fn index_occupancy<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let occ = model
        .inner
        .index_access()
        .occupancy()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    occ.map(|o| occupancy_dict(py, o)).transpose()
}

#[pyfunction]
pub fn index_candidates(
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
    x: PyReadonlyArrayDyn<f64>,
    search_k: usize,
) -> PyResult<Option<Vec<Vec<u32>>>> {
    model
        .inner
        .index_access()
        .candidates(&matrix_view("x", &x)?, search_k)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn index_set_probe_budget(
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
    max_leaves: usize,
    max_candidates: usize,
) -> PyResult<()> {
    model
        .inner
        .index_access()
        .set_probe_budget(max_leaves, max_candidates)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (model, width, slow_us=30.0))]
pub fn index_set_io_width(model: PyRef<'_, PyEpistemicNearestNeighbors>, width: usize, slow_us: f64) -> PyResult<()> {
    model.inner.index_access().set_io_width(width, (slow_us.max(0.0) * 1e3) as u64).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (model, bound=1.0))]
pub fn index_set_bound(model: PyRef<'_, PyEpistemicNearestNeighbors>, bound: f64) -> PyResult<()> {
    model.inner.index_access().set_bound(bound).map_err(py_err)
}

#[pyfunction]
pub fn index_set_split_rule(model: PyRef<'_, PyEpistemicNearestNeighbors>, rule: &str) -> PyResult<()> {
    let rule: ennbo::o1::SplitRule = rule.parse().map_err(py_err)?;
    model.inner.index_access().set_split_rule(rule).map_err(py_err)
}

#[pyfunction]
pub fn index_set_repair(
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
    repair_rows: usize,
    stale_drift: f64,
    prefer_fresh: bool,
) -> PyResult<()> {
    model
        .inner
        .index_access()
        .set_repair(repair_rows, stale_drift, prefer_fresh)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn index_set_pread_pivots(model: PyRef<'_, PyEpistemicNearestNeighbors>, on: bool) -> PyResult<()> {
    model.inner.index_access().set_pread_pivots(on).map_err(py_err)
}

#[pyfunction]
#[pyo3(signature = (model, rows_per_insert, grow=1.3, min_rows=4096))]
pub fn index_set_row_order(model: PyRef<'_, PyEpistemicNearestNeighbors>, rows_per_insert: usize, grow: f64, min_rows: usize) -> PyResult<()> {
    let params = ennbo::o1::OrderParams { rows_per_insert, grow, min_rows };
    model.inner.index_access().set_row_order(params).map_err(py_err)
}

#[pyfunction]
pub fn index_set_par_rebuild(model: PyRef<'_, PyEpistemicNearestNeighbors>, above: u32) -> PyResult<()> {
    model.inner.index_access().set_par_rebuild(above).map_err(py_err)
}

#[pyfunction]
pub fn index_set_rerank(model: PyRef<'_, PyEpistemicNearestNeighbors>, rerank: u32) -> PyResult<()> {
    model.inner.index_access().set_rerank(rerank).map_err(py_err)
}

#[pyfunction]
pub fn index_set_trees(model: PyRef<'_, PyEpistemicNearestNeighbors>, n: usize) -> PyResult<()> {
    model.inner.index_access().set_trees(n).map_err(py_err)
}

#[pyfunction]
pub fn index_order_rows(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> PyResult<usize> {
    model.inner.index_access().order_rows().map_err(py_err)
}

#[pyfunction]
pub fn index_compact(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> PyResult<usize> {
    model.inner.index_access().compact().map_err(py_err)
}

#[pyfunction]
pub fn index_maintain(model: PyRef<'_, PyEpistemicNearestNeighbors>, max_rows: usize) -> PyResult<usize> {
    model
        .inner
        .index_access()
        .maintain(max_rows)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn index_node_io(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> PyResult<Option<(u64, u64)>> {
    model
        .inner
        .index_access()
        .node_io()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn index_maint_stats<'py>(py: Python<'py>, model: PyRef<'_, PyEpistemicNearestNeighbors>) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(m) = model.inner.index_access().maint_stats().map_err(py_err)? else {
        return Ok(None);
    };
    let d = PyDict::new_bound(py);
    d.set_item("insert_ns", m.insert_ns)?;
    d.set_item("repair_ns", m.repair_ns)?;
    d.set_item("repack_ns", m.repack_ns)?;
    d.set_item("repacks", m.repacks)?;
    d.set_item("repacked_nodes", m.repacked_nodes)?;
    d.set_item("rebuilt_rows", m.rebuilt_rows)?;
    d.set_item("splits", m.splits)?;
    d.set_item("order_ns", m.order_ns)?;
    d.set_item("order_rows", m.order_rows)?;
    d.set_item("order_builds", m.order_builds)?;
    d.set_item("ordered_rows", m.ordered_rows)?;
    d.set_item("gathered", m.gathered)?;
    d.set_item("rebuild_gathered", m.rebuild_gathered)?;
    d.set_item("route_rows", m.route_rows)?;
    Ok(Some(d))
}

fn py_err(e: ennbo::ENNError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pyfunction]
#[pyo3(signature = (model, budget_bytes, node_bytes=None))]
pub fn index_set_cache(
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
    budget_bytes: usize,
    node_bytes: Option<usize>,
) -> PyResult<()> {
    let access = model.inner.index_access();
    match node_bytes {
        Some(n) => access.set_cache_split(n, budget_bytes.saturating_sub(n)),
        None => access.set_cache_budget(budget_bytes),
    }
    .map_err(py_err)
}

#[pyfunction]
pub fn index_cache_stats<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(s) = model.inner.index_access().cache_stats().map_err(py_err)? else {
        return Ok(None);
    };
    let d = PyDict::new_bound(py);
    d.set_item("budget_bytes", s.budget_bytes)?;
    d.set_item("node_budget_bytes", s.node_budget_bytes)?;
    d.set_item("node_hits", s.node_hits)?;
    d.set_item("node_misses", s.node_misses)?;
    d.set_item("row_hits", s.row_hits)?;
    d.set_item("row_misses", s.row_misses)?;
    d.set_item("node_evictions", s.node_evictions)?;
    d.set_item("row_evictions", s.row_evictions)?;
    d.set_item("node_bytes", s.node_bytes)?;
    d.set_item("row_bytes", s.row_bytes)?;
    d.set_item("node_entries", s.node_entries)?;
    d.set_item("row_entries", s.row_entries)?;
    Ok(Some(d))
}

#[pyfunction]
pub fn index_set_profile(model: PyRef<'_, PyEpistemicNearestNeighbors>, enabled: bool) -> PyResult<()> {
    model.inner.index_access().set_profile(enabled).map_err(py_err)
}

#[pyfunction]
pub fn index_take_profile<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let Some(p) = model.inner.index_access().take_profile().map_err(py_err)? else {
        return Ok(None);
    };
    let d = PyDict::new_bound(py);
    d.set_item("queries", p.queries)?;
    d.set_item("hash_ns", p.hash_ns)?;
    d.set_item("route_ns", p.route_ns)?;
    d.set_item("leaf_ns", p.leaf_ns)?;
    d.set_item("row_ns", p.row_ns)?;
    d.set_item("distance_ns", p.distance_ns)?;
    d.set_item("search_ns", p.search_ns)?;
    d.set_item("split_visits", p.split_visits)?;
    d.set_item("leaf_visits", p.leaf_visits)?;
    d.set_item("leaf_pages", p.leaf_pages)?;
    d.set_item("candidates", p.candidates)?;
    d.set_item("coded", p.coded)?;
    d.set_item("reranked", p.reranked)?;
    Ok(Some(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_diag_functions_are_linked() {
        let _ = (index_occupancy, index_candidates, index_set_probe_budget, index_set_io_width, index_set_bound, index_set_split_rule, index_node_io, index_set_repair, index_maintain, index_compact, occupancy_dict);
        let _ = (index_set_cache, index_cache_stats, index_set_profile, index_take_profile, index_maint_stats, index_set_pread_pivots, index_set_row_order, index_order_rows, index_set_par_rebuild, index_set_rerank, index_set_trees, py_err);
    }
}
