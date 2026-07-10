# Plan: Raise disk dim cap to 8192 and remove HNSW drivers

## User Request

Increase the disk-backend dimension cap from 1024 to 8192. While doing that, remove HNSW and HNSW_DISK support entirely.

## Current State

### Dimension limits (`MAX_NUM_DIM = 1024`)

Two independent constants enforce disk observation limits:

| Constant | Value | Location | What it guards |
|----------|-------|----------|----------------|
| `MAX_NUM_DIM` | 1024 | `rust/crates/bpann/src/observation.rs`, `rust/crates/ennbo/src/backend/disk_observation.rs` | Feature dimension count |
| `MAX_RECORD_STRIDE` | 8 MiB (`8 * 1024 * 1024`) | same files | Bytes per mmap row (`num_dim × 8` for f64 `train_x`) |

Validation runs at backend open:

- **BPANN (production path for `BPANN_DISK`):** `bpann_validate_dim_limits()` in `rust/crates/bpann/src/observation.rs`, called from `rust/crates/bpann/src/backend.rs`. This is what enforces the dim cap for disk models after HNSW removal.
- **ennbo shared helpers:** `validate_dim_limits()` in `rust/crates/ennbo/src/backend/disk_observation.rs`, called from `rust/crates/ennbo/src/disk_hnsw/enn_backend.rs` today (and unit tests). After HNSW removal it is test-only; still update `MAX_NUM_DIM` there for consistency.

At `d = 8192`, f64 observation stride is 65,536 bytes (64 KiB) — well under the 8 MiB `MAX_RECORD_STRIDE` cap. Raising `MAX_NUM_DIM` alone is sufficient; `MAX_RECORD_STRIDE` does not need to change for 8192.

Tests that encode the 1024 cap:

- `rust/crates/bpann/src/lib.rs` (`test_open_rejects_num_dim_above_max`; also asserts error string contains literal `"1024"`)
- `rust/crates/bpann/tests/observation.rs`
- `rust/crates/bpann/src/observation.rs` (kiss test)
- `rust/crates/ennbo/src/backend/disk_observation.rs` (unit tests)
- No Python test currently asserts the 1024 rejection boundary

Unrelated `1024` elsewhere (do not change): `rust/crates/bpann/src/index/sync.rs` batch row-count threshold; `rust/crates/ennbo/src/posterior/neighbor.rs` perf-test sample sizes; `tests/test_enn_perf.py` sample sizes.

### Index drivers today

Rust `IndexDriver` (`rust/crates/ennbo/src/index.rs`):

| Variant | Role |
|---------|------|
| `Exact` (default) | Faiss `Flat` in-memory KNN |
| `HNSW` | Faiss `HNSW32` in-memory KNN |
| `HNSWDisk` | In-tree mmap HNSW graph under `work_dir/graph/` |
| `BpAnnDisk` | B+ANN disk index via `bpann` crate |

Python mirror (`src/enn/turbo/config/enn_index_driver.py`):

| `ENNIndexDriver` | Rust string |
|------------------|-------------|
| `FLAT` | `exact` |
| `HNSW` | `hnsw` |
| `HNSW_DISK` | `hnsw_disk` |
| `BPANN_DISK` | `bpann_disk` |

String parsing in PyO3:

- `rust/crates/enn-py/src/py_model.rs` — accepts `"HNSW"`, `"hnsw"`, `"HNSW_DISK"`, `"hnsw_disk"`, etc.
- `rust/crates/enn-py/src/py_optimizer.rs` — accepts `"hnsw"`, `"hnsw_disk"`

Default index driver for TuRBO config is already `ENNIndexDriver.FLAT` (`src/enn/turbo/config/enn_surrogate_config.py`).

### HNSW in-memory (`IndexDriver::HNSW`)

Routed through `KnnBackend` → `FaissBackend` (`rust/crates/ennbo/src/knn/mod.rs`, `rust/crates/ennbo/src/knn/faiss_backend.rs`). Faiss `index_factory` spec `"HNSW32"`. Faiss remains required for `Exact`/`Flat` after HNSW removal.

### HNSW disk (`IndexDriver::HNSWDisk`)

Self-contained module `rust/crates/ennbo/src/disk_hnsw/` (11 source files): mmap `nodes.bin`, background flush, graph build/search.

Wired through:

- `DiskEnnBackend::Hnsw(DiskHnswEnnBackend)` enum in `rust/crates/ennbo/src/backend/mod.rs`
- `pub mod disk_hnsw` and `pub use backend::DiskHnswEnnBackend` in `rust/crates/ennbo/src/lib.rs`
- `is_disk_index_driver()` matches `HNSWDisk | BpAnnDisk` in `rust/crates/ennbo/src/index.rs`

`EnnBackend` disk path has substantial HNSW-only logic: `wait_for_flush`, `schedule_background_flush`, and `disk_hnsw::flush` imports in `backend/mod.rs`. `model.rs` also calls `disk_hnsw::flush::try_schedule_background_flush`. BPANN disk has its own flush via `DiskBpannEnnBackend` / `bpann` crate (synchronous, no background thread).

Disk layout for HNSW (`rust/crates/ennbo/README.md`):

```
work_dir/graph/header.json, nodes.bin
metadata.json index_backend: "hnsw_disk"
```

### BPANN vs HNSW disk test APIs (not a drop-in swap)

Shared disk test helpers assume HNSW-only post-construction APIs:

- `rust/crates/ennbo/tests/disk_streaming_helper.rs` — `configure_low_flush_threshold()` downcasts to `DiskEnnBackend::Hnsw` and calls `set_pending_flush_threshold()` / `set_defer_append_indexing()` (HNSW-only setters)
- `DiskBpannEnnBackend` exposes `pending_flush_threshold()` / `append_syncs_at_threshold()` getters only; BPANN configures threshold via `BpannBackend::with_pending_flush_threshold()` at build time
- `rust/crates/ennbo/tests/optimizer_disk_flush.rs` — entirely HNSW background-flush integration (`wait_for_background_flush`, `DiskHnswEnnBackend`); delete or rewrite for BPANN synchronous flush semantics

### Tests and tooling encoding HNSW behavior

**Rust integration/unit tests to delete** (HNSW-specific):

- `rust/crates/ennbo/tests/disk_hnsw_integration.rs`
- `rust/crates/ennbo/tests/disk_hnsw_background_flush_unit.rs`
- `rust/crates/ennbo/tests/disk_hnsw_flush_wait_unit.rs`
- `rust/crates/ennbo/tests/disk_hnsw_pending_buffer.rs`
- `rust/crates/ennbo/tests/hnsw.rs`
- `rust/crates/ennbo/tests/flush_hnsw_helpers.rs`
- `rust/crates/ennbo/tests/kiss_disk_hnsw.rs`
- `rust/crates/ennbo/tests/optimizer_disk_flush.rs` (HNSW background-flush only; delete unless rewritten for BPANN)

**Rust tests to update** (remove HNSW references, keep BPANN disk coverage):

- `rust/crates/ennbo/src/index.rs` — `test_hnsw_search`, `test_hnsw_search_regression_*`, `faiss_spec_for_test(IndexDriver::HNSW)`
- `rust/crates/ennbo/src/knn/mod.rs` — `knn_backend_faiss_hnsw`, `knn_backend_hnsw_disk_driver_errors`
- `rust/crates/ennbo/src/knn/faiss_backend.rs` — HNSW branch and tests
- `rust/crates/ennbo/src/config.rs` — config tests using `IndexDriver::HNSW` / `HNSWDisk`
- `rust/crates/ennbo/src/model.rs` — disk tests using `IndexDriver::HNSWDisk`; remove `disk_hnsw::flush` import/usage
- `rust/crates/ennbo/src/backend/mod.rs` — remove HNSW wiring; delete/replace unit tests `disk_hnsw_enum_dispatch`, `disk_hnsw_new_empty_without_work_dir_errors`, etc.
- `rust/crates/ennbo/src/posterior/neighbor.rs` — one test uses `IndexDriver::HNSW`
- `rust/crates/ennbo/tests/turbo_disk_backend.rs` — defaults disk to `HNSWDisk`
- `rust/crates/ennbo/tests/disk_streaming_helper.rs` — retarget to BPANN; rework flush-threshold setup (see BPANN vs HNSW section)
- `rust/crates/ennbo/tests/disk_observation.rs` — uses `"hnsw_disk"` as example `index_backend` string (retarget to `bpann_disk`)
- `rust/crates/ennbo/src/backend/disk_observation.rs` — inline unit tests also use `"hnsw_disk"` (retarget to `bpann_disk`)
- `rust/crates/ennbo/tests/kiss_gate_coverage.rs` — remove `kiss_disk_hnsw_*` unit-ref macros
- `rust/crates/ennbo/tests/kiss_knn_backends.rs` — remove `DISK_HNSW_SRC` / `kiss_disk_hnsw_helper_names_in_source`
- `rust/crates/ennbo/tests/enn_backend.rs` — `IndexDriver::HNSWDisk`, `kiss_disk_hnsw_static_coverage_names`
- `rust/crates/ennbo/tests/kiss_repo_strings.rs` — remove `"DiskHnswEnnBackend"`, `"HNSWDisk"` from symbol registry

**Python tests to delete:**

- `tests/test_try_hnsw_disk.py`
- `tests/test_disk_hnsw_background_flush.py`

**Python tests to update:**

- `tests/test_enn_index_driver.py` — flat/hnsw metamorphic test; many `HNSW_DISK`-specific disk tests; parametrized disk tests include `HNSW_DISK`
- `tests/test_ops_stress.py` — parametrizes `flat`, `hnsw`, `hnsw_disk`, `bpann_disk`; CLI `hnsw_disk` command
- `tests/test_ops_stress_restart.py` — parametrizes `hnsw_disk`
- `tests/test_kiss_coverage.py` — enum inequality asserts involving HNSW variants
- `tests/test_rust_optimizer_fit_params.py` — `test_rust_optimizer_passes_hnsw_disk_index_driver`
- `tests/test_enn_neighbors.py` — `test_neighbors_hnsw_index_driver_returns_valid_indices`
- `tests/test_enn_index.py` — any HNSW references

**Examples / scripts / ops tooling:**

- `examples/try_hnsw_disk.py` — delete
- `go.sh` — HNSW lines already commented; remove dead references
- `ops/stress.py` (**gitignored**, but required for stress tests) — `INDEX_TYPE_CHOICES` includes `hnsw`, `hnsw_disk`; `DISK_DEFER_SYNC_DRIVERS` includes `HNSW_DISK`; `run_disk_rss_stress` defaults to `ENNIndexDriver.HNSW_DISK`; CLI accepts `hnsw_disk`

**Makefile** (`Makefile`):

- `PYTHON_SLOW_IGNORE` and `python-slow-test` list `test_disk_hnsw_background_flush.py` and `test_try_hnsw_disk.py` — remove those entries after file deletion

**Docs:**

- `rust/crates/ennbo/README.md` — documents `HNSW` and `hnsw_disk` layout; needs rewrite for `Exact` + `BpAnnDisk` only

## Requested Changes

1. Raise `MAX_NUM_DIM` from 1024 to 8192 in both `bpann` and `ennbo` disk observation modules; update tests that assert the old boundary.
2. Remove `IndexDriver::HNSW` and `IndexDriver::HNSWDisk` from Rust, Python `ENNIndexDriver`, and all PyO3 string parsing (including uppercase aliases in `py_model.rs`).
3. Delete the `disk_hnsw` module and all HNSW-specific Rust/Python tests, examples, and Makefile entries.
4. Simplify disk backend wiring so disk storage only supports `BpAnnDisk` (collapse `DiskEnnBackend` enum and HNSW-only flush paths).
5. Update docs, `ops/stress.py`, and remaining disk tests to use `BPANN_DISK` / `bpann_disk` exclusively.

## Q&A

### Q1. Does `MAX_RECORD_STRIDE` also need to increase for 8192 dimensions?

**Answer:** No. At `d = 8192`, f64 mmap row stride is 65,536 bytes. `MAX_RECORD_STRIDE` is 8,388,608 bytes (~1M dimensions for f64 rows). Only `MAX_NUM_DIM` changes.

### Q2. What happens to existing `hnsw_disk` work directories?

**Answer:** Breaking change. Checkpoints with `metadata.json` `"index_backend": "hnsw_disk"` will fail `validate_index_backend` (or equivalent) when reopened. Passing `"hnsw"` / `"hnsw_disk"` as `index_driver` should return a clear `InvalidParameter` / `ValueError` ("unknown index_driver" or "HNSW is no longer supported"). No migration path is required unless the user asks later.

### Q3. Can Faiss be removed entirely?

**Answer:** No. `IndexDriver::Exact` (Python `FLAT`) still uses Faiss `Flat` via `FaissBackend`. Only the `HNSW32` code path and its memory-usage graph estimate are removed; Faiss dependency stays in `rust/crates/ennbo/Cargo.toml`.

### Q4. Should `DiskEnnBackend` remain an enum after HNSW removal?

**Answer:** No — with only BPANN disk left, replace `DiskEnnBackend::BpAnn(DiskBpannEnnBackend)` with `DiskBpannEnnBackend` directly in `EnnBackend::Disk(Arc<Mutex<DiskBpannEnnBackend>>)`. This removes all HNSW match arms and `disk_hnsw::flush` imports from `backend/mod.rs` and `model.rs`.

### Q5. Should disk tests that today compare `HNSW_DISK` vs `FLAT` be kept for `BPANN_DISK`?

**Answer:** Yes. Retain the behavioral coverage (incremental add/search, `train_rows_at`, posterior with pending rows, scale_x reopen) by retargeting existing `HNSW_DISK` tests to `BPANN_DISK`. Drop tests whose sole purpose was HNSW-specific flush/graph mechanics (background flush thread, graph mmap layout, etc.).

### Q6. Is retargeting shared disk tests a simple enum swap?

**Answer:** No. `disk_streaming_helper.rs` must stop using HNSW-only setters (`set_pending_flush_threshold`, `set_defer_append_indexing`). For BPANN, pass low flush threshold via `BpannBackend::with_pending_flush_threshold()` when constructing the backend (may require exposing a builder hook on `DiskBpannEnnBackend::new_empty` or constructing via a test-only path). `optimizer_disk_flush.rs` should be deleted unless rewritten to test BPANN's synchronous `schedule_background_flush` (no background thread).

### Q7. Can HNSW removal be split into separate compile checkpoints?

**Answer:** No. Removing `IndexDriver::HNSW` / `HNSWDisk` from the enum while `disk_hnsw/`, `DiskEnnBackend::Hnsw`, and `backend/mod.rs` match arms still exist will not compile (`cargo check` fails). Enum removal, `disk_hnsw` deletion, `DiskEnnBackend` collapse, and all `src/` test updates must land in **one atomic commit**. Phase 1 (dim cap) is independent and can ship first.

## Plan

### Phase 1 — Raise dimension cap to 8192

- [ ] Change `MAX_NUM_DIM` from `1024` to `8192` in `rust/crates/bpann/src/observation.rs` and `rust/crates/ennbo/src/backend/disk_observation.rs`
- [ ] Update boundary rejection tests: `rust/crates/bpann/src/lib.rs` (including literal `"1024"` in error-string assert → `"8192"` or `MAX_NUM_DIM`), `rust/crates/bpann/tests/observation.rs`, `rust/crates/bpann/src/observation.rs` (kiss test), `rust/crates/ennbo/src/backend/disk_observation.rs` (unit tests)
- [ ] Add acceptance test at `d = 8192` and rejection at `d = 8193` (Rust unit test in `bpann` or `ennbo`; optional thin Python smoke opening a `BPANN_DISK` model at 8192 dims)

**Validation:** `cd rust && cargo test bpann_validate_dim_limits`; `cd rust && cargo test validate_dim_limits`; `cd rust && cargo test open_rejects_num_dim`; assert `bpann_validate_dim_limits(8192)` succeeds and `bpann_validate_dim_limits(8193)` fails with message containing `8192`.

### Phase 2 — Remove HNSW entirely (atomic; do not split)

All items below must land in one commit. Removing enum variants before deleting `disk_hnsw/` and collapsing `DiskEnnBackend` leaves the tree uncompilable (see Q7).

**Types, bindings, and production `src/`**

- [ ] Delete `rust/crates/ennbo/src/disk_hnsw/` directory entirely
- [ ] Remove `pub mod disk_hnsw` and `pub use backend::DiskHnswEnnBackend` from `rust/crates/ennbo/src/lib.rs`
- [ ] Remove `HNSW` and `HNSWDisk` from `IndexDriver` in `rust/crates/ennbo/src/index.rs`; update `is_disk_index_driver()` to match only `BpAnnDisk`; remove HNSW unit tests in the same file
- [ ] Replace `DiskEnnBackend` enum with direct `DiskBpannEnnBackend` in `rust/crates/ennbo/src/backend/mod.rs`; remove HNSW match arms, `disk_hnsw::flush` usage, HNSW-only unit tests, and default `IndexDriver::HNSWDisk` fallback in `driver()`
- [ ] Simplify `FaissBackend` / `KnnBackend` in `rust/crates/ennbo/src/knn/faiss_backend.rs` and `rust/crates/ennbo/src/knn/mod.rs` to accept only `IndexDriver::Exact`; remove HNSW branches and tests
- [ ] Update `rust/crates/ennbo/src/model.rs`: remove `disk_hnsw::flush` import/usage; retarget disk tests to `BpAnnDisk`; update error strings
- [ ] Update `rust/crates/ennbo/src/config.rs` tests to use `Exact` / `BpAnnDisk` instead of HNSW variants
- [ ] Update `rust/crates/ennbo/src/posterior/neighbor.rs` test that uses `IndexDriver::HNSW`
- [ ] Retarget inline unit tests in `rust/crates/ennbo/src/backend/disk_observation.rs` from `"hnsw_disk"` to `"bpann_disk"`
- [ ] Remove `HNSW` and `HNSW_DISK` from `ENNIndexDriver` and `ENN_INDEX_DRIVER_TO_RUST` in `src/enn/turbo/config/enn_index_driver.py`
- [ ] Remove all HNSW string parsing from `rust/crates/enn-py/src/py_model.rs` (`"HNSW"`, `"hnsw"`, `"HNSW_DISK"`, `"hnsw_disk"`) and `rust/crates/enn-py/src/py_optimizer.rs` (`"hnsw"`, `"hnsw_disk"`)

**Delete HNSW-specific tests and examples**

- [ ] Delete Rust integration test files listed in Current State (including `optimizer_disk_flush.rs` unless rewritten for BPANN)
- [ ] Delete `examples/try_hnsw_disk.py`, `tests/test_try_hnsw_disk.py`, `tests/test_disk_hnsw_background_flush.py`

**Update remaining tests and tooling**

- [ ] Update remaining Rust tests (`turbo_disk_backend.rs`, `disk_streaming_helper.rs`, `tests/disk_observation.rs`, `kiss_gate_coverage.rs`, `kiss_knn_backends.rs`, `enn_backend.rs`, `kiss_repo_strings.rs`) to remove HNSW references; rework `disk_streaming_helper.rs` for BPANN flush-threshold API (see Q6)
- [ ] Rewrite Python tests: `tests/test_enn_index_driver.py`, `tests/test_ops_stress.py`, `tests/test_ops_stress_restart.py`, `tests/test_kiss_coverage.py`, `tests/test_rust_optimizer_fit_params.py`, `tests/test_enn_neighbors.py`, `tests/test_enn_index.py`, and any other Python tests referencing HNSW
- [ ] Update `ops/stress.py`: drop `hnsw` / `hnsw_disk` from `INDEX_TYPE_CHOICES` and CLI; remove `HNSW_DISK` from `DISK_DEFER_SYNC_DRIVERS`; default disk RSS stress to `BPANN_DISK`
- [ ] Update `Makefile` slow-test lists to drop deleted test files
- [ ] Rewrite `rust/crates/ennbo/README.md` to document `Exact` + `BpAnnDisk` only; remove `hnsw_disk` layout section
- [ ] Clean `go.sh` commented HNSW lines

**Validation:** `cd rust && cargo check -p ennbo -p enn-py`; `cd rust && cargo nextest run --test-threads=8`; `make test`; repo grep (no matches):

```bash
rg 'disk_hnsw|HNSWDisk|hnsw_disk|IndexDriver::HNSW|DiskHnswEnnBackend|ENNIndexDriver\.HNSW|HNSW_DISK|\bhnsw\b' \
  rust/crates tests examples ops/stress.py src/enn go.sh
```

### Phase 3 — End-to-end validation

- [ ] Run `make lint` (clippy, ruff, kiss)
- [ ] Run `make python-slow-test` (includes `test_enn_index_driver.py`, `test_ops_stress.py`, `test_ops_stress_restart.py`)
- [ ] Smoke: create `BPANN_DISK` model at `d = 8192` with small `n`, call `ensure_index_sync()` and `posterior()`

**Validation:** `make lint` passes; `make test` passes; `make python-slow-test` passes; manual 8192-dim BPANN disk smoke succeeds without `num_dim … exceeds maximum` error.

(Point-in-time `.tex` reports under `docs/` and `reports/` are out of scope — no update needed.)
