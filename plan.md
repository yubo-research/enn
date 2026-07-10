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

- BPANN: `bpann_validate_dim_limits()` in `rust/crates/bpann/src/observation.rs`, called from `rust/crates/bpann/src/backend.rs`
- Shared disk helpers: `validate_dim_limits()` in `rust/crates/ennbo/src/backend/disk_observation.rs`, called from `rust/crates/ennbo/src/disk_hnsw/enn_backend.rs` today

At `d = 8192`, f64 observation stride is 65,536 bytes (64 KiB) — well under the 8 MiB `MAX_RECORD_STRIDE` cap. Raising `MAX_NUM_DIM` alone is sufficient; `MAX_RECORD_STRIDE` does not need to change for 8192.

Tests that encode the 1024 cap:

- `rust/crates/bpann/src/lib.rs` (`test_open_rejects_num_dim_above_max`)
- `rust/crates/bpann/tests/observation.rs`
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

- `rust/crates/enn-py/src/py_model.rs`
- `rust/crates/enn-py/src/py_optimizer.rs`

Default index driver for TuRBO config is already `ENNIndexDriver.FLAT` (`src/enn/turbo/config/enn_surrogate_config.py`).

### HNSW in-memory (`IndexDriver::HNSW`)

Routed through `KnnBackend` → `FaissBackend` (`rust/crates/ennbo/src/knn/mod.rs`, `rust/crates/ennbo/src/knn/faiss_backend.rs`). Faiss `index_factory` spec `"HNSW32"`. Faiss remains required for `Exact`/`Flat` after HNSW removal.

### HNSW disk (`IndexDriver::HNSWDisk`)

Self-contained module `rust/crates/ennbo/src/disk_hnsw/` (11 source files): mmap `nodes.bin`, background flush, graph build/search.

Wired through:

- `DiskEnnBackend::Hnsw(DiskHnswEnnBackend)` enum in `rust/crates/ennbo/src/backend/mod.rs`
- `pub mod disk_hnsw` and `pub use backend::DiskHnswEnnBackend` in `rust/crates/ennbo/src/lib.rs`
- `is_disk_index_driver()` matches `HNSWDisk | BpAnnDisk` in `rust/crates/ennbo/src/index.rs`

`EnnBackend` disk path has substantial HNSW-only logic: `wait_for_flush`, `schedule_background_flush`, and `disk_hnsw::flush` imports in `backend/mod.rs`. BPANN disk has its own flush via `DiskBpannEnnBackend` / `bpann` crate.

Disk layout for HNSW (`rust/crates/ennbo/README.md`):

```
work_dir/graph/header.json, nodes.bin
metadata.json index_backend: "hnsw_disk"
```

### Tests and tooling encoding HNSW behavior

**Rust integration/unit tests to delete** (HNSW-specific):

- `rust/crates/ennbo/tests/disk_hnsw_integration.rs`
- `rust/crates/ennbo/tests/disk_hnsw_background_flush_unit.rs`
- `rust/crates/ennbo/tests/disk_hnsw_flush_wait_unit.rs`
- `rust/crates/ennbo/tests/disk_hnsw_pending_buffer.rs`
- `rust/crates/ennbo/tests/hnsw.rs`
- `rust/crates/ennbo/tests/flush_hnsw_helpers.rs`
- `rust/crates/ennbo/tests/kiss_disk_hnsw.rs`

**Rust tests to update** (remove HNSW references, keep BPANN disk coverage):

- `rust/crates/ennbo/src/index.rs` — `test_hnsw_search`, `test_hnsw_search_regression_*`, `faiss_spec_for_test(IndexDriver::HNSW)`
- `rust/crates/ennbo/src/knn/mod.rs` — `knn_backend_faiss_hnsw`, `knn_backend_hnsw_disk_driver_errors`
- `rust/crates/ennbo/src/knn/faiss_backend.rs` — HNSW branch and tests
- `rust/crates/ennbo/src/config.rs` — config tests using `IndexDriver::HNSW` / `HNSWDisk`
- `rust/crates/ennbo/src/model.rs` — disk tests using `IndexDriver::HNSWDisk`
- `rust/crates/ennbo/src/posterior/neighbor.rs` — one test uses `IndexDriver::HNSW`
- `rust/crates/ennbo/tests/turbo_disk_backend.rs` — defaults disk to `HNSWDisk`
- `rust/crates/ennbo/tests/optimizer_disk_flush.rs` — HNSW disk flush test
- `rust/crates/ennbo/tests/disk_streaming_helper.rs` — parametrized over `HNSWDisk`
- `rust/crates/ennbo/tests/disk_observation.rs` — uses `"hnsw_disk"` as example `index_backend` string (retarget to `bpann_disk`)
- `rust/crates/ennbo/tests/kiss_gate_coverage.rs` — `kiss_disk_hnsw_*` unit-ref macros
- `rust/crates/ennbo/tests/kiss_knn_backends.rs`, `rust/crates/ennbo/tests/enn_backend.rs`

**Python tests to delete:**

- `tests/test_try_hnsw_disk.py`
- `tests/test_disk_hnsw_background_flush.py`

**Python tests to update:**

- `tests/test_enn_index_driver.py` — flat/hnsw metamorphic test; many `HNSW_DISK`-specific disk tests; parametrized disk tests include `HNSW_DISK`
- `tests/test_ops_stress.py` — parametrizes `flat`, `hnsw`, `hnsw_disk`, `bpann_disk`; CLI `hnsw_disk` command
- `tests/test_kiss_coverage.py` — enum inequality asserts involving HNSW variants
- `tests/test_rust_optimizer_fit_params.py` — `test_rust_optimizer_passes_hnsw_disk_index_driver`
- `tests/test_enn_neighbors.py` — `test_neighbors_hnsw_index_driver_returns_valid_indices`
- `tests/test_enn_index.py` — any HNSW references

**Examples / scripts:**

- `examples/try_hnsw_disk.py` — delete
- `go.sh` — HNSW lines already commented; remove dead references

**Makefile** (`Makefile`):

- `PYTHON_SLOW_IGNORE` and `python-slow-test` list `test_disk_hnsw_background_flush.py` and `test_try_hnsw_disk.py` — remove those entries after file deletion

**Docs:**

- `rust/crates/ennbo/README.md` — documents `HNSW` and `hnsw_disk` layout; needs rewrite for `Exact` + `BpAnnDisk` only

## Requested Changes

1. Raise `MAX_NUM_DIM` from 1024 to 8192 in both `bpann` and `ennbo` disk observation modules; update tests that assert the old boundary.
2. Remove `IndexDriver::HNSW` and `IndexDriver::HNSWDisk` from Rust, Python `ENNIndexDriver`, and all PyO3 string parsing.
3. Delete the `disk_hnsw` module and all HNSW-specific Rust/Python tests, examples, and Makefile entries.
4. Simplify disk backend wiring so disk storage only supports `BpAnnDisk` (collapse `DiskEnnBackend` enum and HNSW-only flush paths).
5. Update docs and remaining disk tests to use `BPANN_DISK` / `bpann_disk` exclusively.

## Q&A

### Q1. Does `MAX_RECORD_STRIDE` also need to increase for 8192 dimensions?

**Answer:** No. At `d = 8192`, f64 mmap row stride is 65,536 bytes. `MAX_RECORD_STRIDE` is 8,388,608 bytes (~1M dimensions for f64 rows). Only `MAX_NUM_DIM` changes.

### Q2. What happens to existing `hnsw_disk` work directories?

**Answer:** Breaking change. Checkpoints with `metadata.json` `"index_backend": "hnsw_disk"` will fail `validate_index_backend` (or equivalent) when reopened. Passing `"hnsw"` / `"hnsw_disk"` as `index_driver` should return a clear `InvalidParameter` / `ValueError` ("unknown index_driver" or "HNSW is no longer supported"). No migration path is required unless the user asks later.

### Q3. Can Faiss be removed entirely?

**Answer:** No. `IndexDriver::Exact` (Python `FLAT`) still uses Faiss `Flat` via `FaissBackend`. Only the `HNSW32` code path and its memory-usage graph estimate are removed; Faiss dependency stays in `rust/crates/ennbo/Cargo.toml`.

### Q4. Should `DiskEnnBackend` remain an enum after HNSW removal?

**Answer:** No — with only BPANN disk left, replace `DiskEnnBackend::BpAnn(DiskBpannEnnBackend)` with `DiskBpannEnnBackend` directly in `EnnBackend::Disk(Arc<Mutex<DiskBpannEnnBackend>>)`. This removes all HNSW match arms and `disk_hnsw::flush` imports from `backend/mod.rs`.

### Q5. Should disk tests that today compare `HNSW_DISK` vs `FLAT` be kept for `BPANN_DISK`?

**Answer:** Yes. Retain the behavioral coverage (incremental add/search, `train_rows_at`, posterior with pending rows, scale_x reopen) by retargeting existing `HNSW_DISK` tests to `BPANN_DISK`. Drop tests whose sole purpose was HNSW-specific flush/graph mechanics.

## Plan

### Phase 1 — Raise dimension cap to 8192

- [ ] Change `MAX_NUM_DIM` from `1024` to `8192` in `rust/crates/bpann/src/observation.rs` and `rust/crates/ennbo/src/backend/disk_observation.rs`
- [ ] Update hardcoded `1024` in rejection tests: `rust/crates/bpann/src/lib.rs`, `rust/crates/bpann/tests/observation.rs`, `rust/crates/bpann/src/observation.rs` (kiss test), `rust/crates/ennbo/src/backend/disk_observation.rs` (unit tests)
- [ ] Add acceptance test at `d = 8192` and rejection at `d = 8193` (Rust unit test in `bpann` or `ennbo`; optional thin Python smoke opening a `BPANN_DISK` model at 8192 dims)

**Validation:** `cd rust && cargo test bpann_validate_dim_limits`; `cd rust && cargo test validate_dim_limits`; `cd rust && cargo test open_rejects_num_dim`; assert `bpann_validate_dim_limits(8192)` succeeds and `bpann_validate_dim_limits(8193)` fails with message containing `8192`.

### Phase 2 — Remove HNSW drivers from types and bindings

- [ ] Remove `HNSW` and `HNSWDisk` from `IndexDriver` in `rust/crates/ennbo/src/index.rs`; update `is_disk_index_driver()` to match only `BpAnnDisk`
- [ ] Remove `HNSW` and `HNSW_DISK` from `ENNIndexDriver` and `ENN_INDEX_DRIVER_TO_RUST` in `src/enn/turbo/config/enn_index_driver.py`
- [ ] Remove `"hnsw"` / `"hnsw_disk"` parsing from `rust/crates/enn-py/src/py_model.rs` and `rust/crates/enn-py/src/py_optimizer.rs`
- [ ] Simplify `FaissBackend` / `KnnBackend` in `rust/crates/ennbo/src/knn/faiss_backend.rs` and `rust/crates/ennbo/src/knn/mod.rs` to accept only `IndexDriver::Exact`
- [ ] Update error strings in `rust/crates/ennbo/src/model.rs` and `rust/crates/ennbo/src/backend/mod.rs` from `"HNSWDisk or BpAnnDisk"` to `"BpAnnDisk"` only
- [ ] Update `rust/crates/ennbo/src/config.rs` tests to use `Exact` / `BpAnnDisk` instead of HNSW variants

**Validation:** `cd rust && cargo check -p ennbo -p enn-py`; `grep -r 'IndexDriver::HNSW' rust/crates/ennbo/src rust/crates/enn-py/src` returns no matches; `grep -r 'HNSW_DISK\|ENNIndexDriver.HNSW' src/enn` returns no matches.

### Phase 3 — Delete disk_hnsw and simplify disk backend

- [ ] Delete `rust/crates/ennbo/src/disk_hnsw/` directory entirely
- [ ] Remove `pub mod disk_hnsw` and `pub use backend::DiskHnswEnnBackend` from `rust/crates/ennbo/src/lib.rs`
- [ ] Replace `DiskEnnBackend` enum with direct `DiskBpannEnnBackend` in `rust/crates/ennbo/src/backend/mod.rs`; remove HNSW match arms, `disk_hnsw::flush` usage, and `wait_for_flush`/`schedule_background_flush` HNSW branches (delegate entirely to BPANN paths)
- [ ] Delete HNSW-specific Rust test files listed in Current State
- [ ] Update remaining Rust tests (`index.rs`, `model.rs`, `posterior/neighbor.rs`, `turbo_disk_backend.rs`, `optimizer_disk_flush.rs`, `disk_streaming_helper.rs`, `disk_observation.rs`, `kiss_gate_coverage.rs`, `kiss_knn_backends.rs`, `enn_backend.rs`) to remove HNSW references
- [ ] Delete `examples/try_hnsw_disk.py`, `tests/test_try_hnsw_disk.py`, `tests/test_disk_hnsw_background_flush.py`
- [ ] Rewrite `tests/test_enn_index_driver.py`, `tests/test_ops_stress.py`, `tests/test_kiss_coverage.py`, `tests/test_rust_optimizer_fit_params.py`, `tests/test_enn_neighbors.py`, and any other Python tests referencing HNSW
- [ ] Update `Makefile` slow-test lists to drop deleted test files
- [ ] Rewrite `rust/crates/ennbo/README.md` to document `Exact` + `BpAnnDisk` only; remove `hnsw_disk` layout section
- [ ] Clean `go.sh` commented HNSW lines

**Validation:** `cd rust && cargo nextest run --test-threads=8`; `make test`; `grep -r 'disk_hnsw\|HNSWDisk\|hnsw_disk\|IndexDriver::HNSW' rust/crates tests examples` returns no matches (excluding git history); `grep -r 'ENNIndexDriver.HNSW\|HNSW_DISK' src tests` returns no matches.

### Phase 4 — End-to-end validation

- [ ] Run `make lint` (clippy, ruff, kiss)
- [ ] Run `make python-slow-test` (includes `test_enn_index_driver.py`, `test_ops_stress.py`)
- [ ] Smoke: create `BPANN_DISK` model at `d = 8192` with small `n`, call `ensure_index_sync()` and `posterior()`

**Validation:** `make lint` passes; `make test` passes; `make python-slow-test` passes; manual 8192-dim BPANN disk smoke succeeds without `num_dim … exceeds maximum` error.
