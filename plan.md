# Plan: Persist BPANN disk index on close

## User Request

Persist the BPANN disk index when a disk-backed ENN session closes, so reopen does not repeat ~8s of index catch-up (as seen in `./ops/stress.py sample _enn/ 1000` where `init_s ≈ 8s` and `sample_s ≈ 0.2s`).

## Current State

### Symptom and measured breakdown

| Phase | Time | What runs |
|---|---|---|
| `init_s` | ~8s | `reopen_disk_bpann_enn` → construct + `ensure_index_sync` + query points |
| `sample_s` | ~0.2s | `posterior_function_draw` |

For `_enn/` (10k obs × 1k dims): metadata load ~60µs; reopen/construct ~8s; query points ~5ms. Almost all of `init_s` is BPANN index catch-up for rows 1000–9999, not query-point generation.

### Incremental index build (bpann)

`rust/crates/bpann/src/index/sync.rs`:

- `build_batch` always calls `build_*_with_persist(..., persist: false)` (lines 280–302).
- `maybe_compact_or_persist` only writes to disk when `indices.len() == 1` (lines 176–191).
- For 10k rows with 1000-row flushes, compaction stops at 3 fragments (`index_compact_threshold`), so **persist never runs** after the first 1000-row fragment.

On reopen, `BpannBackend::new` (`rust/crates/bpann/src/backend.rs`):

- Loads `indexed_rows` from `indexed_rows.bin` (authoritative over `metadata.json`; see `observation.rs:127–136`).
- Opens **one** on-disk fragment via `BpannIndex::open` when `header.json` exists.
- Sets in-memory `index.indexed_rows = persisted_rows.min(indexed_rows)` (header gates catch-up).
- If `persisted_rows < indexed_rows`, runs constructor catch-up via `ensure_sync_for_backend`.
- Writes `indexed_rows.bin` and `metadata.json` after catch-up, but **does not update `header.json` / `pages.bin`** when multiple fragments remain.

Result: every reopen repeats O(n·dim) index build; sidecar counters can claim 10k while `header.json` still says 1000.

### ENN disk reopen path (ennbo)

`EpistemicNearestNeighbors::new_with_storage` (`rust/crates/ennbo/src/model.rs:118–173`):

- Detects disk reopen (`train_x`/`train_y` empty + existing `metadata.json`).
- On disk reopen (or `num_obs != backend.len()`), calls `sync_obs_stats_from_backend`, which rescans all rows for scale moments and ends with `backend.ensure_index_sync` (lines 393–416).

`ops/stress.py`:

- `reopen_disk_bpann_enn` constructs empty arrays + `work_dir`, then calls `ensure_index_sync()` (lines 294–308).
- `run_enn_add_stress` (used to build `_enn/`) calls `schedule_background_flush()` per batch but **never** persists index to disk on exit (lines 383–384).
- `run_disk_rss_stress` does call `ensure_index_sync()` at end (line 126), but that alone does not persist multi-fragment stores.

### Close / flush hooks (exist but no-op)

| Location | Behavior today |
|---|---|
| `EpistemicNearestNeighbors::add` (`model.rs:252`) | Calls `backend.wait_for_flush()` **before every append** |
| `EnnBackend::ensure_index_sync` (`backend/mod.rs:219`) | Pre-calls `wait_for_flush()` |
| `EnnBackend::Drop` (`backend/mod.rs:305–308`) | Calls `wait_for_flush()` |
| `EnnBackend::wait_for_flush` | Delegates to disk backend |
| `DiskBpannEnnBackend::wait_for_flush` (`disk_bpann/enn_backend.rs:123–125`) | **No-op** (`Ok(())`) |
| `Surrogate::wait_for_background_flush` | Delegates to `model.backend.wait_for_flush()`; optimizer `tell` awaits it (`optimizer/mod.rs:161–162`) |
| Python `EpistemicNearestNeighbors` | Exposes `schedule_background_flush()` only; **no** persist-on-close / `close` / context manager |

**Important:** `wait_for_flush` is on the hot path for every `add()`. It cannot be repurposed as merge-and-persist-on-close without destroying deferred-ingest throughput (10k-row stress adds one row at a time).

### Tests encoding today's behavior

- `rust/crates/bpann/src/lib.rs`: `test_reopen_search_matches_pre_close` — small store (32 rows), single fragment; passes today.
- `rust/crates/bpann/tests/kiss_coverage.rs`: `ensure_index_sync_noop_and_single_index_persist` — single-fragment persist works.
- `tests/test_enn_index_driver.py`: reopen posterior parity tests; small stores where single-fragment persist succeeds.
- `tests/test_ops_stress_sample.py`: asserts `init_s >= 0` only; no reopen-speed regression guard.
- No test for multi-fragment store (10k rows, 1000-row threshold) reopen cost or header/pages alignment.

## Requested Changes

1. On disk session close, sync any pending rows and **persist a single on-disk BPANN index** covering all observations (`header.json`, `pages.bin`, `indexed_rows.bin`, `metadata.json` aligned with `n`).
2. Add an explicit **`persist_index_to_disk()`** API (Rust + Python) for close; wire **`EnnBackend::Drop`** and optimizer **`wait_for_background_flush`** (tell boundary) to it. **Leave `wait_for_flush` a cheap no-op** on disk so `add()` stays fast.
3. Expose `persist_index_to_disk()` on Python `EpistemicNearestNeighbors` (thin wrapper in `enn_class.py`). Optional later: context manager / `close()` sugar calling the same method.
4. Add regression tests: multi-fragment ingest → persist → reopen is fast and search-correct; on-disk header row count equals `n`.
5. Call `persist_index_to_disk()` at end of `run_enn_add_stress` (in `try/finally`) for disk drivers so `_enn/` and similar fixtures are reopen-ready even if the generator consumer stops early.

## Q&A

### Q1. Is `ensure_index_sync()` at close sufficient?

**Answer:** No. For stores with multiple in-memory fragments (typical after 1000-row deferred flushes to 10k rows), `ensure_index_sync` builds in RAM and updates `indexed_rows.bin`, but `maybe_compact_or_persist` does not write `header.json`/`pages.bin` unless `indices.len() == 1`. Close must **force merge to one fragment and persist** (e.g. `BpannIndex::concat_merge(..., persist: true)` in `rust/crates/bpann/src/index/build.rs:176–254`).

### Q2. Must close persist a single fragment, or support multi-fragment on disk?

**Answer:** Single fragment. Reopen opens one index from `index/header.json` (`backend.rs:75–76`). Multi-fragment disk format would require reopen changes; out of scope. Close-time compact-to-one matches the existing reopen contract.

### Q3. Use `wait_for_flush`, `persist_index_to_disk`, or `close()`?

**Answer:** Add **`persist_index_to_disk`** as the explicit close/persist contract (Rust + Python). **Do not** overload `wait_for_flush` — it is already called before every `add()` and must remain cheap (no-op on the synchronous disk path). **`EnnBackend::Drop`** and **`Surrogate::wait_for_background_flush`** (optimizer tell) call `persist_index_to_disk`, not `wait_for_flush`. Optional later: Python `close()` or context manager as sugar. Keep `wait_for_flush` defined for API stability and future async flush work.

### Q4. Is `Drop` / GC sufficient in Python?

**Answer:** No as the only mechanism. Rust `EnnBackend::Drop` will call `persist_index_to_disk`, but Python lifecycle is GC-driven and unreliable at interpreter shutdown. **Explicit `persist_index_to_disk()` is the contract** for notebooks, stress harnesses, and fixtures; `Drop` remains best-effort backup when the PyO3 wrapper is released.

### Q5. Should incremental `schedule_background_flush` also persist every batch?

**Answer:** No. Keep deferred in-memory indexing during ingest for throughput. Persist-on-close (explicit call, Drop, or optimizer tell) is the one-time cost; reopen becomes O(mmap) instead of O(n·dim) rebuild. Optimizer `tell` already calls `wait_for_background_flush`; retarget that hook to `persist_index_to_disk` so the store is reopen-ready after each tell — acceptable for that workflow and does not run on every row-level `add()`.

### Q6. Does close need to persist scale stats (`x_sum`, `y_sum`, etc.)?

**Answer:** Not required for the 8s fix. Reopen rescans rows in `sync_obs_stats_from_backend` today; with index persisted, `ensure_index_sync` becomes a no-op and reopen is still fast. Persisting stats is a separate optimization; do not block this plan on it.

## Plan

### Phase 1 — bpann: persist-to-disk primitive

- [ ] Add `IncrementalIndex::persist_to_disk(&mut self, ctx: &IndexBuildContext)` in `rust/crates/bpann/src/index/sync.rs`:
  - Call existing `ensure_sync(ctx, train_x.nrows)` to index any pending rows.
  - If `indices` is empty and `indexed_rows == 0`, return `Ok(())`.
  - If `indices.len() > 1`, merge all fragments via `BpannIndex::concat_merge(take(indices), index_dir, false)`, replace `indices` with single merged index.
  - Call `indices[0].persist()` and `obs::bpann_write_metadata(...)` + `obs::write_indexed_rows(...)`.
- [ ] Add `BpannBackend::persist_index_to_disk(&mut self)` in `rust/crates/bpann/src/backend.rs` that builds `IndexBuildContext` from backend fields and calls `self.index.persist_to_disk`.
- [ ] Add unit test in `rust/crates/bpann/tests/kiss_coverage.rs` or `lib.rs`: append ≥2000 rows with 1000-row threshold → `persist_index_to_disk` → reopen → `indexed_rows() == n`, `header.json` `indexed_rows == n`, `pages.bin` checksum stable across second reopen.

**Validation:** `cargo test -p bpann kiss_coverage` and `cargo test -p bpann test_reopen`; new test passes; after persist, `header.json` `indexed_rows` equals `num_obs.bin` / `train_x` row count.

### Phase 2 — ennbo: wire `persist_index_to_disk`

- [ ] Add `DiskBpannEnnBackend::persist_index_to_disk(&mut self)` (`rust/crates/ennbo/src/disk_bpann/enn_backend.rs`) → `self.inner.persist_index_to_disk()`.
- [ ] Add `EnnBackend::persist_index_to_disk(&self)` dispatch in `backend/mod.rs` (disk path via `Arc<Mutex<>>`; in-memory no-op).
- [ ] Add `EpistemicNearestNeighbors::persist_index_to_disk(&self)` on the model (`rust/crates/ennbo/src/model.rs`).
- [ ] Change `EnnBackend::Drop` to call `persist_index_to_disk()` instead of `wait_for_flush()`.
- [ ] Change `Surrogate::wait_for_background_flush` to call `model.backend.persist_index_to_disk()` (not `wait_for_flush`).
- [ ] **Leave `wait_for_flush` a no-op on disk** (`DiskBpannEnnBackend::wait_for_flush` stays `Ok(())`). Do **not** call `persist_index_to_disk` from `ensure_index_sync`'s existing `wait_for_flush` pre-call.
- [ ] Add Rust integration test: disk ENN, stream adds with deferred flush (threshold 1000, ≥2500 rows), `persist_index_to_disk`, drop model, reopen empty construct → `ensure_index_sync` is fast (no rebuild); neighbor indices match pre-close. Assert ingest with only `schedule_background_flush` (no persist) remains fast.

**Validation:** `cargo test -p ennbo`; disk reopen integration test passes; `BpannBackend::reopen` after persist does not mutate `pages.bin` on second open.

### Phase 3 — Python API and stress harness

- [ ] Expose `persist_index_to_disk()` on `PyEpistemicNearestNeighbors` (`rust/crates/enn-py/src/py_model.rs`) and `EpistemicNearestNeighbors` (`src/enn/enn/enn_class.py`).
- [ ] Update KISS symbol registries: `tests/test_kiss_fullrepo_symbol_registry.py`, `rust/crates/ennbo/tests/kiss_repo_strings.rs`, `rust/crates/enn-py` link test.
- [ ] In `run_enn_add_stress` (`ops/stress.py`), wrap the ingest loop in `try/finally` and call `model.persist_index_to_disk()` in `finally` when `index_driver in DISK_DEFER_SYNC_DRIVERS`.
- [ ] Add pytest in `tests/test_enn_index_driver.py` or `tests/test_ops_stress_sample.py`: build multi-batch disk store (e.g. 2500 rows, dim 32), `persist_index_to_disk()`, reopen, assert `header.json` indexed_rows == n and posterior matches fresh model.

**Validation:** `pytest tests/test_enn_index_driver.py tests/test_ops_stress_sample.py -k disk`; `pre-commit run --all-files` passes.

### Phase 4 — Reopen speed regression

- [ ] Add test (pytest, marked `@pytest.mark.slow`) that builds a disk store mimicking `_enn/` shape (10k rows, 1k dims optional/reduced for CI timing), calls `persist_index_to_disk()`, then times reopen via `run_sample_stress` or direct construct; assert `init_s < 1.0` (or a generous bound vs pre-fix ~8s).
- [ ] Document in test comment that `_enn/` must be rebuilt with `persist_index_to_disk()` after this lands for local benchmarking.

**Validation:** Slow test passes locally; `./ops/stress.py enn bpann_disk 10000 --work-dir /tmp/enn_persist_test --num-dim 100` followed by `./ops/stress.py sample /tmp/enn_persist_test 1000` reports `init_s` ≪ 8s.
