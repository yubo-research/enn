# Plan: Harden BPANN persist-on-close

## User Request

Malvin implemented the original persist-on-close plan. Fix/improve based on these review notes:

- concat-merge can change neighbors/posteriors before persist
- Every optimizer tell re-persists whole index
- Drop swallows errors; writes non-atomic
- Tests don’t guard pre-persist parity or 10k reopen

## Current State

### persist-on-close implementation (landed)

| Layer | Location | Behavior today |
|---|---|---|
| Index merge + write | `rust/crates/bpann/src/index/sync.rs:186–208` | `persist_to_disk` calls `ensure_sync`, then **`std::mem::take`s in-memory fragments**, `concat_merge`s them into one tree, **replaces `self.indices`**, and writes `header.json` / `pages.bin` / `skip_edges.bin`. |
| Backend wrapper | `rust/crates/bpann/src/backend.rs:273–284` | `persist_index_to_disk` → `persist_to_disk_for_backend`; clears `pending_unindexed` / `index_dirty`. |
| ENN dispatch | `rust/crates/ennbo/src/backend/mod.rs:52–57`, `312–315` | `persist_enn_backend_index`; **`Drop` calls it and discards errors** (`let _ = …`). |
| Model API | `rust/crates/ennbo/src/model.rs:276–278`, `src/enn/enn/enn_class.py:108–109` | Public `persist_index_to_disk()`. |
| Optimizer tell hook | `rust/crates/ennbo/src/optimizer/mod.rs:161–162` → `rust/crates/ennbo/src/surrogate.rs:332–337` | **`ENNSurrogate::wait_for_background_flush` calls `model.persist_index_to_disk()`** on every tell. |
| Stress harness | `ops/stress.py:400–402` | `run_enn_add_stress` `finally` calls `persist_index_to_disk()` for disk drivers. |

### Why concat-merge changes live results

Multi-fragment search (`IncrementalIndex::search_candidates`, `sync.rs:355–400`) queries a **fragment budget** of in-memory trees and merges top-k. `concat_merge` (`build.rs:176–254`) builds a **new single tree** with a fresh internal layout. Replacing `self.indices` during persist changes the search path **in the same session**, before any disk read on reopen.

`disk_persist_index.rs` captures neighbors **before** persist and compares to **after reopen**, not to **after persist in the same model**. No test asserts in-session pre/post persist parity for neighbors or posterior.

### Why every tell re-persists

`persist_to_disk` has **no fast path** when disk is already aligned: even with one fragment it always calls `index.persist()` and rewrites sidecars (`sync.rs:196–207`). Wiring this to `wait_for_background_flush` means **every optimizer tell** pays full merge + disk rewrite as `n` grows.

Disk `wait_for_flush` remains a no-op (`disk_bpann/enn_backend.rs:123–125`); `add()` is still cheap. The tell hook is the problem.

### Non-atomic, silent Drop

`BpannIndex::persist` (`build.rs:325–338`) writes `header.json`, `pages.bin`, and `skip_edges.bin` **in place, sequentially**. A crash mid-write can leave `header.json` pointing at a partial `pages.bin`. `EnnBackend::Drop` swallows failures; there is no logging crate in `ennbo`/`bpann`.

### Existing tests (gaps)

| Test | What it checks | Gap |
|---|---|---|
| `rust/crates/ennbo/tests/disk_persist_index.rs` | 2500 rows; neighbors before persist == after reopen; reopen < 1s | No pre/post persist in-session parity; reopen assertion compares multi-fragment pre-persist to merged-on-reopen (fragile — should use reference model per Q1) |
| `rust/crates/bpann/tests/kiss_coverage.rs:157–187` | Multi-fragment persist; header row count; `pages.bin` stable across second reopen | No search/posterior parity |
| `tests/test_enn_index_driver.py:271–320` | 2500 rows; post-persist posterior == post-reopen | No **pre**-persist baseline |
| `tests/test_ops_stress_sample.py:100–143` | Reopen < 1s after persist | **2000 × 32**, not 10k rows; no `@pytest.mark.slow` 10k-row reopen guard (`_enn/` has 10k rows × 1000 dim — test should use 10k rows, dim 32 for CI speed) |

## Requested Changes

1. **Tell boundary:** `wait_for_background_flush` syncs pending rows in RAM (`ensure_index_sync` / `index_access().ensure_sync()`); it must **not** call `persist_index_to_disk`.
2. **In-session parity:** `persist_index_to_disk` must **not** replace in-memory fragments via `concat_merge`; merge only for the on-disk snapshot. Neighbors and posterior in the live model must be unchanged across persist.
3. **Atomic disk write:** Persist index files atomically so crash mid-write cannot corrupt a readable store.
4. **Drop visibility:** Do not silently discard persist failures in `Drop`; surface them on stderr (explicit `persist_index_to_disk()` remains the contract that returns `Result`).
5. **Tests:** Add pre/post persist in-session parity (neighbors + posterior) and a **10k-row** reopen-speed regression.

## Q&A

### Q1. Should post-reopen match pre-persist multi-fragment neighbors?

**Answer:** No — that is not required. Multi-fragment search is an approximate path (fragment budget in `search_candidates`). Reopen loads the merged canonical on-disk tree. **In-session** pre/post persist must match (disk-only merge). **Reopen** correctness is: post-reopen matches a **reference** disk model built with the same data via eager single-tree indexing (existing pattern in `test_enn_disk_bpann_reopen_scale_x_posterior_matches_fresh_without_sync`: separate `work_dir`, `ensure_index_sync`, compare neighbors + posterior).

### Q2. Rename `wait_for_background_flush`?

**Answer:** No. Restore its semantics to “drain pending in-memory index work” without renaming the optimizer/surrogate trait method. `persist_index_to_disk` stays the explicit close/durability API.

### Q3. Idempotent persist when disk is already aligned?

**Answer:** Yes. After tell no longer persists, this matters mainly for repeated explicit calls and `Drop`. Skip rewrite when: pending rows are synced, `indices.len() == 1`, on-disk `header.json` `indexed_rows == n`, and `index_dirty` is false. Still rewrite when `indices.len() > 1` (need merged disk snapshot) or header lags `n`.

### Q4. Atomic write strategy?

**Answer:** Write `pages.bin` and `skip_edges.bin` to `*.tmp` siblings, `rename` into place, then write `header.json` last (commit marker). Update `metadata.json` and `indexed_rows.bin` after index files. Same pattern inside `BpannIndex::persist`.

## Plan

**Ordering:** Ship Phase 2 `disk_persist_index.rs` updates in the same PR as Phase 1 disk-only merge — otherwise the existing `pre_idx == post_reopen` assertion may fail once in-memory fragments stop being replaced on persist.

### Phase 1 — bpann: disk-only merge, idempotent persist, atomic writes

- [ ] Add `Clone` for `BpannIndex` (`build.rs`): `IndexHeader` and `Page` already derive `Clone`; needed so disk-only merge can clone `self.indices` before `concat_merge` (which consumes its input).
- [ ] Refactor `IncrementalIndex::persist_to_disk` (`sync.rs`):
  - Keep `ensure_sync(ctx, train_x.nrows)`.
  - When `indices.len() > 1`: `let merged = BpannIndex::concat_merge(self.indices.clone(), …, false)?`; call `merged.persist()` only; **leave `self.indices` unchanged**.
  - When `indices.len() == 1`: persist that fragment directly (unchanged path).
  - Add `needs_disk_rewrite(ctx)` guard: skip body when single fragment, `!index_dirty`, and on-disk header `indexed_rows == train_x.nrows`. (When `indices.len() > 1`, always merge-write — in-memory fragments stay multi after disk-only persist.)
- [ ] Add `BpannBackend::index_needs_disk_persist(&self) -> bool` exposing the guard (uses `index_dirty`, fragment count, on-disk header read). Optional sugar for callers; not required for core fix.
- [ ] Make `BpannIndex::persist` atomic (`build.rs:325–338`): write `pages.bin.tmp` + `skip_edges.bin.tmp`, rename, then write `header.json` last; propagate I/O errors.
- [ ] Add unit test `multi_fragment_persist_preserves_in_session_neighbors`: 2500 rows, 1000-row threshold, capture neighbors + distances before persist, call `persist_index_to_disk`, assert identical on same backend; assert `header.json` `indexed_rows == n`.
- [ ] Add unit test `persist_idempotent_skips_pages_rewrite`: **single-fragment** fixture (one batch or `len == 1` after sync); persist twice with no new rows; assert `pages.bin` checksum unchanged on second call (multi-fragment in-memory state always rewrites per Q3).
- [ ] Add unit test `persist_atomic_survives_partial_write`: simulate failed write (e.g. inject error after pages rename) and assert previous index still opens.

**Validation:** `cargo test -p bpann multi_fragment_persist`; `cargo test -p bpann kiss_coverage`; `cargo test -p bpann test_reopen`.

### Phase 2 — ennbo: tell sync-only, Drop logging

- [ ] Change `ENNSurrogate::wait_for_background_flush` (`surrogate.rs:332–337`) to call `model.index_access().ensure_sync()` instead of `model.persist_index_to_disk()`.
- [ ] Leave `EnnBackend::Drop` calling `persist_enn_backend_index`, but on error log to stderr (e.g. `eprintln!("ennbo: persist_index_to_disk on drop failed: {e}")`); keep `let _ =` only after logging.
- [ ] Add Rust integration test `disk_tell_does_not_rewrite_index`: build multi-batch disk ENN, call `persist_index_to_disk` once and record `pages.bin` checksum; simulate tell by calling `index_access().ensure_sync()` only; assert checksum unchanged and neighbors unchanged.
- [ ] Update `disk_persist_index.rs`:
  - Add in-session pre/post persist neighbor equality (same model, no reopen).
  - **Replace** `assert_eq!(pre_idx, post_idx)` with reference-model check: build fresh disk ENN with same data in a separate `work_dir`, `ensure_index_sync`, assert post-reopen neighbors match reference (pattern from `test_enn_disk_bpann_reopen_scale_x_posterior_matches_fresh_without_sync`).
  - Keep reopen speed assertion (`< 1s`).

**Validation:** `cargo test -p ennbo disk_persist_index`; `cargo test -p ennbo disk_tell`.

### Phase 3 — Python tests: pre-persist parity and 10k reopen

- [ ] Extend `tests/test_enn_index_driver.py`: in `test_enn_disk_persist_index_multi_batch_reopen_matches_posterior`, capture **pre-persist** neighbors + posterior; assert **post-persist** (same model, no reopen) matches pre-persist; assert **post-reopen** matches a **reference** fresh disk model (not pre-persist multi-fragment baseline).
- [ ] Replace/extend `tests/test_ops_stress_sample.py::test_disk_persisted_store_reopens_fast` with `@pytest.mark.slow` `test_disk_persisted_store_10k_reopens_fast`: **10_000 rows**, `num_dim=32` (row count matches `_enn/`; dim kept small for CI), batch ingest with `schedule_background_flush`, `persist_index_to_disk()`, time empty reopen construct, assert `init_s < 1.0`.
- [ ] Add pytest `test_disk_persist_pre_post_in_session_posterior_unchanged` if not covered by the extended index-driver test (2500 rows, multi-batch, explicit pre/post persist posterior/allclose on neighbors).

**Validation:** `pytest tests/test_enn_index_driver.py tests/test_ops_stress_sample.py -k persist`; `pre-commit run --all-files`.
