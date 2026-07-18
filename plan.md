# Plan: Async soft index sync + deferred hard persist

## User Request

Implement ideas **A** and **B** from the deferred BPANN disk-write discussion:

- **A.** Make `schedule_background_flush` genuinely asynchronous (worker thread), with a real `wait_for_flush` / `wait_for_background_flush`.
- **B.** Split in-memory index catch-up (“soft sync”) from on-disk index file writes (“hard persist”), so callers get a fast return from scheduling and can query while durability work is deferred.

(Summarized from the prior design thread; goal remains: fast `add` return path via deferred flush, queryable before write/index catch-up completes.)

## Current State

### Add / flush / query pipeline

| Step | Behavior today | Files |
|---|---|---|
| `add` | Validates, **`wait_for_flush`**, then `append_rows` | `rust/crates/ennbo/src/model.rs` |
| `append_rows` | `mmap_append` into `train_*.bin` (no `msync`); marks pending; does **not** index when `defer_append_indexing=true` (default) | `rust/crates/bpann/src/backend.rs`, `mmap_store.rs` |
| `schedule_background_flush` | If pending ≥ threshold, calls **`ensure_index_sync()` on the calling thread** | `rust/crates/ennbo/src/disk_bpann/enn_backend.rs` |
| `wait_for_flush` | **No-op** (`Ok(())`) | same |
| Search | Hybrid: ANN over indexed fragments + brute-force pending tail; disk sets `defer_index_sync_for_search() == true` | `bpann/src/backend.rs` `search`, `ennbo/.../disk_bpann/enn_backend.rs` |
| Drop / explicit persist | `EnnBackend::Drop` and `persist_index_to_disk` call full disk rewrite | `ennbo/src/backend/mod.rs`, `bpann` `persist_index_to_disk` |

Default pending threshold is **250** (`bpann/src/tuning.rs` `DEFAULT_PENDING_FLUSH_THRESHOLD`).

### Soft vs hard work is already partially mixed

In `IncrementalIndex` (`bpann/src/index/sync.rs`):

- `build_batch` builds fragments with **`persist=false`** (RAM only).
- `ensure_sync` then calls `maybe_compact_or_persist`, which **`persist()`s when there is a single fragment**, and always `write_indexed_rows`.
- `persist_to_disk` does `ensure_sync` then merges/persists fragments + metadata.

So “flush” today is not only indexing: after enough compaction it also does atomic index-file I/O (`persist_atomic.rs`).

### Call sites that assume the fake-async API

| Caller | Calls | Intent |
|---|---|---|
| `ops/stress.py`, Python ENN class | `schedule_background_flush` after adds | Don’t block every add on index work |
| Optimizer `ask` | `schedule_background_flush` | Overlap index work with next ask |
| Optimizer `tell` | `wait_for_background_flush` | Drain before tell mutates |
| Surrogate `wait_for_background_flush` | **`ensure_sync()`** (full soft sync via `ensure_index_sync`), **not** `wait_for_flush` | Historically “make index ready”; currently sync |
| `EnnBackend::ensure_index_sync` | `wait_for_flush` then sync | Safe under a future real worker |
| `model.add` | `wait_for_flush` before append | Avoid concurrent mmap remap vs flush readers |

Python exposes `schedule_background_flush` and `persist_index_to_disk` (`enn-py`, `enn_class.py`) but **not** `wait_for_flush`.

### Concurrency today

- Disk backend is `Arc<Mutex<DiskBpannEnnBackend>>` (`ennbo/src/backend/mod.rs`).
- All disk ops take the full mutex; there is no flush worker and no index snapshot publish path.
- Search during a long `ensure_index_sync` holds the same mutex as the sync caller, so the process is single-threaded on the disk backend for that duration.

### Adjacent tests / docs

- `tests/test_enn_index_driver.py` — multi-batch flush + `ensure_index_sync` + persist/reopen.
- `rust/crates/ennbo/tests/disk_persist_index.rs` — flush then persist-on-close reopen.
- `rust/crates/bpann/tests/persist_hardening.rs` — persist idempotency.
- `bpann` unit tests — `test_search_includes_pending_without_sync` (query before index sync).
- `report.md` already notes flush is sync despite the name.

## Requested Changes

1. Split BPANN index maintenance into **soft sync** (build/compact fragments in memory, update `indexed_rows` / pending) and **hard persist** (atomic write of index files + durable metadata), with `schedule_background_flush` performing soft sync only.
2. Run soft sync on a **background worker** so `schedule_background_flush` returns without waiting for index build; implement a real join in `wait_for_flush`.
3. Keep query correctness for new rows without waiting on soft sync or hard persist (existing pending brute-force path).
4. Keep hard persist on explicit `persist_index_to_disk` and `EnnBackend::Drop` (after draining the worker); do not fold hard persist into the background schedule path.
5. Wire surrogate/optimizer wait so it joins the worker (and completes soft sync if still pending) without forcing hard persist on every `tell`.

## Q&A

### Q1. Where should the worker live — `bpann` or `ennbo`?

**Answer:** Own the worker in **`ennbo`’s disk backend** (`DiskBpannEnnBackend` / `EnnBackend::Disk`), because that is where `Arc<Mutex<_>>`, `schedule_background_flush`, and `wait_for_flush` already live. Keep `bpann` responsible for the **soft_sync vs persist** API split on `BpannBackend` / `IncrementalIndex` (callable under the mutex). Do not put a thread inside `BpannBackend` itself; that would fight the outer mutex and Drop story.

### Q2. Must search run concurrently with an in-flight soft sync?

**Answer:** Yes for A’s latency goal. A worker that holds the full disk mutex for the entire soft sync would unblock `schedule_*` but still stall `search`/`posterior`. Required design: while soft-syncing, either (a) build the new fragment using a stable mmap view then **publish under a short write lock**, or (b) use `RwLock` / snapshot `Arc` for `indices` + `indexed_rows` so readers proceed. `add` must continue to `wait_for_flush` before `mmap_append` so the mapping cannot remap under the worker.

### Q3. Does soft sync write `indexed_rows` / metadata files?

**Answer:** Soft sync updates **in-memory** `indexed_rows` and clears `pending_unindexed` / `index_dirty` as today after catch-up. It must **not** call `BpannIndex::persist()` or rewrite `pages.bin` / `skip_edges.bin`. Writing `indexed_rows.bin` during soft sync is allowed only if reopen continues to rebuild when on-disk index lags (existing `persisted_rows < indexed_rows` path in `BpannBackend::new`); prefer matching that reopen contract in tests rather than inventing a new durability story.

### Q4. What should `wait_for_background_flush` (surrogate / optimizer `tell`) do after this change?

**Answer:** Join the worker (`wait_for_flush`), then ensure soft sync is complete for current `len()` (same as today’s `ensure_sync` drain). It must **not** call `persist_index_to_disk` (that was previously falsified as a tell-cost bug in `_kpop/exp_log_plan_persist_bpann.md`). Hard persist remains Drop / explicit API.

### Q5. Error reporting for background soft sync?

**Answer:** Store the last worker error on the disk backend. Surface it on `wait_for_flush`, `schedule_background_flush` (if a prior job failed), `ensure_index_sync`, and `persist_index_to_disk`. Do not swallow failures only in the worker thread.

## Plan

### Phase 1 — Soft sync vs hard persist API (sync, no thread yet)

- [ ] In `IncrementalIndex` (`sync.rs`), split `maybe_compact_or_persist` into:
  - `maybe_compact` — RAM amalgamation only, **no** `persist()`
  - leave `persist()` / `persist_to_disk` as the hard path
- [ ] Add `BpannBackend::soft_sync` (or rename-clarifying wrapper around ensure-sync-without-persist) that: `build_batch` for `[indexed_rows, len)`, `maybe_compact`, update pending/dirty counters; **does not** write index binaries.
- [ ] Keep `persist_index_to_disk` as: soft sync if needed, then merge/persist files + metadata (existing `persist_to_disk_for_backend` behavior, adjusted so it does not double-persist mid soft sync).
- [ ] Point today’s synchronous `schedule_background_flush` at **soft sync only** (still sync in this phase) so behavior change is testable before threading.
- [ ] Update `disk_bpann/enn_backend.rs` accordingly; leave `wait_for_flush` as no-op until Phase 2.

**Validation:**

- `cargo test -p bpann`
- `cargo test -p ennbo disk_persist`
- Assert multi-batch `schedule_background_flush` leaves searchable index (`indexed_rows` advanced) while `index/pages.bin` is unchanged until `persist_index_to_disk` (new unit/integration assertion in `persist_hardening.rs` or `disk_persist_index.rs`).
- Existing reopen tests still pass when persist is called before drop.

### Phase 2 — Background soft-sync worker

- [ ] On `EnnBackend::Disk`, add a single-slot flush worker (dedicated thread or `std::thread` + condition/`JoinHandle`):
  - state: idle | running | failed(error)
  - `schedule_background_flush`: if pending ≥ threshold and idle, spawn/signal soft sync; return immediately; if prior failure, return that error
  - `wait_for_flush`: join current job; propagate stored error; no-op if idle
- [ ] Publish protocol so search can proceed during soft sync (short write lock or `Arc` swap of fragment list + `indexed_rows`); document that `add` still waits before append.
- [ ] `ensure_index_sync` / `persist_index_to_disk` / `Drop`: `wait_for_flush` then proceed (Drop already persists; ensure wait happens first).
- [ ] Fix surrogate `wait_for_background_flush` to call `backend.wait_for_flush()` then soft-ensure (or equivalent), still **without** hard persist.
- [ ] Expose `wait_for_flush` on the Rust→Python path only if needed for tests; stress harness may keep using schedule + final persist.

**Validation:**

- New Rust test: schedule soft sync with threshold 1; assert `schedule_background_flush` returns while a barrier/slow build is in flight (injectible delay or large batch), then `wait_for_flush` completes and `pending_unindexed_count()==0`.
- New test: `add` → schedule → **search/posterior without wait** returns finite neighbors including new rows (pending or newly published fragments).
- New test: concurrent `search` during soft sync does not deadlock (timeout-guarded).
- `cargo test -p ennbo`
- `PYTHONPATH=src pytest tests/test_enn_index_driver.py -q` (disk persist / reopen cases)
- Optimizer path: `ask` schedules, `tell` waits — no hard persist per tell (timing or mock: `persist` not called between tell boundaries except Drop).

### Phase 3 — Hard persist contract + stress/docs alignment

- [ ] Confirm `persist_index_to_disk` and Drop: wait → soft sync if dirty → hard persist; idempotent fast path when on-disk already matches (`needs_disk_rewrite`).
- [ ] Update `ops/stress.py` expectations only if timers change meaning (schedule no longer includes soft-sync CPU on the main thread); keep final `persist_index_to_disk`.
- [ ] Adjust `report.md` / comments that claim flush is synchronous once behavior matches.
- [ ] Keep KISS symbol registry entries if new public methods are added (`soft_sync` only if public).

**Validation:**

- `rust/crates/bpann/tests/persist_hardening.rs` and `ennbo/tests/disk_persist_index.rs` pass.
- `PYTHONPATH=src pytest tests/test_ops_stress.py -q` (flush mocks / stress helpers).
- Manual smoke: `PYTHONPATH=src ./ops/stress.py enn bpann_disk 5000 --num-dim=100 --work-dir=/tmp/enn_async_flush --heartbeat-seconds=0` completes; reopen with same work-dir after process exit shows persisted index (`header.json` / fast init).
- `pre-commit run --all-files` if landing as a PR-ready change set.
