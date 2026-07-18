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
| Drop / explicit persist | `EnnBackend::Drop` and `persist_index_to_disk` both go through `persist_enn_backend_index` → full disk rewrite (no `wait_for_flush` today) | `ennbo/src/backend/mod.rs`, `bpann` `persist_index_to_disk` |

Default pending threshold is **250** (`bpann/src/tuning.rs` `DEFAULT_PENDING_FLUSH_THRESHOLD`).

### Soft vs hard work is already partially mixed

In `IncrementalIndex` (`bpann/src/index/sync.rs`):

- `build_batch` builds fragments with **`persist=false`** (RAM only).
- `ensure_sync` then calls `maybe_compact_or_persist`, which **`persist()`s when there is a single fragment**, and always `write_indexed_rows`.
- `compact()` **also** `persist()`s when amalgamation collapses to a single fragment (common: `index_compact_threshold` returns 1 for `indexed_rows <= 1000`).
- `persist_to_disk` does `ensure_sync` then merges/persists fragments + metadata.

So “flush” today is not only indexing: after enough compaction it also does atomic index-file I/O (`persist_atomic.rs`).

### Call sites that assume the fake-async API

| Caller | Calls | Intent |
|---|---|---|
| `ops/stress.py`, Python ENN class | `schedule_background_flush` after adds | Don’t block every add on index work |
| Optimizer `ask` | `schedule_background_flush` | Overlap index work with next ask |
| Optimizer `tell` | `wait_for_background_flush` | Drain before tell mutates |
| Surrogate `wait_for_background_flush` | **`ensure_sync()`** (soft sync via `ensure_index_sync`), **not** `wait_for_flush`; already **no** hard persist | Historically “make index ready”; currently sync |
| `EnnBackend::ensure_index_sync` | `wait_for_flush` then sync | Safe under a future real worker |
| `model.add` | `wait_for_flush` before append | Avoid concurrent mmap remap vs flush readers |

Python exposes `schedule_background_flush` and `persist_index_to_disk` (`enn-py`, `enn_class.py`) but **not** `wait_for_flush`.

### Concurrency today

- Disk backend is `Arc<Mutex<DiskBpannEnnBackend>>` (`ennbo/src/backend/mod.rs`).
- All disk ops take the full mutex; there is no flush worker and no index snapshot publish path.
- Search during a long `ensure_index_sync` holds the same mutex as the sync caller, so the process is single-threaded on the disk backend for that duration.
- Today’s dispatch (`disk_lock(arc)?.wait_for_flush()` / `schedule_*`) holds the data mutex for the whole call — unsafe to join a worker that also needs that mutex.

### Adjacent tests / docs

- `tests/test_enn_index_driver.py` — multi-batch flush + `ensure_index_sync` + persist/reopen.
- `rust/crates/ennbo/tests/disk_persist_index.rs` — flush then persist-on-close reopen; `disk_tell_does_not_rewrite_index` asserts post-persist `ensure_sync` does not rewrite `pages.bin` (does **not** cover pre-persist soft schedule).
- `rust/crates/bpann/tests/persist_hardening.rs` — persist idempotency.
- `bpann` unit tests — `test_search_includes_pending_without_sync` (query before index sync).
- `report.md` already notes flush is sync despite the name.

## Requested Changes

1. Split BPANN index maintenance into **soft sync** (build/compact fragments in memory, update `indexed_rows` / pending; optional `indexed_rows.bin`) and **hard persist** (atomic write of `pages.bin` / `skip_edges.bin` + durable metadata), with `schedule_background_flush` performing soft sync only.
2. Run soft sync on a **background worker** so `schedule_background_flush` returns without waiting for index build; implement a real join in `wait_for_flush`.
3. Keep query correctness for new rows without waiting on soft sync or hard persist (existing pending brute-force path).
4. Keep hard persist on explicit `persist_index_to_disk` and `EnnBackend::Drop` (after draining the worker); do not fold hard persist into the background schedule path.
5. Wire surrogate/optimizer wait so it joins the worker (and completes soft sync if still pending) without forcing hard persist on every `tell`.

## Q&A

### Q1. Where should the worker live — `bpann` or `ennbo`?

**Answer:** Own the worker in **`ennbo`’s `EnnBackend::Disk`**, because that is where `Arc<Mutex<_>>`, `schedule_background_flush`, and `wait_for_flush` already live. Keep `bpann` responsible for the **soft_sync vs persist** API split on `BpannBackend` / `IncrementalIndex` (callable under a short data lock). Do not put a thread inside `BpannBackend` itself.

**Flush controller layout (required to avoid deadlock):** Keep job state (`JoinHandle` / condvar / last error / idle|running|failed) **beside** the data mutex — e.g. a sibling field on `EnnBackend::Disk` or a small flush controller wrapping `Arc<Mutex<DiskBpannEnnBackend>>` — **not** inside the data mutex. Rules:

- `schedule_background_flush` must not hold the data lock across worker start.
- `wait_for_flush` must join **without** holding the data lock.
- The worker takes the data lock only for soft sync / publish (Phase 2a: for the whole soft sync; Phase 2b: short publish only).

### Q2. Must search run concurrently with an in-flight soft sync?

**Answer:** Yes for A’s latency goal (optimizer `ask` overlap). Ship it in **two milestones** inside Phase 2:

- **2a:** Real async worker + join; search may still serialize on the data mutex during soft sync (schedule returns immediately; ask/search may still block until soft sync finishes).
- **2b:** Concurrent readers — while soft-syncing, either (a) build the new fragment using a stable mmap view then **publish under a short write lock**, or (b) use `RwLock` / snapshot `Arc` for `indices` + `indexed_rows` so readers proceed.

`add` must continue to `wait_for_flush` before `mmap_append` so the mapping cannot remap under the worker. Single-slot schedule is safe only while that invariant holds (pending cannot grow during an in-flight soft sync).

### Q3. Does soft sync write `indexed_rows` / metadata files?

**Answer:** Soft sync updates **in-memory** `indexed_rows` and clears `pending_unindexed`. It **may** write `indexed_rows.bin` (matches today’s `ensure_sync` + reopen lag contract: `persisted_rows < indexed_rows` rebuild in `BpannBackend::new`). It must **not** call `BpannIndex::persist()` or rewrite `pages.bin` / `skip_edges.bin`. Full work-dir metadata rewrite stays on the hard path with persist.

**Dirty semantics:** Do **not** clear a “disk needs rewrite” bit on soft sync alone. Prefer either (a) keep `index_dirty` true until hard persist, or (b) introduce a separate `disk_dirty` cleared only by `persist_index_to_disk` / successful hard persist, so Drop/`needs_disk_rewrite` cannot skip rewriting when RAM index advanced but `pages.bin` lags. Prefer matching the existing reopen contract in tests rather than inventing a new durability story.

### Q4. What should `wait_for_background_flush` (surrogate / optimizer `tell`) do after this change?

**Answer:** Join the worker (`wait_for_flush`), then ensure soft sync is complete for current `len()` (same soft drain as today’s `ensure_sync`). Surrogate wait already avoids hard persist; this change adds the join, it is not a persist “fix.” It must **not** call `persist_index_to_disk` (previously falsified as a tell-cost bug in `_kpop/exp_log_plan_persist_bpann.md`). Hard persist remains Drop / explicit API.

### Q5. Error reporting for background soft sync?

**Answer:** Store the last worker error on the flush controller. Surface it on `wait_for_flush`, `schedule_background_flush` (if a prior job failed), `ensure_index_sync`, and `persist_index_to_disk` (via `persist_enn_backend_index`). Do not swallow failures only in the worker thread.

## Plan

### Phase 1 — Soft sync vs hard persist API (sync, no thread yet)

- [ ] In `IncrementalIndex` (`sync.rs`), strip hard I/O from **both** soft-path compact helpers:
  - Rename/split `maybe_compact_or_persist` → `maybe_compact` — RAM amalgamation only, **no** `persist()`, **no** metadata rewrite.
  - Update `compact()` the same way: after amalgamation, **do not** `persist()` when `indices.len() == 1`.
  - Leave `persist()` / `persist_to_disk` as the only hard path for `pages.bin` / `skip_edges.bin` + full metadata.
- [ ] Add `BpannBackend::soft_sync` (or rename-clarifying wrapper around ensure-sync-without-`persist`) that: `build_batch` for `[indexed_rows, len)`, `maybe_compact`, update pending counters; **may** write `indexed_rows.bin`; **must not** write `pages.bin` / `skip_edges.bin`.
- [ ] Keep disk-dirty true across soft sync (Q3); clear it only on successful hard persist / idempotent fast path.
- [ ] Keep `persist_index_to_disk` as: soft sync if needed, then merge/persist files + metadata (existing `persist_to_disk_for_backend` behavior, adjusted so soft sync does not mid-flight hard-persist).
- [ ] Point today’s synchronous `schedule_background_flush` at **soft sync only** (still sync in this phase) so behavior change is testable before threading.
- [ ] Update `disk_bpann/enn_backend.rs` accordingly; leave `wait_for_flush` as no-op until Phase 2.

**Validation:**

- `cargo test -p bpann`
- `cargo test -p ennbo disk_persist`
- Assert multi-batch `schedule_background_flush` leaves searchable index (`indexed_rows` advanced) while `index/pages.bin` is unchanged until `persist_index_to_disk` (new unit/integration assertion in `persist_hardening.rs` or `disk_persist_index.rs`; existing tell test only covers post-persist `ensure_sync`).
- Existing reopen tests still pass when persist is called before drop.

### Phase 2 — Background soft-sync worker

#### Phase 2a — Worker + real `wait_for_flush` (search may still serialize)

- [ ] On `EnnBackend::Disk`, add a single-slot flush controller **beside** the data mutex (Q1):
  - state: idle | running | failed(error)
  - `schedule_background_flush`: if pending ≥ threshold and idle, spawn/signal soft sync **without holding the data lock across start**; return immediately; if prior failure, return that error
  - `wait_for_flush`: join current job **without holding the data lock**; propagate stored error; no-op if idle
  - Worker: take data lock, soft sync, release (search still blocked while lock held — acceptable for 2a)
- [ ] Wire `wait_for_flush` into `persist_enn_backend_index` (shared by Drop and `model.persist_index_to_disk`), then soft sync if needed, then hard persist.
- [ ] `ensure_index_sync`: `wait_for_flush` then soft sync (already waits first; keep that once wait is real).
- [ ] Surrogate `wait_for_background_flush`: call `backend.wait_for_flush()` then soft-ensure (or equivalent); still **without** hard persist.
- [ ] Expose `wait_for_flush` on the Rust→Python path only if needed for tests; stress harness may keep using schedule + final persist.
- [ ] Document invariant: `add` always waits before append; single-slot schedule relies on it.

**Validation (2a):**

- New Rust test: schedule soft sync with threshold 1; assert `schedule_background_flush` returns while a barrier/slow build is in flight (injectable delay or large batch), then `wait_for_flush` completes and `pending_unindexed_count()==0`.
- New test: `add` → schedule → search/posterior without wait returns finite neighbors including new rows (pending path and/or completed soft sync under mutex).
- `cargo test -p ennbo`
- Optimizer path: `ask` schedules, `tell` waits — no hard persist per tell (timing or mock: `persist` not called between tell boundaries except Drop).

#### Phase 2b — Concurrent search during soft sync

- [ ] Publish protocol so search/posterior can proceed during soft sync (short write lock or `Arc` swap of fragment list + `indexed_rows`); keep `add` waiting before append.
- [ ] New test: concurrent `search` during soft sync does not deadlock (timeout-guarded).
- [ ] New test: `add` → schedule → **search/posterior without wait** observes new rows via pending or newly published fragments while soft sync is still in flight.

**Validation (2b + full Phase 2):**

- All 2a tests still pass.
- Concurrent search / in-flight publish tests above.
- `PYTHONPATH=src pytest tests/test_enn_index_driver.py -q` (disk persist / reopen cases)

### Phase 3 — Hard persist contract + stress/docs alignment

- [ ] Confirm `persist_index_to_disk` and Drop via `persist_enn_backend_index`: wait → soft sync if needed → hard persist; idempotent fast path when on-disk already matches (`needs_disk_rewrite` + disk-dirty cleared only after real hard persist).
- [ ] Update `ops/stress.py` expectations only if timers change meaning (schedule no longer includes soft-sync CPU on the main thread); keep final `persist_index_to_disk`.
- [ ] Adjust `report.md` / comments that claim flush is synchronous once behavior matches.
- [ ] Keep KISS symbol registry entries if new public methods are added (`soft_sync` only if public).

**Validation:**

- `rust/crates/bpann/tests/persist_hardening.rs` and `ennbo/tests/disk_persist_index.rs` pass.
- `PYTHONPATH=src pytest tests/test_ops_stress.py -q` (flush mocks / stress helpers).
- Manual smoke: `PYTHONPATH=src ./ops/stress.py enn bpann_disk 5000 --num-dim=100 --work-dir=/tmp/enn_async_flush --heartbeat-seconds=0` completes; reopen with same work-dir after process exit shows persisted index (`header.json` / fast init).
- `pre-commit run --all-files` if landing as a PR-ready change set.
