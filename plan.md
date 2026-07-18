# Plan: Concurrent search during soft sync

## User Request

Finish the remaining work from async soft sync + deferred hard persist: allow
**search/posterior to proceed while a background soft sync is in flight**
(former Phase 2b), and clean up stale docs that still claim flush is synchronous.

## Current State

### Done (not in scope)

- Soft sync vs hard persist: `maybe_compact` is RAM-only; `ensure_index_sync` does
  not write `pages.bin` / `skip_edges.bin`; `index_dirty` clears only on hard
  persist (`bpann/src/index/sync.rs`, `bpann/src/backend.rs`).
- Background worker: `DiskBackendHandle` in `ennbo/src/backend/flush_controller.rs`
  — job state beside the data mutex; `schedule_background_flush` returns while
  soft sync runs; `wait_for_flush` joins without holding the data lock.
- Surrogate `wait_for_background_flush`: `wait_for_flush` then soft `ensure_sync`
  (no hard persist). Drop / `persist_index_to_disk` wait then hard persist.
- Tests: soft sync leaves `pages.bin` untouched; schedule returns while barrier
  holds the worker (`persist_hardening.rs`, `disk_persist_index.rs`,
  `flush_controller.rs` tests).

### Remaining gap

- The flush worker still takes the **full data mutex for the entire soft sync**
  (`flush_controller.rs` → `soft_sync_inner`). Search/posterior serialize behind
  that lock while soft sync runs.
- `report.md` summary (bullet 5 / intro) still says
  `schedule_background_flush` “runs synchronously despite its name”; the
  mechanism section below it is already updated.

## Requested Changes

1. Publish protocol so search/posterior can proceed during soft sync (short write
   lock or `Arc` swap of fragment list + `indexed_rows`); keep `add` waiting
   before append so mmap cannot remap under the worker.
2. Fix stale `report.md` claims that flush is still synchronous.

## Q&A

### Q1. How should concurrent readers be implemented?

**Answer:** While soft-syncing, either (a) build the new fragment using a stable
mmap view then **publish under a short write lock**, or (b) use `RwLock` /
snapshot `Arc` for `indices` + `indexed_rows` so readers proceed. Prefer the
smallest change that unblocks `EnnBackend::Disk` search without holding the data
mutex for the whole soft sync. `add` must continue to `wait_for_flush` before
`mmap_append`; single-slot schedule relies on that invariant.

### Q2. Must pending brute-force still cover in-flight rows?

**Answer:** Yes. Until publish advances `indexed_rows`, new rows remain on the
pending leg (`bpann` hybrid search). After publish, they move to ANN fragments.
Tests must cover search **while soft sync is still in flight**, not only after
`wait_for_flush`.

## Plan

### Phase 1 — Concurrent search during soft sync

- [ ] Change soft sync so build/compact work does not hold the data mutex for the
      whole job; publish updated `indices` + `indexed_rows` under a short write
      lock (or equivalent `Arc` swap / `RwLock`).
- [ ] Keep `add` → `wait_for_flush` before `mmap_append`; keep single-slot
      schedule semantics.
- [ ] New test: concurrent `search` during soft sync does not deadlock
      (timeout-guarded), using the existing test flush barrier in
      `flush_controller.rs`.
- [ ] New test: `add` → schedule → **search/posterior without wait** observes new
      rows via pending or newly published fragments while soft sync is still in
      flight.

**Validation:**

- `cargo test -p ennbo` (includes `flush_controller` + disk persist tests).
- Concurrent search / in-flight publish tests above pass.
- Prior 2a tests still pass: schedule returns while barrier holds; wait drains
  pending.
- `PYTHONPATH=src pytest tests/test_enn_index_driver.py -q`

### Phase 2 — Doc alignment

- [ ] Update `report.md` summary text that still claims flush runs synchronously
      despite its name (mechanism section already correct).

**Validation:**

- Grep `report.md` shows no remaining claim that `schedule_background_flush` is
  synchronous.
