# Plan: Cap pending brute-force cost (idea E)

## User Request

Plan the remaining work from the A–E flush design: **E. Cap pending
brute-force cost** — when soft flush lags or a large add leaves a big unindexed
tail, bound `leg_b` (`O(pending × D)` per query) with a hard backstop (sync
flush / block adds until catch-up). Soft threshold scheduling already exists.

## Current State

### Soft threshold only

| Piece | Behavior today |
|---|---|
| `pending_flush_threshold` (default **250**) | Soft gate: `schedule_background_flush` spawns soft sync when `pending_unindexed >=` this |
| Soft sync | Builds/compacts RAM index; clears pending; does **not** write `pages.bin` |
| Hard persist | `persist_index_to_disk` / Drop only |
| Search | Hybrid ANN(`indexed`) ∪ brute-force pending (`bpann_brute_force_topk_mmap` as `leg_b`) |
| Disk search | `defer_index_sync_for_search() == true` — queries do **not** force sync |

Key files: `bpann/src/backend.rs` (append + search), `ennbo/src/backend/flush_controller.rs`,
`bpann/src/tuning.rs`, `ennbo/src/file_config.rs`.

### Why pending can still spike

- **Well-behaved `add` + `schedule`:** `model.add` already `wait_for_flush` before
  append (`ennbo/src/model.rs`), and the flush worker is single-slot, so steady
  state pending is roughly bounded by one batch after a soft sync. Soft threshold
  alone is often enough here.
- **Large one-shot add:** appending `B ≫ soft` rows in one call leaves
  `pending ≈ B` until the next wait/sync; search pays full `leg_b` in that window.
- **Add without `schedule_background_flush`:** pending grows without bound;
  every query brute-forces the whole tail.
- **No hard knob** exists in `BpannTuning` / config today.

### Explicitly not chosen here

- Over-fetch from the ANN index to shrink the pending window (approximate; changes
  neighbor semantics). Out of scope for this plan.
- Search-time sync when pending ≥ hard (would surprise
  `defer_index_sync_for_search` callers). Reopen with a large pending tail is
  bounded on the next `add` / `schedule`, not on query.

## Requested Changes

1. Add a configurable **hard** pending cap,
   `pending_hard_flush_threshold`, with default `1000` (`4 ×` default soft `250`),
   validated `>= pending_flush_threshold`.
2. After a deferred-path append, if `pending_unindexed >=` hard cap: soft-sync
   on the calling thread so pending is drained before `add` returns.
   (`model.add` already `wait_for_flush` before append; schedule’s hard path
   also waits. Join lives at the ennbo flush layer, not inside `bpann`.)
3. If `schedule_background_flush` sees pending already at/above the hard cap:
   `wait_for_flush` + soft-sync synchronously instead of fire-and-forget.
4. Keep soft-threshold async scheduling unchanged when
   `soft <= pending < hard`.
5. Comparison is **`>=` hard** everywhere (append and schedule). A batch of
   exactly `hard` forces sync on that add.

## Q&A

### Q1. Soft sync vs hard persist when hitting the cap?

**Answer:** Soft sync only. The goal is to shrink the search pending tail, not
to rewrite `pages.bin`. Hard persist stays on explicit persist / Drop.
Soft sync clears all pending (`pending → 0`), not merely to just under hard.

### Q2. Default hard cap value?

**Answer:** `1000` (`4 × DEFAULT_PENDING_FLUSH_THRESHOLD`). Validate
`pending_hard_flush_threshold >= pending_flush_threshold`. Explicit saves that
set `hard < soft` are rejected.

### Q3. Enforce on search as well?

**Answer:** No. Enforce on **append** and **schedule** so `add` latency absorbs
the catch-up and queries keep using the existing pending path without a hidden
sync. Search-time sync would surprise callers who opted into
`defer_index_sync_for_search`.

### Q4. Does `add`’s existing `wait_for_flush` make E redundant?

**Answer:** No. It prevents pending growth *while a job runs*, but does not
cap a single large append or unbounded growth when schedule is never called.
E closes those cases.

### Q5. Migration when an existing TOML has soft > 1000 and no hard key?

**Answer:** Do **not** let serde’s field default (`1000`) + `hard >= soft`
validation trigger `Config::load`’s full-file fallback (which would silently
reset soft to 250). On load / `BpannConfig` construction, if the hard key is
absent, set
`pending_hard_flush_threshold = max(DEFAULT_PENDING_HARD_FLUSH_THRESHOLD, pending_flush_threshold)`.
Fresh writes still emit an explicit `pending_hard_flush_threshold = …` line.
Explicit `hard < soft` in a file remains invalid (save rejects; load falls back
to defaults as today).

### Q6. Builder setters that only bump soft?

**Answer:** Keep `hard >= soft` on the runtime path too:
`with_pending_flush_threshold` raises hard to at least the new soft when needed;
`with_pending_hard_flush_threshold` rejects (or clamps per local convention)
`hard < soft`. `new_empty_with_flush_threshold` / `apply_config_flush_threshold`
apply both soft and hard from `current_tuning()` (or the explicit args).

## Plan

### Phase 1 — Config + tuning

- [ ] Add `pending_hard_flush_threshold: usize` to `BpannTuning`
      (`bpann/src/tuning.rs`) with default `1000` and constant
      `DEFAULT_PENDING_HARD_FLUSH_THRESHOLD`.
- [ ] Validate: `>= 1` and `>= pending_flush_threshold`.
- [ ] Mirror through `BpannConfig` / `From` / `to_tuning` / accessor /
      default TOML in `ennbo/src/file_config.rs`.
- [ ] Migration: missing hard key →
      `max(DEFAULT_PENDING_HARD_FLUSH_THRESHOLD, soft)` (see Q5); do not
      full-default-fallback solely because hard was absent.
- [ ] Update `tests/test_ennbo_config.py` and Rust config/tuning unit tests
      (defaults written; explicit `hard < soft` rejected; missing hard +
      soft=2000 loads with hard=2000, soft preserved).

**Validation:**

- From `rust/`: `cargo test -p bpann tuning`
- From `rust/`: `cargo test -p ennbo file_config`
- `PYTHONPATH=src pytest tests/test_ennbo_config.py -q`
- Fresh config contains `pending_hard_flush_threshold = 1000`.

### Phase 2 — Enforce on append and schedule

- [ ] Thread hard threshold onto `BpannBackend` (constructor /
      `with_pending_hard_flush_threshold` / apply soft+hard from
      `current_tuning()` in `disk_bpann/enn_backend.rs`
      `apply_config_flush_threshold`).
- [ ] Enforce `hard >= soft` on builder setters (Q6).
- [ ] After successful `append_rows` on the deferred path: if
      `pending_rows() >= hard`, run soft sync immediately (exclusive
      `ensure_index_sync` / publish on the caller; `model.add` already
      `wait_for_flush` before append).
- [ ] In `DiskBackendHandle::schedule_background_flush`: if
      `pending >= hard`, `wait_for_flush` then soft-sync under write lock and
      return; else keep today’s async soft schedule when `pending >= soft`.
- [ ] Do not change hard-persist behavior.

**Validation:**

- Unit test: soft `2`, hard `5`; append 4 rows → pending stays, no forced sync;
  append more to cross hard → after add, `pending_unindexed_count() == 0`
  and search no longer brute-forces that tail alone.
- Unit test: soft `2`, hard `5`; append past soft, `schedule_background_flush`
  with barrier still async; append past hard without schedule still syncs on add.
- Unit test: `schedule` with `pending >= hard` completes soft sync before return
  (no in-flight job left).
- Unit test: `with_pending_flush_threshold(2000)` leaves `hard >= 2000`.
- `cargo test -p bpann`
- `cargo test -p ennbo` (incl. `flush_controller` concurrent-search tests).

### Phase 3 — Integration / stress sanity

- [ ] Comparison is `>=` hard (documented in Requested Changes). A batch of
      exactly hard forces sync on that add — including default
      `ENN_ADD_STRESS_BATCH_SIZE=1000` with default hard=1000 under `--batch`.
- [ ] Separately confirm async soft schedule when
      `soft <= pending < hard` (e.g. soft=2, hard=5, or stress without
      `--batch` / smaller batches so pending stays under hard).
- [ ] Existing disk persist / reopen tests still pass.

**Validation:**

- `PYTHONPATH=src pytest tests/test_enn_index_driver.py tests/test_ennbo_config.py -q`
- Async-path smoke (pending stays under hard): soft=250, hard=1000, **no**
  `--batch` (row adds + `schedule`), or an explicit smaller batch:
  `./ops/stress.py enn bpann_disk 5000 --num-dim=100 --work-dir=/tmp/enn_hard_cap_async --heartbeat-seconds=0`
  completes with soft schedule still async for ordinary adds.
- Hard-cap smoke: soft=250, hard=1000, `--batch` (batch=1000 → sync on each add);
  plus a forced huge single add above hard — `add` time includes soft sync and
  subsequent query does not scan a multi-thousand pending tail:
  `./ops/stress.py enn bpann_disk 5000 --num-dim=100 --batch --work-dir=/tmp/enn_hard_cap --heartbeat-seconds=0`
