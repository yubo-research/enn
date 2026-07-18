# Plan: Expose BPANN search-mode row limits in config

## User Request

(Summarized from the prior next-target decision.) Expose the duplicated
compile-time search-mode cliffs as tunable config:

- `exhaustive_search_row_limit` (default 2500)
- `skip_refinement_row_limit` (default 150_000)

Wire them through `BpannTuning` / `~/.ennbo/config.toml` with a single shared
source of truth for build and search.

## Current State

### Search-mode dispatch (hardcoded ×2)

| Constant | Value | Build use | Search use |
|---|---:|---|---|
| `EXHAUSTIVE_SEARCH_ROW_LIMIT` | 2500 | `needs_skip_edges` in `rust/crates/bpann/src/index/build.rs` | `search_index_candidates` in `rust/crates/bpann/src/index/sync.rs` |
| `SKIP_REFINEMENT_ROW_LIMIT` | 150_000 | same | same |

Runtime behavior for a fragment with `rows` indexed rows:

1. `rows <= exhaustive` → exhaustive leaf search; build stores **no** skip edges
2. `exhaustive < rows <= skip_refinement` → skip-refinement search; build **writes** skip edges
3. `rows > skip_refinement` → greedy blocks only; build stores **no** skip edges

Both files define identical private `const`s. They are not in `BpannTuning`.

### Existing tuning / config path (pattern to extend)

| Layer | File | Role |
|---|---|---|
| `BpannTuning` | `rust/crates/bpann/src/tuning.rs` | Process-wide snapshot; `current_tuning()`; `validate()` |
| `BpannConfig` | `rust/crates/ennbo/src/file_config.rs` | `[bpann]` TOML; `From`/`to_tuning`; `serde(default)` |
| Provider | `install_bpann_tuning_from_config()` | Reads config on each `current_tuning()` access |
| Already tunable | flush threshold, structured build limit, beam width, compaction/budget knobs | Wired end-to-end |

`search_beam_width` already goes through `current_tuning()` in `sync.rs`; the
row-limit cliffs do not.

### Adjacent / out of scope today

- Persist/reopen tests that use **2500 rows** as a data size
  (`persist_hardening.rs`, `disk_persist_index.rs`, `test_enn_index_driver.py`)
  are about `indexed_rows` counts, not the exhaustive cliff constant.
- `ops/tune_bpann.py` grids only compaction/budget knobs; it cannot tune these
  limits until they exist in config (follow-up, not this plan).
- Async soft sync / concurrent search work is already shipped.
- If limits change between build and search, search may take the skip-refinement
  path when `skip_edges` is empty (lookups are optional — fewer candidates, no
  panic). Shared runtime config keeps build and search agreed *at each call*;
  on-disk indexes are not rewritten until rebuild/compaction.

## Requested Changes

1. Add `exhaustive_search_row_limit` and `skip_refinement_row_limit` to
   `BpannTuning` and `BpannConfig`, with defaults `2500` and `150_000`.
2. Remove the duplicated compile-time constants; build `needs_skip_edges` and
   search `search_index_candidates` both read `current_tuning()`.
3. Validate: `exhaustive_search_row_limit >= 1` and
   `skip_refinement_row_limit >= exhaustive_search_row_limit`.
4. Update default-config and validation tests so new keys appear in generated
   `config.toml` and invalid pairs are rejected.

## Q&A

### Q1. Include extending `ops/tune_bpann.py` in this plan?

**Answer:** No. Library wiring must land first; the tuner cannot grid fields
that do not exist. Tuner expansion is a separate follow-up after this change.

### Q2. Keep today’s boundary semantics (`<= exhaustive` vs `>`)?

**Answer:** Yes. Preserve current behavior at defaults:
`needs_skip_edges` uses `row_count > exhaustive && row_count <= skip_refinement`;
search uses `rows <= exhaustive` then `rows <= skip_refinement`. Do not change
cliff meaning while exposing the knobs.

### Q3. What happens to existing user `config.toml` files missing the new keys?

**Answer:** `#[serde(default)]` on `BpannConfig` fills missing fields from
`BpannTuning::default()`, so old files keep today’s 2500 / 150_000 behavior
without migration.

### Q4. Rebuild on-disk indexes when these config values change?

**Answer:** No. Match existing Tier 1 behavior: tuning is read at call time.
Build uses current limits when creating skip edges; search uses current limits
when choosing a path. Rebuild/reindex remains an operator concern (same as
changing beam width today). Optionally document the mismatch risk in comments
near `needs_skip_edges` / `search_index_candidates`; do not add automatic
rebuild.

## Plan

### Phase 1 — Tuning types + validation

- [ ] Add fields to `BpannTuning` in `rust/crates/bpann/src/tuning.rs` with
      defaults `exhaustive_search_row_limit: 2500`,
      `skip_refinement_row_limit: 150_000`.
- [ ] Extend `validate()`:
      - `exhaustive_search_row_limit == 0` → error
      - `skip_refinement_row_limit < exhaustive_search_row_limit` → error
- [ ] Mirror fields through `BpannConfig` `From` / `to_tuning` / accessors in
      `rust/crates/ennbo/src/file_config.rs`.
- [ ] Update `file_config` and `tuning` unit tests (defaults in written TOML,
      invalid exhaustive=0, invalid `skip < exhaustive`).
- [ ] Update `tests/test_ennbo_config.py` assertions for the new default keys.

**Validation:**

- `cd rust && RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo test -p bpann tuning`
- `cd rust && RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo test -p ennbo file_config`
- `PYTHONPATH=src pytest tests/test_ennbo_config.py -q`
- Fresh `ensure_config_file` output contains
  `exhaustive_search_row_limit = 2500` and
  `skip_refinement_row_limit = 150000` (or TOML-equivalent).

### Phase 2 — Call sites use `current_tuning()`

- [ ] Delete local `EXHAUSTIVE_SEARCH_ROW_LIMIT` /
      `SKIP_REFINEMENT_ROW_LIMIT` from
      `rust/crates/bpann/src/index/build.rs` and
      `rust/crates/bpann/src/index/sync.rs`.
- [ ] Implement `needs_skip_edges` via `current_tuning()` limits (same
      inequalities as today).
- [ ] Implement `search_index_candidates` mode selection via the same tuning
      fields.
- [ ] Optionally comment near those call sites that on-disk skip edges reflect
      build-time limits (Q4).
- [ ] Add a focused test that changing tuning (test provider or
      `set_tuning_provider`) switches search/build mode at a fixed row count
      (e.g. force exhaustive above former default, or force skip-edges on/off).

**Validation:**

- `rg EXHAUSTIVE_SEARCH_ROW_LIMIT SKIP_REFINEMENT_ROW_LIMIT rust/crates/bpann`
  returns no matches.
- `cd rust && RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo test -p bpann`
- `cd rust && RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup" cargo test -p ennbo`
- Mode-switch test passes; default-path persist/search tests still pass
  (`persist_hardening`, `disk_persist_index`, kiss coverage).

### Phase 3 — Integration sanity

- [ ] Confirm `install_bpann_tuning_from_config` path picks up overrides: write
      a temp config with non-default limits, `set_config_path`, open a disk
      backend / run a small search or build that exercises `needs_skip_edges`
      consistently with search.

**Validation:**

- `PYTHONPATH=src pytest tests/test_ennbo_config.py tests/test_enn_index_driver.py -q`
- Manual: set `exhaustive_search_row_limit` high enough that a small multi-k
  tree build skips skip-edge generation; reopen/search still succeeds (empty
  skip edges degrade candidate recall, not panic).
