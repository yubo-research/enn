# Plan: `turbo-enn --tell-all` (per-observation tell)

## User Request

For small \(N\), `stress.py turbo-enn` should be able to call `tell()` for each individual observation. Add a CLI option `--tell-all` to `./ops/stress.py turbo-enn`.

## Current State

### `turbo-enn` loop

- CLI: `turbo-enn {flat|bpann_disk} NUM_OBS NUM_ASK [--num-dim] [--work-dir]` in `ops/stress.py`.
- `run_turbo_enn_stress` walks exponentially spaced stops from `turbo_enn_ask_stops(num_obs, num_ask)`.
- Between stops it bulk-seeds the gap with `seed_turbo_enn_to_n(..., chunk=seed_chunk)`, then times `ask(1)` (arms discarded).
- Default `seed_chunk` is large (`TURBO_ENN_SEED_CHUNK = 100_000` for disk, `TURBO_ENN_SEED_CHUNK_FLAT = 200_000`), so small gaps become **one multi-row `tell`**.

### Seeding helper

```1165:1197:ops/stress.py
def seed_turbo_enn_to_n(..., chunk: int = PROPOSAL_SCALE_SEED_CHUNK) -> None:
    """Bulk-seed optimizer with N synthetic Ackley points via chunked ``tell``."""
    ...
    for start in range(0, n, chunk):
        ...
        opt.tell(x, y)
```

- `chunk=1` already means one row per `tell` (no new tell API needed).
- `run_turbo_enn_stress` accepts `seed_chunk=` but the CLI does not expose it.

### Tests

- `tests/test_ops_stress_turbo_enn.py` asserts default behavior: tell batch sizes equal stop gaps (`test_run_turbo_enn_stress_dgp_tell_then_ask_ignore`, `test_turbo_enn_gaps_sum_to_num_obs`).
- CLI mocked tests do not pass a tell-all flag.

### Adjacent

- `proposal-scale` also uses `seed_turbo_enn_to_n` with `PROPOSAL_SCALE_SEED_CHUNK = 1000`. Out of scope unless this plan is extended; user asked only for `turbo-enn`.

## Requested Changes

1. Add `--tell-all` flag to `turbo-enn` (boolean / is_flag).
2. When set, every DGP observation is delivered with its own `tell()` (one row per call).
3. Reflect the mode in the config header so runs are distinguishable.
4. Cover with unit/CLI tests; keep default (batched gap tell) unchanged.

## Q&A

### Q1. Implement as `seed_chunk=1`, or a separate code path?

**Answer:** Public API is **`tell_all`**. Mechanism is still `seed_chunk=1` via existing `seed_turbo_enn_to_n` (no separate seed loop). CLI and tests pass `tell_all=True`; the runner applies that by forcing `seed_chunk = 1` internally.

### Q2. Change `proposal-scale` too?

**Answer:** **No.** Only `turbo-enn` CLI + `run_turbo_enn_stress` wiring for this flag.

### Q3. Header / default?

**Answer:** Default remains batched (`--tell-all` off). When on, append `tell_all=true` to `format_turbo_enn_config_header` output.

### Q4. Interaction with large `NUM_OBS`?

**Answer:** Flag is always legal. Document in help that it is intended for small-\(N\) fidelity (per-obs tell/fit/sync cost); large \(N\) will be much slower by design.

## Plan

### Phase 1 — Flag + runner wiring

- [x] Add `tell_all: bool = False` to `run_turbo_enn_stress`. Resolve `seed_chunk` in this order:
  1. if `seed_chunk is None`, set driver default via `turbo_enn_default_seed_chunk`;
  2. if `tell_all`, set `seed_chunk = 1` (wins over any explicit `seed_chunk`);
  3. then validate `seed_chunk >= 1`.
- [x] Extend `format_turbo_enn_config_header` with `tell_all: bool = False`; when true, append ` tell_all=true` (omit when false, same pattern as `work_dir`).
- [x] Add Click `click.Option(["--tell-all"], is_flag=True, default=False, help=...)` to the `turbo-enn` `params=[...]` list and accept `tell_all` on `def turbo_enn(...)`.
- [x] Pass `tell_all` into header formatting and `run_turbo_enn_stress` (do **not** pass `seed_chunk=1` from the CLI; let the runner apply it).

**Validation:**

```bash
./ops/stress.py turbo-enn flat 10 3 --num-dim=2
# header without tell_all; existing row count

./ops/stress.py turbo-enn flat 10 3 --num-dim=2 --tell-all
# header contains tell_all=true; 3 data rows still
```

### Phase 2 — Tests

- [x] Unit test: `run_turbo_enn_stress(..., tell_all=True)` with mock optimizer → every `tell` has `x.shape[0] == 1`, and number of tells equals `num_obs`.
- [x] Keep existing gap-batch tests as the default (`tell_all` false) path.
- [x] CLI mocked test: invoke with `--tell-all`, assert fake_run received `tell_all=True` and header includes `tell_all=true`.

**Validation:**

```bash
pytest -sv tests/test_ops_stress_turbo_enn.py -k 'tell_all or turbo_enn'
```
