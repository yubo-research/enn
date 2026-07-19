# Plan: Tier-1 `proposal-scale` scout for TuRBO-ENN timing

## User Request

Implement Tier 1 from the chat: a fast approximation of the full-optimization BPANN “mean proposal time vs N” curves that runs in minutes instead of hours.

Target recipe:

1. At each log-spaced N, bulk-seed a TuRBO-ENN optimizer to N observations (not 1→N ask growth).
2. Time K warm `ask(1)` + `tell` rounds with no eval sleep.
3. Emit per-N mean `ask_s` / `tell_s` / proposal (`ask+tell`).
4. Default N grid covers up to 1e5 so scouts can match the Modal figure’s x-axis.

## Current State

### Full-opt charts (slow path)

- Produced via yubo `enn_incremental_batches` `full_optimization`: cumulative designer `tell+ask` wall, plotted as Δt/ΔN at checkpoints up to N≈1e5, multi-function × multi-rep — hours for BPANN disk.

### Existing stress harness (`ops/stress.py`)

- Commands: `enn`, `sample`, `draw`, `turbo-enn`.
- `turbo-enn` (`run_turbo_enn_stress`): sequential rounds from empty opt; each round `ask(1)` → Ackley → `sleep(0.1)` → `tell`; prints every round. Good for microbench, **bad** as a 1e5 scaling scout (pays N asks + sleep).
- Shared helpers already usable for Tier 1:
  - `build_turbo_enn_optimizer_config` — UCB / k=10 / `num_fit_samples=100` / noise-aware TR; `bpann_disk` requires `enn_storage="disk"` + `work_dir`.
  - `parse_index_driver`, work-dir Click rules (`DISK_INDEX_TYPE_CHOICES`).
  - `checkpoint_ns` / `_next_checkpoint` — log-ish grid 1,3,10,30,… (usable or replace with an explicit proposal-scale grid).
- Tests: `tests/test_ops_stress_turbo_enn.py` (config, formatting, mocked call order, CliRunner).

### Feasibility probe (this branch)

- One bulk `tell(x,y)` with N=300 synthetic Ackley points succeeds; `init_progress` clears when N ≥ `num_init`; subsequent `ask(1)`/`tell` are on the turbo path.
- So “seed then probe” does not require N timed asks.

### Adjacent / non-goals

- Tier 1 is a **scout**, not a replacement for Modal full_opt figures.
- Do not use `enn` stress `query_s` (unsynced posterior) as the y-metric.
- No change to TuRBO sync policy or production BPANN defaults in this work.

## Requested Changes

1. Add `ops/stress.py` command `proposal-scale` that times TuRBO-ENN proposal cost at log-spaced N.
2. For each N: fresh optimizer → seed to N → warmup → K probe rounds → one summary row (mean ask / tell / proposal).
3. Support `{flat|bpann_disk}` with the same `--work-dir` rules as `turbo-enn`.
4. No eval sleep; defaults suitable for a minutes-scale run up to `--max-n=100000`.
5. Add CLI/unit tests (mocked optimizer for order/aggregates; small live flat path optional only if already cheap like other stress tests).

## Q&A

### Q1. Command name and CLI shape?

**Answer:** `proposal-scale`, mirroring other stress verbs:

```text
./ops/stress.py proposal-scale --num-dim=10 --max-n=100000 --num-probes=30 \
  {flat|bpann_disk} [--work-dir=DIR]
```

### Q2. Fresh optimizer per N, or grow one run across the grid?

**Answer:** **Fresh per N** (subdirectory `work_dir/n{N}` for disk). Isolates each point; matches “cost at size N” reading of the Modal curves. Growing one session is faster wall-clock but couples RNG/index history across N.

### Q3. How to seed to N without N asks?

**Answer:** Draw N points uniform in Ackley bounds, evaluate Ackley, `tell` in chunks of `PROPOSAL_SCALE_SEED_CHUNK` (default **1000**; last chunk shorter). Chunked tell keeps soft-sync/fragment behavior closer to streaming than a single mega-`tell(N)`, while staying ≪ N asks. Use `num_init=TURBO_ENN_NUM_INIT` so a seed with N≥10 leaves hybrid init. Require `max_n >= TURBO_ENN_NUM_INIT` and every grid N ≥ `TURBO_ENN_NUM_INIT` (drop smaller checkpoints).

### Q4. Default N grid?

**Answer:** Fixed tuple filtered by `--max-n`:

`(10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)` ∩ `{n | n <= max_n}`.

Do not reuse `checkpoint_ns` (includes 1 and 3, below init).

### Q5. Warmup and aggregation?

**Answer:** Discard **2** warmup ask/tell rounds (untimed). Then time `num_probes` rounds (default **30**). Print **means** of `ask_s`, `tell_s`, and `proposal_s = ask_s + tell_s` (no sleep in the sum). Header once; one data row per N.

### Q6. Sleep / Ackley noise / hyperparams?

**Answer:** `sleep` = no-op (hardcode 0; no CLI flag). Reuse Ackley noise and `build_turbo_enn_optimizer_config` defaults from `turbo-enn` so scout hyperparams match the existing stress TuRBO path.

## Plan

### Phase 1 — Runner + CLI

- [x] Add constants in `ops/stress.py`, e.g. `PROPOSAL_SCALE_NS`, `PROPOSAL_SCALE_NUM_PROBES = 30`, `PROPOSAL_SCALE_WARMUP = 2`, `PROPOSAL_SCALE_SEED_CHUNK = 1000`, reuse `TURBO_ENN_*` for Ackley/config.
- [x] Add helpers:
  - `proposal_scale_ns(max_n) -> tuple[int, ...]`
  - `seed_turbo_enn_to_n(opt, objective, bounds, n, *, rng, chunk)` — chunked synthetic tell
  - `probe_turbo_enn_proposal(opt, objective, *, warmup, num_probes) -> (ask_mean, tell_mean, proposal_mean)`
  - `run_proposal_scale_stress(...)` — iterate N grid; for disk use `work_dir/n{N}`; yield per-N result
- [x] Add Click command `proposal-scale`:
  - Args: `index_type`, optional none beyond options
  - Options: `--num-dim` (default 10), `--max-n` (default 100000), `--num-probes` (default 30), `--work-dir`
  - Validate: `num_dim >= 1`, `max_n >= TURBO_ENN_NUM_INIT`, `num_probes >= 1`, work-dir rules like `turbo-enn`
  - Header: `num_dim=… max_n=… num_probes=… index_type=… work_dir=…`
  - Rows: `N ask_s tell_s proposal_s` with formatting consistent with other stress floats (3 decimals)

**Validation:**

```bash
# minutes-scale scout (cap N for smoke)
rm -rf _enn_ps; RAYON_NUM_THREADS=1 ./ops/stress.py proposal-scale \
  --num-dim=10 --max-n=1000 --num-probes=10 bpann_disk --work-dir=_enn_ps
# header + rows for N in {10,30,100,300,1000}; bpann ask_s >> flat ask_s at same N

./ops/stress.py proposal-scale --max-n=1000 flat
# no --work-dir; succeeds

./ops/stress.py proposal-scale bpann_disk --max-n=1000   # fails: missing --work-dir
./ops/stress.py proposal-scale flat --work-dir=/tmp/x --max-n=1000  # fails
./ops/stress.py proposal-scale --max-n=5 flat  # fails: max_n < num_init
```

### Phase 2 — Tests

- [x] Add `tests/test_ops_stress_proposal_scale.py` (or extend turbo-enn test module):
  - Unit: `proposal_scale_ns(1000)` equals `(10, 30, 100, 300, 1000)`; empty/invalid `max_n` errors.
  - Unit: `probe_turbo_enn_proposal` with mocked opt — warmup untimed, then `num_probes` timed; means match fixtures; call order ask→tell.
  - Unit: `seed_turbo_enn_to_n` tells chunked sizes that sum to N (mock tell recording shapes).
  - CLI: CliRunner `flat --max-n=30 --num-probes=2` with monkeypatched runner returning fixed means — exit 0, header, expected N rows.
  - CLI: reject disk without work-dir; flat with work-dir; `max_n < 10`.

**Validation:**

```bash
pytest -sv tests/test_ops_stress_proposal_scale.py -k proposal_scale
```
