#!/bin/bash
# Usage: add_phase_prof.sh -- phase timers inside Tree::insert_row (report 0233).
# Applies add_phase_prof.patch (temporary instrumentation, patch -p0 style paths are
# sync.rs/tree.rs under rust/crates/bpann/src/index), builds, runs N=1e5 at D=10,100 under a 1 GB scope, prints ns/row.
# Revert afterwards: patch -R. Never commit the instrumented build.
set -e; cd "$(dirname "$0")/../.."
S=rust/crates/bpann/src/index
patch $S/sync.rs <(awk '/^--- /{n++} n==1' evals/meta_loop/add_phase_prof.patch)
patch $S/tree.rs <(awk '/^--- /{n++} n==2' evals/meta_loop/add_phase_prof.patch)
maturin develop --release 2>&1 | tail -1
for D in 10 100; do W=$HOME/bench_work/prof_$D; rm -rf $W
  PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=1G -p MemoryHigh=1G -p CPUQuota=100% \
    python evals/meta_loop/quality_cmp.py $D 1e5 0 $W /tmp/prof_$D.jsonl 2>&1 | grep PROF | tail -3; rm -rf $W; done
