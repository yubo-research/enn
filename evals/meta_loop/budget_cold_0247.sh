#!/bin/bash
# Usage: budget_cold_0247.sh OUTDIR   (repo root; report 0247) cold cost to N=1e6, D=10, 1 GB, leaf/internal 30/10 vs 100/100.
O=$1
ENN_PBUF=0 LEAF_COEF=30 INTERNAL_COEF=10 evals/meta_loop/cold_cost.sh $O/c30 10 1e6 1G 3000
ENN_PBUF=0 LEAF_COEF=100 INTERNAL_COEF=100 evals/meta_loop/cold_cost.sh $O/c100 10 1e6 1G 3000
