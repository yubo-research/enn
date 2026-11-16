#!/bin/bash
# Usage: quality_0248.sh OUT   (repo root; report 0248). Buffer size R: 0 (none) / 1000 / 3000 with ENN_PBUF_ALL=1, 10 seeds,
# D=10 uniform and D=100 INTRINSIC=5, N=1e4 1e5, library-default search budget (LEAF=0). PP = overlay PYTHONPATH.
O=$1; mkdir -p $O/d10 $O/d100i5
ENN_PBUF_ALL=1 LEAF=0 ARMS="0 1000 3000" bash "$(dirname "$0")"/pbuf_quality.sh $O/d10 10 "1e4 1e5" 10
ENN_PBUF_ALL=1 LEAF=0 INTRINSIC=5 ARMS="0 1000 3000" bash "$(dirname "$0")"/pbuf_quality.sh $O/d100i5 100 "1e4 1e5" 10
