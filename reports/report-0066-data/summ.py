import collections
import glob
import json
import statistics as s
from pathlib import Path

d = collections.defaultdict(list)
for f in glob.glob("e2_s*.jsonl"):
    for l in Path(f).read_text().splitlines():
        r = json.loads(l)
        d[r["N"]].append(r)
for n in sorted(d):
    L = d[n]
    print(
        n,
        len(L),
        "q_ms",
        [round(r["med_ms"], 2) for r in L],
        "leaves",
        round(s.mean(r["leaves_per_q"] for r in L), 1),
        "add_us",
        [round(r["add_us_per_row"]) for r in L],
        "anon_mb",
        [round(r["rss_anon_mb"]) for r in L],
    )
