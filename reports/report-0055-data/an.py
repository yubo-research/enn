import json
import sys
from pathlib import Path

import numpy as np

rows = {}
for f in sys.argv[1:]:
    for l in Path(f).read_text().splitlines():
        r = json.loads(l)
        rows.setdefault(r["n"], []).extend(r["q"])
print(
    "{:>8} {:>8} {:>7} {:>6} {:>7} {:>7} {:>8} {:>8} {:>7} {:>6}".format(
        "N",
        "med_ms",
        "leaves",
        "64lnN",
        "stop%",
        "fallbk",
        "us/leaf",
        "minflt",
        "majflt",
        "KBread",
    )
)
out = []
for n in sorted(rows):
    q = rows[n]
    dt = np.median([x["dt"] for x in q]) * 1e3
    lv = np.mean([x["w"][0] + x["w"][3] for x in q])
    print(
        "{:8d} {:8.2f} {:7.0f} {:6.0f} {:7.0f} {:7.2f} {:8.1f} {:8.0f} {:7.1f} {:6.0f}".format(
            n,
            dt,
            lv,
            64 * np.log(n),
            100 * np.mean([x["w"][2] > 0 for x in q]),
            np.mean([x["w"][7] for x in q]),
            dt * 1e3 / lv,
            np.mean([x["minflt"] for x in q]),
            np.mean([x["majflt"] for x in q]),
            np.mean([x["rbytes"] for x in q]) / 1024,
        )
    )
