"""Build data files and the table for metric_12d.tex from run1.out..run3.out.

Each run*.out is the output of `./ops/evaluate.py run short/metric_12d`.
Usage: python reports/metric_12d/make_metric_12d.py
"""

import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LINE = re.compile(
    r"model = (\w+) n = (\d+) LARGER\(loglik\) = (\S+) SMALLER\(nrmse\) = (\S+) "
    r"SMALLER\(add_s\) = (\S+) SMALLER\(query_s\) = (\S+)"
)
MODELS = [
    ("flat", "FLAT, \\texttt{NONE}"),
    ("flat_scale_x", "FLAT, \\texttt{SCALE\\_X}"),
    ("bpann_disk", "BPANN\\_DISK, \\texttt{NONE}"),
    ("bpann_disk_metric_learning", "BPANN\\_DISK, \\texttt{METRIC\\_LEARNING}"),
]


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(HERE, "run*.out"))):
        rows = {}
        with open(path) as f:
            for m in LINE.finditer(f.read()):
                model, n, *vals = m.groups()
                rows[(model, int(n))] = [float(v) for v in vals]
        runs.append(rows)
    return runs


def write_data_files(runs, keys):
    for model, _ in MODELS:
        ns = sorted(n for m, n in keys if m == model)
        with open(os.path.join(HERE, f"d_{model}.dat"), "w") as f:
            f.write("n loglik nrmse add_med add_lo add_hi q_med q_lo q_hi\n")
            for n in ns:
                v = np.array([r[(model, n)] for r in runs])
                add, q = v[:, 2], v[:, 3]
                f.write(
                    f"{n} {v[0, 0]:.4f} {v[0, 1]:.4f} "
                    f"{np.median(add):.4f} {add.min():.4f} {add.max():.4f} "
                    f"{np.median(q):.4f} {q.min():.4f} {q.max():.4f}\n"
                )


def write_table(runs, keys):
    n_max = max(n for _, n in keys)
    lines = []
    for model, label in MODELS:
        v = np.array([r[(model, n_max)] for r in runs])
        total_add = np.array([sum(r[(model, n)][2] for m, n in keys if m == model) for r in runs])
        lines.append(
            f"{label} & ${v[0, 0]:+.2f}$ & ${v[0, 1]:.2f}$ & "
            f"${np.median(total_add):.1f}$ [{total_add.min():.1f}, {total_add.max():.1f}] & "
            f"${np.median(v[:, 3]):.2f}$ [{v[:, 3].min():.2f}, {v[:, 3].max():.2f}]\\\\"
        )
    with open(os.path.join(HERE, "t_final.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    runs = load_runs()
    keys = sorted(runs[0], key=lambda k: (k[0], k[1]))
    acc_diff = max(abs(r[k][i] - runs[0][k][i]) for r in runs for k in keys for i in (0, 1))
    print(f"runs={len(runs)} max loglik/nrmse difference across runs = {acc_diff:.4g}")
    write_data_files(runs, keys)
    write_table(runs, keys)


if __name__ == "__main__":
    main()
