_CLI_DESCRIPTION = "Render measured CSV results as a two-column LaTeX report and PDF figure.\n\nPYTHONPATH=src:. python evals/long/semantic_output_report.py\ntectonic docs/semantic_output_sharing.tex\n"

import argparse
import csv
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENARIOS = ("A_aligned", "B_independent", "C_opposing", "D_misleading")
METHODS = ("independent", "semantic", "shuffled", "random")


def mean_se(rows, metric):
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(float(row[metric]))
    values = np.array([np.mean(group) for group in by_seed.values()])
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))


def paired(rows):
    keys = ("dimensions", "outputs", "n", "seed", "scenario")
    baseline = {
        tuple(row[k] for k in keys): row
        for row in rows
        if row["method"] == "independent"
    }
    result = []
    for row in rows:
        record = row.copy()
        reference = baseline[tuple(row[k] for k in keys)]
        record["delta"] = float(row["nrmse"]) - float(reference["nrmse"])
        result.append(record)
    return result


def formatted(rows, metric, digits=4):
    mean, error = mean_se(rows, metric)
    return rf"${mean:.{digits}f} \pm {error:.{digits}f}$"


def metric_table(rows):
    lines = []
    for scenario in SCENARIOS:
        for method in METHODS:
            subset = [
                r for r in rows if r["scenario"] == scenario and r["method"] == method
            ]
            values = [
                formatted(subset, metric)
                for metric in ("nrmse", "loglik", "coverage95")
            ]
            lines.append(" & ".join([scenario[0], method, *values]) + r" \\")
    return "\n".join(lines)


def delta_table(rows):
    lines = []
    for outputs in sorted({int(r["outputs"]) for r in rows}):
        for count in sorted({int(r["n"]) for r in rows}):
            values = []
            for scenario in SCENARIOS:
                selected = [
                    r
                    for r in rows
                    if int(r["outputs"]) == outputs
                    and int(r["n"]) == count
                    and r["scenario"] == scenario
                    and r["method"] == "semantic"
                ]
                values.append(formatted(selected, "delta"))
            lines.append(" & ".join([str(outputs), str(count), *values]) + r" \\")
    return "\n".join(lines)


def runtime_table(rows):
    lines = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        values = [
            formatted(selected, metric, 3)
            for metric in ("selection_seconds", "fit_seconds", "prediction_seconds")
        ]
        memory = np.mean([float(r["python_peak_bytes"]) for r in selected]) / 1024
        lines.append(" & ".join([method, *values, f"{memory:.1f}"]) + r" \\")
    return "\n".join(lines)


def plot_deltas(rows, path):
    figure, axis = plt.subplots(figsize=(7, 2.4), layout="constrained")
    positions = np.arange(len(SCENARIOS))
    for offset, method in zip((-0.15, 0, 0.15), METHODS[1:]):
        estimates = [
            mean_se(
                [r for r in rows if r["scenario"] == case and r["method"] == method],
                "delta",
            )
            for case in SCENARIOS
        ]
        means, errors = np.array(estimates).T
        axis.errorbar(
            positions + offset, means, yerr=errors, fmt="o", capsize=3, label=method
        )
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(
        positions, ["A: aligned", "B: independent", "C: opposing", "D: misleading"]
    )
    axis.set_ylabel("Change in NRMSE")
    axis.legend(frameon=False, ncols=3)
    figure.savefig(path)
    plt.close(figure)


PREAMBLE = r"""\documentclass[10pt,twocolumn]{article}
\usepackage[margin=0.65in]{geometry}
\usepackage{amsmath,booktabs,graphicx,hyperref,balance}
\renewcommand{\dbltopfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\renewcommand{\dblfloatpagefraction}{0.85}
\setcounter{dbltopnumber}{4}
\title{Semantic Output Sharing in Epistemic Nearest Neighbors}
\author{ENN numerical experiment report}
\date{28 October 2026}
\begin{document}
\raggedbottom
\maketitle
\begin{abstract}
A Python wrapper constructs a fixed, uncentered semantic output subspace and
blends latent ENN predictions with independent output predictions. Marginal
uncertainty is calibrated on row-wise out-of-fold predictions. This report
records synthetic experiments, including shuffled and random embedding controls.
The construction is optional; ENNBO and the Rust core are unchanged.
@@RESULT@@
\end{abstract}
\section{Model and calibration}
Sentence descriptions are embedded with the existing CPU
\texttt{all-MiniLM-L6-v2} encoder. Missing descriptions fall back to metric names.
Normalized embedding rows form $E$. The leading numerically nonzero left singular
vectors of the uncentered SVD form $A$, with $A^TA=I$ and rank $k<m$.
Embedding and loading arrays can be cached and reused without an encoder.
For training means $b$ and scales $s$, let $Z=(Y-b)/s$.
Existing ENNs predict each independent output and each column of $H=ZA$;
the existing multi-output API batches these scalar predictions and reuses
neighbor searches. The returned mean is
\[
 \hat Y=b+s\odot[(1-\alpha)\hat Z+\alpha\hat G A^T].
\]
Independent predictions permit departures from the prior. The blend is not an
explicit fitted residual model. Zero embeddings give no latent factors;
constant training outputs retain their training means. Nonfinite training
observations are rejected.

Let $u_i$ be the independent posterior standard deviation and $v_j$ the latent
standard deviation, in standardized units. A conservative raw envelope is
\[
 t_i=(1-\alpha)u_i+\alpha\sum_j |A_{ij}|v_j.
\]
This uses the triangle inequality and does not assume independent errors among
components or latent factors. Each row is held out as a whole in one of three
folds; every fold refits means, scales, and ENNs. In original units, an OOF
raw variance $q_{ni}$ is floored at $\max(10^{-6}s_i,10^{-12})^2$ and calibrated by
\[
 c_i=\frac1N\sum_n\frac{(y_{ni}-\hat y^{\rm OOF}_{ni})^2}{q_{ni}}.
\]
A tiny numerical lower bound on $c_i$ keeps variances positive. The final model
refits all supplied rows and returns variance $c_i q_i(x)$. These are marginal
Gaussian estimates, with no joint covariance or independent-error assumption.
The envelope need not remain a bound after calibration, and held-out coverage
must still be measured.

\section{Experimental protocol}
Case A generates signals from the rank-two semantic loading subspace, with
fixed row normalization. It is an oracle-aligned favorable control, not evidence
that language identifies real output relationships. Case B uses separate random
functions for outputs. Case C flips signs within repeated description groups,
so identical descriptions can correspond to opposing responses. Case D permutes
true loading rows without changing descriptions, avoiding a chance-identical
relationship matrix. Every case adds independent Gaussian noise of SD 0.1.
Inputs are uniform on $[-1,1]^d$; signals combine sine and cosine of random linear
projections. Descriptions repeat four physical metric types: temperature,
pressure, voltage, and motor speed.

The full grid uses $d\in\{20,50,100\}$, $m\in\{5,20,60\}$, and training counts
$N\in\{50,100,200,500\}$. Seeds are 0, 1, and 2. Each method receives identical
training, validation, and test rows for a given setting and seed. Validation
contains $\max(30,\lfloor N/3\rfloor)$ extra observations; test contains 200.
Candidates use $k\in\{1,2,4,8\}$ below $m$ and
$\alpha\in\{0,0.1,0.3,1\}$; numerical rank can reduce $k$.
Validation minimizes mean per-output NRMSE, with $\alpha=0$ available to all
methods. Independent ENN has only that candidate. ENN parameters are fixed at
8 neighbors, epistemic scale 1, and aleatoric scale 0. After selection, the model
refits training plus validation and recalibrates, without seeing test labels.
Thus $N$ excludes validation rows used in the final refit.

NRMSE divides each output RMSE by its final-training SD, then averages outputs.
LogLik is mean Gaussian log density in the same standardized units, using the
calibrated variance. Coverage is the fraction inside marginal 95\% Gaussian
intervals. Tables average settings within each seed, then give mean $\pm$ one
standard error across three seeds; outputs are not treated as independent
repetitions. Three seeds provide limited evidence about sampling uncertainty.

\section{Measured results}
"""


ENDING = r"""
\section{Interpretation and limits}
The optional wrapper and controls permit testing the prior without changing
optimization behavior. Validation can choose the independent baseline when
sharing is harmful, but finite validation data provide no guarantee of doing so.
Descriptions encode a fixed subspace rather than a signed correlation matrix;
semantic similarity alone cannot determine whether two outputs respond with
opposite signs. Calibration is empirical and marginal, not a guarantee of
nominal conditional or joint coverage.

Hypothesis: when all outputs use the same target-independent linear ENN
prediction weights, shared means reduce to projecting independent means through
$AA^T$. In that regime the blend acts as output regularization; it cannot add
new information about the input response. A unit test checks this identity on a
small reproducible example. Real tasks, more seeds, alternate descriptions,
and explicit residual models would be needed before broader integration.
No ENNBO integration is justified solely by these synthetic controls.

\section{Reproduction and artifacts}
\balance
\raggedright
From the repository root, set \texttt{OPENBLAS\_NUM\_THREADS=1},
\texttt{OMP\_NUM\_THREADS=1}, \texttt{RAYON\_NUM\_THREADS=1}, and
\texttt{PYTHONPATH=src:.}; run
\texttt{python evals/long/eval\_semantic\_outputs.py --full}
with \texttt{--directory \_reports/semantic\_outputs\_full}.
Cached pretrained embeddings and projections are retained in that directory.
\texttt{raw.csv}, \texttt{per\_output.csv}, and \texttt{summary.csv} contain the
measurements and seed-level summaries. Run
\texttt{python evals/long/semantic\_output\_report.py}, then
\texttt{tectonic docs/semantic\_output\_sharing.tex} to rebuild this report.
See \texttt{docs/semantic\_outputs.md} for the API, edge cases, and pilot command.
Timing excludes encoder startup, downloads, and the separate allocation pass.
The full benchmark and cached projections accompany this report locally.
\end{document}
"""


def _validate_grid(rows):
    keys = ("dimensions", "outputs", "n", "seed", "scenario", "method")
    expected = set(
        product(
            ("20", "50", "100"),
            ("5", "20", "60"),
            ("50", "100", "200", "500"),
            ("0", "1", "2"),
            SCENARIOS,
            METHODS,
        )
    )
    actual = [tuple(row[key] for key in keys) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("report requires the complete three-seed full-grid benchmark")


def render(directory, destination):
    with (directory / "raw.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    _validate_grid(rows)
    deltas = paired(rows)
    figure = destination.with_name("semantic_output_deltas.pdf")
    plot_deltas(deltas, figure)
    aligned = [
        r for r in deltas if r["scenario"] == "A_aligned" and r["method"] == "semantic"
    ]
    peak = max(float(r["process_peak_rss_bytes"]) for r in rows) / 2**20
    summary = (
        f"The run contains {len(rows)} selected-model evaluations. "
        "The semantic-minus-independent NRMSE change in case A is "
        + formatted(aligned, "delta", 5)
        + ". Negative changes favor sharing.\n"
    )
    fallback = []
    for scenario in SCENARIOS[1:]:
        selected = [
            r for r in rows if r["scenario"] == scenario and r["method"] == "semantic"
        ]
        fraction = np.mean([float(r["alpha"]) == 0 for r in selected])
        fallback.append(f"{scenario[0]}: {100 * fraction:.1f}\\%")
    summary += (
        "Validation selected zero sharing in " + ", ".join(fallback) + " of settings.\n"
    )
    summary += "This run does not establish a predictive accuracy benefit in the favorable case.\n"
    tables = (
        r"""\begin{table*}[t]\centering\small
\begin{tabular}{llrrr}\toprule
Case & Method & NRMSE & Gaussian LogLik & 95\% coverage\\\midrule
"""
        + metric_table(rows)
        + r"""
\bottomrule\end{tabular}
\caption{Balanced full-grid averages, with across-seed standard errors.}
\end{table*}
\begin{figure*}[t]\centering
\includegraphics[width=0.9\textwidth]{semantic_output_deltas.pdf}
\caption{Paired changes from independent ENN, averaged across grid settings
within each seed. Error bars show one across-seed standard error.}
\end{figure*}
\begin{table*}[t]\centering\small
\begin{tabular}{rrrrrr}\toprule
Outputs & $N$ & $\Delta$ A & $\Delta$ B & $\Delta$ C & $\Delta$ D\\\midrule
"""
        + delta_table(deltas)
        + r"""
\bottomrule\end{tabular}
\caption{Semantic-minus-independent NRMSE by output count and training size,
averaged across input dimensions within each seed. Negative favors sharing.}
\end{table*}
\section{Runtime and memory}
\begin{table*}[t]\centering\small
\begin{tabular}{lrrrr}\toprule
Method & Selection (s) & Final fit (s) & Prediction (s) & Python peak (KiB)\\\midrule
"""
        + runtime_table(rows)
        + r"""
\bottomrule\end{tabular}
\caption{Mean costs across grid settings and seeds. Selection includes all
candidate fits and calibration. Final fit includes calibration and refitting;
prediction covers 200 test rows. Memory is a separate refit/prediction pass.}
\end{table*}
"""
    )
    memory = (
        f"The process peak RSS was {peak:.1f} MiB. "
        + r"""
This is a cumulative high-water mark including native allocations and encoder
startup, not a per-method increment. Python peak allocation measurements exclude
Rust, BLAS, and native index storage. Timing runs are untraced; a separate pass
measures Python allocations. Native peak memory cannot be attributed to methods
from these records. Candidate search adds cost beyond any final-model overhead.
"""
    )
    split = tables.index(r"\begin{table*}[t]", tables.index(r"\begin{figure*}"))
    early, late = tables[:split], tables[split:]
    preamble = PREAMBLE.replace(
        r"\section{Model and calibration}", early + r"\section{Model and calibration}"
    )
    abstract = (
        "The mean semantic-minus-independent NRMSE change in the favorable case is "
        + formatted(aligned, "delta", 5)
        + "."
    )
    abstract += " This run does not establish a predictive accuracy benefit."
    preamble = preamble.replace("@@RESULT@@", abstract)
    destination.write_text(preamble + summary + late + memory + ENDING)


def main():
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION)
    parser.add_argument(
        "--directory", type=Path, default=Path("_reports/semantic_outputs_full")
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("docs/semantic_output_sharing.tex")
    )
    arguments = parser.parse_args()
    render(arguments.directory, arguments.destination)


if __name__ == "__main__":
    main()
