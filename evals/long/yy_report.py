"""Build tables and figures from completed y-y study CSVs; never rerun fitting."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import t

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evals.long.yy_problem import CASES, DESCRIPTIONS

ROOT = Path("_reports/yy")
LABELS = (
    "Aligned, noisy",
    "Aligned, moderate",
    "Aligned, nearly clean",
    "Opposing",
    "Unrelated groups",
    "Misleading text",
    "Correlated noise",
)
METHODS = ("semantic", "shuffled", "random", "pca", "repeated")


def estimate(values):
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return mean, half


def interval(values):
    mean, half = estimate(values)
    if mean == 0 and half == 0:
        return "$0$"
    digits = 6 if half < 0.00005 else 4
    return rf"${mean:+.{digits}f}\pm{half:.{digits}f}$"


def paired(frame):
    keys = ["case", "n", "variance_cutoff", "seed"]
    baseline = frame[frame.method == "independent"].set_index(keys)
    result = frame.copy()
    for metric in ("nrmse", "clean_nrmse", "loglik", "coverage95"):
        result["delta_" + metric] = (
            result[metric].to_numpy()
            - baseline.loc[pd.MultiIndex.from_frame(result[keys]), metric].to_numpy()
        )
    return result


def select(frame, case, method="semantic", alpha=1, n=100, variance_cutoff=0.9):
    selected = frame[
        (frame.case == case)
        & (frame.method == method)
        & (frame.n == n)
        & (frame["variance_cutoff"] == variance_cutoff)
    ]
    if not method.endswith("selected"):
        selected = selected[selected.alpha == alpha]
    return selected.sort_values("seed")


def tables(frame):
    main, controls, uncertainty, learning = [], [], [], []
    for case, label in zip(CASES, LABELS):
        baseline = select(frame, case, "independent", 0)
        full = select(frame, case)
        chosen = select(frame, case, "semantic_selected")
        main.append(
            " & ".join(
                [
                    label,
                    f"{baseline.nrmse.mean():.4f}",
                    interval(full.delta_nrmse),
                    interval(chosen.delta_nrmse),
                    f"{100 * (chosen.alpha == 0).mean():.0f}\\%",
                ]
            )
            + r" \\"
        )
        controls.append(
            " & ".join(
                [label]
                + [
                    interval(select(frame, case, method + "_selected").delta_nrmse)
                    for method in METHODS
                ]
            )
            + r" \\"
        )
        uncertainty.append(
            " & ".join(
                [
                    label,
                    f"{baseline.loglik.mean():.3f}",
                    interval(full.delta_loglik),
                    interval(chosen.delta_loglik),
                    f"{baseline.coverage95.mean():.3f}",
                    f"{full.coverage95.mean():.3f}",
                    f"{chosen.coverage95.mean():.3f}",
                ]
            )
            + r" \\"
        )
        learning.extend(learning_rows(frame, case, label))
    for name, lines in zip(
        ("main", "controls", "uncertainty", "learning"),
        (main, controls, uncertainty, learning),
    ):
        (ROOT / (name + "_table.tex")).write_text("\n".join(lines) + "\n")


def learning_rows(frame, case, label):
    lines = []
    for n in (30, 100, 300):
        full = select(frame, case, n=n)
        chosen = select(frame, case, "semantic_selected", n=n)
        lines.append(
            " & ".join(
                [
                    label,
                    str(n),
                    interval(full.delta_nrmse),
                    interval(chosen.delta_nrmse),
                    interval(full.delta_clean_nrmse),
                ]
            )
            + r" \\"
        )
    return lines


def cutoff_table(frame):
    lines = []
    for case, label in (("aligned_noisy", "Aligned, noisy"), ("opposing", "Opposing")):
        for variance_cutoff in (0.3, 0.9, 1):
            full = select(frame, case, variance_cutoff=variance_cutoff)
            chosen = select(
                frame, case, "semantic_selected", variance_cutoff=variance_cutoff
            )
            lines.append(
                " & ".join(
                    [
                        label,
                        str(variance_cutoff),
                        interval(full.delta_nrmse),
                        interval(chosen.delta_nrmse),
                    ]
                )
                + r" \\"
            )
    (ROOT / "cutoff_table.tex").write_text("\n".join(lines) + "\n")


def figure(frame):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout="constrained")
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    cases = ("aligned_noisy", "aligned_clean", "opposing", "misleading")
    for case, color in zip(cases, colors):
        selected = frame[
            (frame.case == case) & (frame.method == "semantic") & (frame.n == 100)
        ]
        grouped = [
            estimate(selected[selected.alpha == alpha].delta_nrmse)
            for alpha in (0, 0.1, 0.3, 1)
        ]
        means, half = np.array(grouped).T
        axes[0].errorbar(
            (0, 0.1, 0.3, 1),
            means,
            yerr=half,
            color=color,
            marker="o",
            capsize=2,
            label=LABELS[list(CASES).index(case)],
        )
    axes[0].set(
        xlabel=r"Sharing strength $\alpha$",
        ylabel="Change in test NRMSE",
        title="Fixed sharing: lower is better",
    )
    axes[0].axhline(0, color="0.4", linewidth=0.7)
    axes[0].legend(frameon=False, fontsize=7)
    for offset, method, color in zip(
        (-0.2, 0, 0.2), ("semantic", "shuffled", "pca"), colors
    ):
        grouped = [
            estimate(select(frame, case, method + "_selected").delta_nrmse)
            for case in CASES
        ]
        means, half = np.array(grouped).T
        axes[1].errorbar(
            means,
            np.arange(7) + offset,
            xerr=half,
            fmt="o",
            color=color,
            capsize=2,
            label=method.upper() if method == "pca" else method,
        )
    axes[1].set_yticks(np.arange(7), LABELS)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color="0.4", linewidth=0.7)
    axes[1].set(
        xlabel="Change in test NRMSE", title="Validation-selected: left is better"
    )
    axes[1].legend(frameon=False, fontsize=7)
    fig.savefig(ROOT / "effects.pdf")
    fig.savefig(ROOT / "effects.png", dpi=180)
    plt.close(fig)


def diagnostics():
    block = np.repeat(np.eye(4), 4, axis=0) / 2
    signed = block * np.tile([1, -1, 1, -1], 4)[:, None]
    records = {}
    for name in ("paraphrase", "repeated"):
        with np.load(ROOT / (name + "_projection.npz")) as data:
            a = data["loadings"]
        p = a @ a.T
        records[name] = {
            "aligned_discarded_energy": float(
                np.linalg.norm(block - p @ block) ** 2 / 4
            ),
            "opposing_retained_energy": float(np.linalg.norm(p @ signed) ** 2 / 4),
            "projector_block_error": float(np.linalg.norm(p - block @ block.T)),
        }
    (ROOT / "geometry.json").write_text(json.dumps(records, indent=2) + "\n")
    (ROOT / "geometry_text.tex").write_text(
        f"The paraphrase subspace discards {100 * records['paraphrase']['aligned_discarded_energy']:.2f}\\% "
        "of the ideal aligned group signal energy and retains "
        f"{100 * records['paraphrase']['opposing_retained_energy']:.2f}\\% "
        "of the ideal opposing signal energy. The repeated-description projector "
        f"differs from exact within-group averaging by Frobenius norm "
        + scientific(records["repeated"]["projector_block_error"])
        + ". "
        "These checks use the population group basis before finite-sample output scaling.\n"
    )


def descriptions():
    lines = [r"\begin{enumerate}\setlength{\itemsep}{2pt}"]
    lines.extend(r"\item " + text for text in DESCRIPTIONS)
    lines.append(r"\end{enumerate}")
    (ROOT / "descriptions.tex").write_text("\n".join(lines) + "\n")


def summaries(frame):
    rows = []
    for key, group in frame.groupby(
        ["case", "n", "variance_cutoff", "method", "alpha"], sort=False
    ):
        row = dict(zip(("case", "n", "variance_cutoff", "method", "alpha"), key))
        row["count"] = len(group)
        for metric in ("nrmse", "clean_nrmse", "loglik", "coverage95", "delta_nrmse"):
            mean, half = (
                estimate(group[metric])
                if len(group) > 1
                else (group[metric].mean(), np.nan)
            )
            row[metric + "_mean"], row[metric + "_ci95_half"] = mean, half
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "summary.csv", index=False)
    lines = []
    for method in ("independent", "semantic", "pca"):
        subset = frame[
            (frame.n == 100)
            & (frame.method == method)
            & (frame.alpha == (0 if method == "independent" else 1))
        ]
        lines.append(
            f"{method} & {1000 * subset.fit_seconds.mean():.2f} & "
            f"{1000 * subset.prediction_seconds.mean():.2f}" + r" \\"
        )
    (ROOT / "runtime_table.tex").write_text("\n".join(lines) + "\n")


def delta(frame, case, metric="nrmse", method="semantic"):
    return interval(select(frame, case, method)["delta_" + metric])


def result_numbers(frame):
    noisy = select(frame, "aligned_noisy", "semantic_selected")
    opposing = select(frame, "opposing", "semantic_selected")
    correlated = frame[
        (frame.case == "correlated_noise")
        & (frame.method == "repeated")
        & (frame.alpha == 1)
    ]
    record = {
        "NoisyDelta": delta(frame, "aligned_noisy"),
        "NoisyBaseline": f"{select(frame, 'aligned_noisy', 'independent', 0).nrmse.mean():.4f}",
        "NoisyCleanDelta": delta(frame, "aligned_noisy", "clean_nrmse"),
        "NoisyFullFraction": f"{100 * (noisy.alpha == 1).mean():.0f}\\%",
        "CleanDelta": delta(frame, "aligned_clean"),
        "CleanRepeatedDelta": delta(frame, "aligned_clean", method="repeated"),
        "OpposingDelta": delta(frame, "opposing"),
        "OpposingZeroFraction": f"{100 * (opposing.alpha == 0).mean():.0f}\\%",
        "OpposingPCADelta": delta(frame, "opposing", method="pca_selected"),
        "MisleadingDelta": delta(frame, "misleading"),
        "MisleadingSelectedDelta": delta(
            frame, "misleading", method="semantic_selected"
        ),
        "CorrelatedDifference": scientific(correlated.delta_nrmse.abs().max()),
    }
    return record


def scientific(value):
    coefficient, exponent = f"{value:.1e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def numbers(frame):
    record = result_numbers(frame)
    environment = json.loads((ROOT / "environment.json").read_text())
    record["PeakMemory"] = f"{environment['process_peak_rss_mib']:.1f}"
    record["PrimarySeconds"] = f"{environment['elapsed_seconds']:.1f}"
    record["LikelihoodFinding"] = (
        "For noisy aligned outputs, full sharing changes LogLik by "
        + delta(frame, "aligned_noisy", "loglik")
        + "; for opposing outputs, the change is "
        + delta(frame, "opposing", "loglik")
        + "."
    )
    (ROOT / "numbers.tex").write_text(
        "\n".join(rf"\newcommand{{\{key}}}{{{value}}}" for key, value in record.items())
        + "\n"
    )


def validate(frame, cutoffs, sizes, cases):
    keys = ["case", "n", "variance_cutoff", "seed", "method", "alpha"]
    assert not frame.duplicated(keys).any()
    assert len(frame) == len(cutoffs) * len(sizes) * len(cases) * 30 * 26
    assert set(frame.seed) == set(range(30))
    assert set(frame.n) == set(sizes)
    assert set(frame["variance_cutoff"]) == set(cutoffs)
    assert set(frame.case) == set(cases)
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    assert frame.coverage95.between(0, 1).all()
    for _, group in frame.groupby(["case", "n", "variance_cutoff", "seed"]):
        for method in METHODS:
            candidates = group[group.method == method]
            chosen = group[group.method == method + "_selected"].iloc[0]
            assert chosen.validation_nrmse == candidates.validation_nrmse.min()


def main():
    frame = pd.read_csv(ROOT / "raw.csv")
    validate(frame, [0.3, 0.9, 1], [30, 100, 300], CASES)
    frame = paired(frame)
    tables(frame)
    cutoff_table(frame)
    figure(frame)
    diagnostics()
    descriptions()
    summaries(frame)
    numbers(frame)
    print("Validated complete paired grids; generated tables, summaries, and figures.")


if __name__ == "__main__":
    main()
