import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    D = {
        10: {
            "nearest": 575,
            "update": 448,
            "push": 265,
            "split_compute": 104 + 15 + 47,
            "split_alloc": 234,
            "split_write": 892 - 104 - 15 - 47 - 234,
        },
        100: {
            "nearest": 917,
            "update": 475,
            "push": 1303,
            "split_compute": 206 + 58 + 103,
            "split_alloc": 229,
            "split_write": 2576 - 206 - 58 - 103 - 229,
        },
    }
    cols = ["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#E45756", "#9D755D"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (d, parts) in enumerate(D.items()):
        b = 0
        for (k, v), c in zip(parts.items(), cols):
            ax.bar(i, v, bottom=b, color=c, label=k if i == 0 else None, width=0.6)
            b += v
        ax.text(i, b + 60, f"{b / 1000:.1f} us", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["D=10", "D=100"])
    ax.set_ylabel("ns per added row")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig("reports/report-0233-data/fig.pdf")


if __name__ == "__main__":
    main()
