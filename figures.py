"""Draw publication figures from the pinned RTTs and recorded results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study import BANDS, WORKLOADS, load_network


EDGE = "#087669"
CLOUD = "#9d4a15"


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.04)
    fig.savefig(path.with_suffix(".png"), dpi=230, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def main(data, results, out):
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#172632",
                         "xtick.color": "#172632", "ytick.color": "#172632",
                         "legend.frameon": False, "pdf.fonttype": 42})
    pairs, _ = load_network(data)
    matrix = pd.read_csv(results / "matrix.csv")
    sensitivity = pd.read_csv(results / "sensitivity.csv")

    fig, ax = plt.subplots(figsize=(4.15, 2.25), layout="constrained")
    for key, name, color, line in [("edge_rtt", "Edge", EDGE, "-"), ("cloud_rtt", "Cloud", CLOUD, "--")]:
        values = np.sort(pairs[key])
        ax.step(values, (np.arange(len(values)) + 1) / len(values), where="post",
                linewidth=1.7, color=color, linestyle=line, label=name)
    ax.set(xscale="log", xlabel="Closest observed endpoint RTT (ms, log scale)",
           ylabel="Fraction of probes", ylim=(0, 1.035))
    ax.grid(axis="both", alpha=.23)
    ax.legend(loc="lower right")
    save(fig, out / "rtt_cdf")

    fig, ax = plt.subplots(figsize=(4.20, 2.75), layout="constrained")
    center = matrix["delta_mean_ms"].to_numpy()
    err = (center - matrix["delta_ci_low_ms"].to_numpy(), matrix["delta_ci_high_ms"].to_numpy() - center)
    ypos = np.arange(len(matrix)) + np.repeat([0, .35, .70], 3)
    for i in range(len(matrix)):
        ax.errorbar(center[i], ypos[i], xerr=np.array([[err[0][i]], [err[1][i]]]),
                    marker="o" if center[i] > 0 else "s", markersize=4, capsize=2,
                    color=EDGE if center[i] > 0 else CLOUD)
    ax.axvline(0, color="#253644", linewidth=.9, linestyle="--")
    ax.set(yticks=ypos, yticklabels=[f"{r.network} / {r.workload}" for r in matrix.itertuples()],
           xlabel="Mean cloud − edge latency (ms)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.22)
    save(fig, out / "paired_effect")

    fig, ax = plt.subplots(figsize=(4.15, 2.50), layout="constrained")
    for band, color, mark in zip(BANDS, (EDGE, "#224c8a", CLOUD), ("o", "s", "^")):
        segment = sensitivity.query('grouping == "Cloud RTT" and workload == "Heavy" and network == @band')
        ax.plot(segment["cloud_speed"], segment["mean_delta_ms"], marker=mark, markersize=3,
                color=color, label=band)
    ax.axhline(0, color="#253644", linestyle="--", linewidth=.9)
    ax.axvline(float(matrix.iloc[0].cloud_speed), color="#607381", linewidth=1, alpha=.65)
    ax.set(xlabel="Assumed cloud CPU speed / edge CPU speed", ylabel="Cloud − edge\nmean delay (ms)",
           xticks=[1, 2, 3, 4, 6, 8])
    ax.grid(axis="y", alpha=.22)
    ax.legend(ncols=3, loc="upper center")
    save(fig, out / "heavy_sensitivity")

    fig, ax = plt.subplots(figsize=(6.3, 2.8), layout="constrained")
    matrix = matrix.set_index(["network", "workload"])
    vals = matrix["delta_mean_ms"].unstack().loc[list(BANDS), list(WORKLOADS)].to_numpy()
    biggest = np.max(np.abs(vals))
    ax.imshow(vals, cmap="BrBG", vmin=-biggest, vmax=biggest, aspect="auto")
    for i, band in enumerate(BANDS):
        for j, task in enumerate(WORKLOADS):
            row = matrix.loc[(band, task)]
            ax.text(j, i, f"{row.mean_winner} {abs(vals[i,j]):.1f} ms\nP95 favors {row.p95_winner}",
                    ha="center", va="center", fontsize=8.5,
                    color="white" if abs(vals[i, j]) / biggest > .57 else "#172632")
    ax.set(xticks=np.arange(3), xticklabels=WORKLOADS, yticks=np.arange(3), yticklabels=BANDS)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    save(fig, out / "decision_matrix")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    parser.add_argument("--results", default=Path("results"), type=Path)
    parser.add_argument("--out", default=Path("figures"), type=Path)
    a = parser.parse_args()
    main(a.data_dir, a.results, a.out)
