"""Relevance and token value side by side in ONE column.

(a) relevance over time for one Video-MME question (fig/relevance_example.json: "What is the color of the pneumatic
    air gun?", video WOGaS19VZ8Q): SigLIP 2 cosine of each candidate frame (left axis) and BGE cosine of each
    subtitle cue (right axis) against the question;
(b) accuracy against the tokens per frame with four fitted curve families (fig/value_curve.csv), as value_curve.py.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy --with scipy python fig/value_pair.py
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from orx_figstyle import COLUMN, PALETTE, panel_labels, save, use_style
from value_curve import FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLUMN, COLUMN * 0.5), layout="constrained")

    # (a) relevance over time
    d = json.load(open(os.path.join(HERE, "relevance_example.json")))
    blue, orange = PALETTE["blue"], PALETTE["orange"]
    ax.plot(np.array(d["frame_t"]) / 60, d["siglip"], color=blue, lw=0.8)
    ax.set_xlim(0, 9)
    ax.set_xticks([0, 3, 6, 9])
    ax.set_ylim(-0.04, 0.12)
    ax.set_yticks([-0.04, 0.0, 0.04, 0.08, 0.12])
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("SigLIP 2 score", color=blue, labelpad=1)
    ax.tick_params(axis="y", colors=blue)
    tx = ax.twinx()
    tx.plot(np.array(d["cue_t"]) / 60, d["bge"], color=orange, lw=0.8)
    tx.set_ylim(0.2, 0.7)
    tx.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    tx.set_ylabel("BGE score", color=orange, labelpad=1)
    tx.tick_params(axis="y", colors=orange)
    tx.spines["right"].set_visible(True)
    tx.grid(False)

    # (b) token value
    rows = list(csv.DictReader(open(os.path.join(HERE, "value_curve.csv"))))
    b = np.array([float(r["tokens_per_frame"]) for r in rows])
    acc = np.array([100 * float(r["accuracy"]) for r in rows])
    grid = np.exp(np.linspace(np.log(3.5), np.log(1000), 300))
    for name, (f, p0, color, ls) in FAMILIES.items():
        p, _ = curve_fit(f, b, acc, p0=p0, maxfev=20000)
        bx.plot(grid, f(grid, *p), color=color, ls=ls, lw=0.9, label=name, zorder=2)
    bx.plot(b, acc, "o", color="black", ms=2.2, zorder=3)
    bx.set_xscale("log")
    bx.minorticks_off()
    bx.set_xlim(2, 2048)
    bx.set_xticks([2, 8, 32, 128, 512, 2048])
    bx.set_xticklabels(["2", "8", "32", "128", "512", "2048"])
    bx.set_ylim(20, 50)
    bx.set_yticks([20, 30, 40, 50])
    bx.set_xlabel("Token budget")
    bx.set_ylabel("Accuracy (%)", labelpad=1)
    bx.legend(loc="lower right", frameon=False, fontsize=5, handlelength=1.4, labelspacing=0.25)
    panel_labels([ax, bx])
    print(save(fig, os.path.join(HERE, "value_pair")))


if __name__ == "__main__":
    main()
