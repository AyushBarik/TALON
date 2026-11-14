"""Accuracy against the token budget of each frame, with four curve families fitted to the measured means (one column).

Data: fig/value_curve.csv, one row per token level, from the held-out LVBench calibration (Qwen3-VL-8B, 280 questions,
the 8 most relevant frames per question, every frame at b tokens). The native level is plotted at its mean token count.
Points are the measured means (no intervals shown). Each family is fitted by least squares to the 9 level means:
  Sigmoid      a + c / (1 + exp(-k (log b - m)))
  Log          a + c log b
  Exponential  a + c (1 - exp(-b / t))
  Rational     a + c b / (b + t)

Regenerate:  uv run --no-project --with matplotlib --with numpy --with scipy python fig/value_curve.py
"""

import csv
import os

import numpy as np
from scipy.optimize import curve_fit

from orx_figstyle import COLUMN, PALETTE, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = {
    "Sigmoid": (lambda b, a, c, k, m: a + c / (1 + np.exp(-k * (np.log(b) - m))), [20, 25, 1.5, 3.0],
                PALETTE["blue"], "-"),
    "Log": (lambda b, a, c: a + c * np.log(b), [15, 4], PALETTE["orange"], "-"),
    "Exponential": (lambda b, a, c, t: a + c * (1 - np.exp(-b / t)), [20, 25, 50], PALETTE["green"], "-"),
    "Rational": (lambda b, a, c, t: a + c * b / (b + t), [20, 25, 30], PALETTE["purple"], "-"),
}

# Which line is drawn on top: Sigmoid, then Exponential, then Rational, then Log (the data points stay above all).
ZORDER = {"Sigmoid": 2.4, "Exponential": 2.3, "Rational": 2.2, "Log": 2.1}


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "value_curve.csv"))))
    b = np.array([float(r["tokens_per_frame"]) for r in rows])
    acc = np.array([100 * float(r["accuracy"]) for r in rows])

    use_style()
    import matplotlib
    matplotlib.rcParams.update({"xtick.labelsize": 6, "ytick.labelsize": 6, "axes.labelsize": 7})
    import matplotlib.pyplot as plt
    # Three quarters of a column wide (included at 0.75\linewidth), so it prints at this size.
    fig, ax = plt.subplots(figsize=(COLUMN * 0.75, COLUMN * 0.75 * 0.62))
    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.27, top=0.96)
    grid = np.exp(np.linspace(np.log(3.5), np.log(1000), 300))
    for name, (f, p0, color, ls) in FAMILIES.items():
        p, _ = curve_fit(f, b, acc, p0=p0, maxfev=20000)
        rmse = np.sqrt(np.mean((f(b, *p) - acc) ** 2))
        print(f"{name:12s} params={np.round(p, 3)} RMSE={rmse:.2f} pts")
        ax.plot(grid, f(grid, *p), color=color, ls=ls, lw=1.1, label=name, zorder=ZORDER[name])
    ax.plot(b, acc, "o", color="black", ms=2.5, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([2, 8, 32, 128, 512, 2048])
    ax.set_xticklabels(["2", "8", "32", "128", "512", "2048"])
    ax.minorticks_off()
    ax.set_xlim(2, 2048)  # every axis starts at its first tick and ends at its last tick
    ax.set_ylim(20, 50)
    ax.set_yticks([20, 30, 40, 50])
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=5.5, labelspacing=0.25, handlelength=1.5)
    print(save(fig, os.path.join(HERE, "value_curve")))


if __name__ == "__main__":
    main()
