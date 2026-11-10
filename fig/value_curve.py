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


def main():
    rows = list(csv.DictReader(open(os.path.join(HERE, "value_curve.csv"))))
    b = np.array([float(r["tokens_per_frame"]) for r in rows])
    acc = np.array([100 * float(r["accuracy"]) for r in rows])

    use_style()
    fig, ax = figure(width=COLUMN, ratio=0.72)
    grid = np.exp(np.linspace(np.log(3.5), np.log(1000), 300))
    for name, (f, p0, color, ls) in FAMILIES.items():
        p, _ = curve_fit(f, b, acc, p0=p0, maxfev=20000)
        rmse = np.sqrt(np.mean((f(b, *p) - acc) ** 2))
        print(f"{name:12s} params={np.round(p, 3)} RMSE={rmse:.2f} pts")
        ax.plot(grid, f(grid, *p), color=color, ls=ls, lw=1.1, label=name, zorder=2)
    ax.plot(b, acc, "o", color="black", ms=3, zorder=3)

    ax.set_xscale("log")
    ax.set_xticks([4, 16, 64, 256, 1024])
    ax.set_xticklabels(["4", "16", "64", "256", "1024"])
    ax.minorticks_off()
    ax.set_xlim(3, 1200)
    ax.set_ylim(15, 55)
    ax.set_yticks([20, 30, 40, 50])
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", frameon=False, fontsize=6)
    print(save(fig, os.path.join(HERE, "value_curve")))


if __name__ == "__main__":
    main()
