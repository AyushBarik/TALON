"""Accuracy against compute, single turn and multi turn side by side in ONE column.

(a) single turn: MLVU accuracy against TFLOPs per query (fig/tflops_curve.csv);
(b) multi turn: MT-Video-Bench accuracy against TFLOPs per conversation, all turns (fig/multiturn_curve.csv).
Each method is one line through its four per-turn budgets (1,536 / 3,072 / 6,144 / 12,288 tokens). One shared legend.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/tflops_pair.py
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    rows = defaultdict(list)
    with open(os.path.join(HERE, name)) as handle:
        for r in csv.DictReader(handle):
            rows[r["method"]].append((float(r["tflops"]), float(r["accuracy"])))
    return rows


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, COLUMN * 0.62), layout="constrained")
    # (file, x label, y label, x ticks, y ticks): every axis starts at its first tick and ends at its last tick
    panels = [("tflops_curve.csv", "TFLOPs per query", "MLVU accuracy (%)", [10, 100, 1000], [64, 68, 72, 76, 80]),
              ("multiturn_curve.csv", "TFLOPs per conversation", "MTVB accuracy (%)", [50, 500, 5000], [52, 54, 56, 58, 60, 62])]
    handles = {}
    for ax, (name, xlabel, ylabel, ticks, yticks) in zip(axes, panels):
        rows = load(name)
        ax.set_xscale("log")
        for method, (color, marker) in METHODS.items():
            pts = sorted(rows.get(method, []))
            if not pts:
                continue
            ours = is_ours(method)
            x, y = zip(*pts)
            (h,) = ax.plot(x, y, color=color, marker=marker, ms=3 if ours else 2.5, lw=1.4 if ours else 0.8,
                           zorder=3 if ours else 2, label=method)
            handles.setdefault(method, h)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_yticks(yticks)  # accuracy ticks are always integers
        ax.set_ylim(yticks[0], yticks[-1])
    order = [m for m in METHODS if m in handles]
    fig.legend([handles[m] for m in order], order, loc="outside upper center", ncol=4, frameon=False, fontsize=5.5,
               columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    panel_labels(axes)
    print(save(fig, os.path.join(HERE, "tflops_pair")))


if __name__ == "__main__":
    main()
