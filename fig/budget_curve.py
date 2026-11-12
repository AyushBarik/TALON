"""Accuracy against compute as the token budget varies (one column).

Each method is one line through its four budgets (1,536 / 3,072 / 6,144 / 12,288 tokens), placed at the TFLOPs it
measured at that budget. Data: fig/budget_curve.csv with columns method,budget,tflops,accuracy (one row per method and
budget, from the benchmark runs). Until that file exists the script draws the empty axes as a placeholder.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/budget_curve.py
"""

import csv
import os
from collections import defaultdict

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "budget_curve.csv")
BUDGETS = [1536, 3072, 6144, 12288]


def load(path):
    rows = defaultdict(list)
    if not os.path.exists(path):
        return rows
    with open(path) as handle:
        for r in csv.DictReader(handle):
            rows[r["method"]].append((float(r["tflops"]), float(r["accuracy"]), int(r["budget"])))
    return rows


def main():
    use_style()
    rows = load(DATA)
    fig, ax = figure(width=COLUMN, ratio=0.72)

    if rows:
        # TFLOPs on a log axis (QuoTA's frame scorer costs ~5x the rest); accuracy in whole points.
        ax.set_xscale("log")
        ax.set_xlim(200, 2000)
        ax.set_xticks([200, 500, 1000, 2000])
        ax.set_xticklabels(["200", "500", "1000", "2000"])
        ax.minorticks_off()
        ax.set_ylim(70, 80)
        ax.set_yticks([70, 72, 74, 76, 78, 80])
        for method, (color, marker) in METHODS.items():
            pts = sorted(rows.get(method, []))
            if not pts:
                continue
            x, y, _ = zip(*pts)
            ours = is_ours(method)
            ax.plot(x, y, color=color, marker=marker, ms=4 if ours else 3.5, lw=1.6 if ours else 1.0,
                    zorder=3 if ours else 2, label=method)
        ax.legend(loc="lower right", frameon=False, fontsize=6)
    else:
        # Placeholder: four x ticks, one per budget, and no data yet.
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([""] * 4)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels([""] * 4)

    ax.set_xlabel("TFLOPs per question (lower is better)")
    ax.set_ylabel("MLVU accuracy (%, higher is better)")
    print(save(fig, os.path.join(HERE, "budget_curve")))


if __name__ == "__main__":
    main()
