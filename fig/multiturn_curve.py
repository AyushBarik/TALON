"""Multi-turn: dialogue accuracy against the compute of the whole dialogue on MT-Video-Bench (one column).

Each method is one line through its budgets, placed at the TFLOPs it spent over the whole dialogue (all turns) at that
budget. Data: fig/multiturn_curve.csv with columns method,budget,tflops,accuracy (one row per method and budget, from
the MT-Video-Bench runs). Until that file exists the script draws the empty axes as a placeholder.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/multiturn_curve.py
"""

import csv
import os
from collections import defaultdict

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "multiturn_curve.csv")
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
        ax.set_xscale("log")  # compute spans a wide range; log keeps the cheap end readable
        for method, (color, marker) in METHODS.items():
            pts = sorted(rows.get(method, []))
            if not pts:
                continue
            x, y, _ = zip(*pts)
            ours = is_ours(method)
            ax.plot(x, y, color=color, marker=marker, ms=4 if ours else 3.5, lw=1.6 if ours else 1.0,
                    zorder=3 if ours else 2, label=method)
        ax.set_ylim(54, 62)  # the 95% interval of a score is about +-3.3 points at 223 judged turns
        ax.legend(loc="lower center", frameon=False, fontsize=6, ncol=2)
    else:
        # Placeholder: four x ticks, one per budget, and no data yet.
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0, 1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([""] * 4)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels([""] * 4)

    ax.set_xlabel("TFLOPs per conversation")
    ax.set_ylabel("MTVB accuracy (%)")
    print(save(fig, os.path.join(HERE, "multiturn_curve")))


if __name__ == "__main__":
    main()
