"""Single-turn accuracy against compute (TFLOPs per query) as the token budget varies, on MLVU (one column).

Each method is one line through its four budgets (1,536 / 3,072 / 6,144 / 12,288 tokens), placed at the TFLOPs it
measured at that budget, on LongVideoBench (LVB). QuoTA's frame scorer puts it far right of the axis (~1,600+ TFLOPs). Data: fig/budget_curve.csv with columns method,budget,tflops,accuracy (one row per method and
budget, from the benchmark runs). Until that file exists the script draws the empty axes as a placeholder.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/budget_curve.py
"""

import csv

import matplotlib.ticker
import os
from collections import defaultdict

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "tflops_curve.csv")
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
    """Broken log x-axis: TFLOPs per question (LLM prefill of every VLM call, Sec. 4) for every method, and a narrow
    panel for QuoTA, whose 128 Qwen2-VL-2B scorer calls per question put it near 600-800 TFLOPs."""
    import matplotlib.pyplot as plt
    use_style()
    rows = load(DATA)
    fig = plt.figure(figsize=(COLUMN, COLUMN * 0.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.25], wspace=0.07, left=0.13, right=0.97, bottom=0.16, top=0.97)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    LEFT, RIGHT = (8.5, 300.0), (560.0, 860.0)
    for a in (ax, ax2):
        a.set_xscale("log")
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(*LEFT)
    ax.set_xticks([10, 25, 50, 100, 200])
    ax.set_xticklabels(["10", "25", "50", "100", "200"])
    ax2.set_xlim(*RIGHT)
    ax2.set_xticks([600, 800])
    ax2.set_xticklabels(["600", "800"])
    ax.set_ylim(60, 80)
    ax.set_yticks([60, 64, 68, 72, 76, 80])
    ax2.tick_params(labelleft=False, left=False)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    d = 0.015  # diagonal break marks, same angle on both panels
    kw = dict(color="black", clip_on=False, lw=0.8)
    ax.plot((1 - d, 1 + d), (-d * 1.6, d * 1.6), transform=ax.transAxes, **kw)
    r = 5.0 / 1.25
    ax2.plot((-d * r, d * r), (-d * 1.6, d * 1.6), transform=ax2.transAxes, **kw)
    # As in Table 1: "RAG" is a label only (no line), and each method below it is "+ method".
    ax.plot([], [], " ", label="RAG")
    for method, (color, marker) in METHODS.items():
        pts = sorted(rows.get(method, []))
        if method == "RAG" or not pts:
            continue
        ours = is_ours(method)
        style = dict(color=color, marker=marker, ms=4 if ours else 3.5, lw=1.6 if ours else 1.0, zorder=3 if ours else 2)
        left = [p for p in pts if LEFT[0] <= p[0] <= LEFT[1]]
        right = [p for p in pts if p[0] >= RIGHT[0]]
        if left:
            x, y, _ = zip(*left)
            ax.plot(x, y, label="+ " + method, **style)
        else:
            ax.plot([], [], label="+ " + method, **style)
        if right:
            x, y, _ = zip(*right)
            ax2.plot(x, y, **style)
    ax.legend(loc="lower right", frameon=False, fontsize=6, ncol=2, columnspacing=0.8)
    ax.set_xlabel("TFLOPs per query")
    l0, l1 = ax.get_position().x0, ax2.get_position().x1
    ax.xaxis.set_label_coords(((l0 + l1) / 2 - l0) / ax.get_position().width, -0.12)
    ax.set_ylabel("MLVU accuracy (%)")
    print(save(fig, os.path.join(HERE, "tflops_curve")))


if __name__ == "__main__":
    main()
