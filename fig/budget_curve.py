"""Accuracy against compute as the token budget varies (one column).

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
    """Broken x-axis: $1-8 for every method, and a narrow panel for QuoTA, whose 128 scorer calls per question put
    it near $50 (log axes). Cost = OpenRouter API bill (input + output) on the 2,174 MLVU questions, scaled to 1,000 queries (csv column
    "tflops" holds these dollars)."""
    import matplotlib.pyplot as plt
    use_style()
    rows = load(DATA)
    fig = plt.figure(figsize=(COLUMN, COLUMN * 0.72))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.25], wspace=0.07, left=0.13, right=0.97, bottom=0.17, top=0.97)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax)
    LEFT, RIGHT = (0.4, 5.0), (22.0, 25.0)  # each axis starts and ends at a labelled tick
    # The axis starts at 0.4, which is not a 1-2-5 value, so the ticks are set here: the start and the 1-2-5 values
    # are labelled; 0.5 sits too close to 0.4 for a label and gets a short tick like the other in-between values.
    ax.set_xscale("log")
    ax.set_xlim(*LEFT)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.4, 1, 2, 5]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([0.5, 0.6, 0.7, 0.8, 0.9, 3, 4]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # The right panel is a cut-out of the same log scale around QuoTA: it spans less than one 1-2-5 step, so its two
    # end ticks are labelled and the integers between them get the short ticks.
    ax2.set_xscale("log")
    ax2.set_xlim(*RIGHT)
    ax2.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([22, 25]))
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([23, 24]))
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.get_xticklabels()[-1].set_ha("right")  # keep "4" and "22" apart at the axis break
    ax2.get_xticklabels()[0].set_ha("left")
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
    for method, (color, marker) in METHODS.items():
        pts = sorted(rows.get(method, []))
        if not pts:
            continue
        ours = is_ours(method)
        style = dict(color=color, marker=marker, ms=4 if ours else 3.5, lw=1.6 if ours else 1.0, zorder=3 if ours else 2)
        left = [p for p in pts if p[0] <= LEFT[1]]
        right = [p for p in pts if p[0] >= RIGHT[0]]
        if left:
            x, y, _ = zip(*left)
            ax.plot(x, y, label=method, **style)
        else:
            ax.plot([], [], label=method, **style)
        if right:
            x, y, _ = zip(*right)
            ax2.plot(x, y, **style)
    # Boxed legend in the lower right corner of the whole plot (drawn on the right panel so it covers both).
    handles, labels = ax.get_legend_handles_labels()
    legend = ax2.legend(handles, labels, loc="lower right", frameon=True, framealpha=1.0, edgecolor="black",
                        fancybox=False, fontsize=6, ncol=2, columnspacing=0.8, borderaxespad=0.5)
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(5)
    # "Better" arrow towards the good corner: lower cost, higher accuracy.
    ax.annotate("", xy=(0.04, 0.96), xytext=(0.16, 0.84), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0, mutation_scale=9))
    ax.text(0.17, 0.84, "Better", transform=ax.transAxes, fontsize=7, ha="left", va="center")
    ax.set_xlabel("Cost per 1k queries ($)")
    # one x label for both panels, centred under the whole axis
    l0, l1 = ax.get_position().x0, ax2.get_position().x1
    ax.xaxis.set_label_coords(((l0 + l1) / 2 - l0) / ax.get_position().width, -0.12)
    ax.set_ylabel("MLVU accuracy (%)")
    print(save(fig, os.path.join(HERE, "budget_curve")))


if __name__ == "__main__":
    main()
