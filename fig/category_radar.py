"""Accuracy per question category on Video-MME, LongVideoBench and MLVU, for both VLMs and all seven Table 1 rows.

Columns are benchmarks, rows are VLMs; one shared legend. Each benchmark has one radial scale shared by its two VLMs.
Data (model, method, group, accuracy in %, [n], source result files), aggregated from the per-question results of the
Table 1 runs: videomme_radar.csv (2,700 questions per row; 12 task types grouped into six as in Q-Frame),
lvb_radar.csv (1,337 questions, 17 LongVideoBench categories), mlvu_radar.csv (2,174 questions, 7 MLVU tasks).

Regenerate:  uv run --no-project --with matplotlib --with numpy python category_radar.py
"""
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from orx_figstyle import PALETTE, WIDE, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["Qwen3-VL-8B", "InternVL3.5-4B"]
METHODS = {  # same colors and markers as the main results figures; ours last, drawn on top
    "Uniform sampling": (PALETTE["black"], "x"),
    "Top-K": (PALETTE["cyan"], "P"),
    "Q-Frame": (PALETTE["orange"], "s"),
    "QuoTA": (PALETTE["green"], "D"),
    "FOCUS": (PALETTE["red"], "^"),
    "AdaAlloc": (PALETTE["purple"], "X"),
    "Lembas (Ours)": (PALETTE["blue"], "o"),
}
OURS = "Lembas (Ours)"
LABEL = {OURS: "Lembas"}
BENCH = [  # (csv, title, groups clockwise from the top as (csv key, label), show counts, label size)
    ("videomme", "Video-MME",
     [("Reasoning", "Reasoning"), ("Recognition", "Recognition"), ("Perception", "Perception"),
      ("Counting Problem", "Counting"), ("OCR Problems", "OCR"), ("Information Synopsis", "Synopsis")], True, 6),
    ("lvb", "LongVideoBench",
     [(k, k) for k in ["S2E", "S2O", "S2A", "E2O", "O2E", "T2E", "T2O", "T2A",
                       "E3E", "O3O", "SSS", "SOS", "SAA", "T3E", "T3O", "TOS", "TAA"]], True, 5),
    ("mlvu", "MLVU",
     [("plotQA", "PlotQA"), ("needle", "Needle"), ("ego", "Ego"), ("count", "Count"), ("order", "Order"),
      ("anomaly_reco", "Anomaly"), ("topic_reasoning", "Topic")], True, 6),
]
VIDEOMME_N = {"Reasoning": 972, "Recognition": 667, "Perception": 331, "Counting Problem": 268, "OCR Problems": 139,
              "Information Synopsis": 323}


def load(name):
    acc, n = defaultdict(dict), dict(VIDEOMME_N) if name == "videomme" else {}
    with open(os.path.join(HERE, f"{name}_radar.csv")) as handle:
        for r in csv.DictReader(handle):
            acc[(r["model"], r["method"])][r["group"]] = float(r["accuracy"])
            if r.get("n"):
                n[r["group"]] = int(r["n"])
    return acc, n


def scale(values):
    """Three rings at multiples of 5; the outer circle is the last ring."""
    hi = math.ceil(max(values) / 5) * 5
    step = math.ceil((hi - min(values) + 8) / 3 / 5) * 5
    return hi - 3 * step, hi, [hi - 2 * step, hi - step, hi]


def radar(ax, model, acc, n, groups, counts, fs, lo, hi, rings):
    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(lo, hi)
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{lab}\n({n[k]})" if counts else lab for k, lab in groups], fontsize=fs)
    ax.tick_params(axis="x", pad=1 if counts else -1)
    ax.set_yticks(rings)
    ax.set_yticklabels([])
    for v in rings:  # ring values on a white patch, between the last and the first axis, above the lines
        ax.text(-np.pi / len(groups), v, str(v), fontsize=5, ha="center", va="center", zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.85))
    ax.grid(color="#BBBBBB", lw=0.4, ls=":")
    ax.spines["polar"].set_linewidth(0.6)
    for method, (color, marker) in METHODS.items():
        v = np.array([acc[(model, method)][k] for k, _ in groups])
        a, v = np.append(angles, angles[0]), np.append(v, v[0])
        ours = method == OURS
        ax.plot(a, v, color=color, marker=marker, ms=2.2, lw=1.2 if ours else 0.6, zorder=3 if ours else 2,
                clip_on=False)
        if ours:
            ax.fill(a, v, color=color, alpha=0.12, zorder=1)


def main():
    use_style()
    W, H = WIDE, 4.7
    size, col_w = 1.45, (WIDE - 0.3) / 3          # radar diameter and column width, inches
    row_cy = [3.25, 1.2]                          # radar centres from the bottom, inches
    fig = plt.figure(figsize=(W, H))
    for j, (name, title, groups, counts, fs) in enumerate(BENCH):
        acc, n = load(name)
        lo, hi, rings = scale([acc[k][g] for k in acc for g, _ in groups])
        cx = 0.3 + col_w * (j + 0.5)
        fig.text(cx / W, 1 - 0.04 / H, title, ha="center", va="top", fontsize=8, fontweight="bold")
        for i, model in enumerate(MODELS):
            ax = fig.add_axes([(cx - size / 2) / W, (row_cy[i] - size / 2) / H, size / W, size / H], polar=True)
            radar(ax, model, acc, n, groups, counts, fs, lo, hi, rings)
    for i, model in enumerate(MODELS):
        fig.text(0.06 / W, row_cy[i] / H, model, ha="left", va="center", rotation=90, fontsize=7)
    handles = [Line2D([], [], color=c, marker=m, ms=3.5, lw=1.4 if k == OURS else 0.8, label=LABEL.get(k, k))
               for k, (c, m) in METHODS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=7, handlelength=1.8,
               columnspacing=1.2, borderaxespad=0.1)
    print(save(fig, os.path.join(HERE, "category_radar")))
    plt.close(fig)


if __name__ == "__main__":
    main()
