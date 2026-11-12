"""Video-MME accuracy per task group, one radar per VLM, side by side in one column.

The 12 Video-MME task types are grouped into six as in Q-Frame: Reasoning (temporal, spatial, action and object
reasoning), Recognition (action and object recognition), Perception (temporal, spatial and attribute perception),
Counting, OCR and Information Synopsis. Question counts are from the Video-MME annotations (2,700 questions).

Data: fig/videomme_radar.csv with columns model,method,group,accuracy (accuracy in %, per-question results of the main
table aggregated by group). Until that file exists the script draws the empty radars as a placeholder.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/videomme_radar.py
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "videomme_radar.csv")
MODELS = ["Qwen3-VL-8B"]  # one radar, half a column wide, next to Table 2
# (key in the CSV, axis label, questions), clockwise from the top
GROUPS = [
    ("Reasoning", "Reasoning", 972),
    ("Recognition", "Recognition", 667),
    ("Perception", "Perception", 331),
    ("Counting Problem", "Counting", 268),
    ("OCR Problems", "OCR", 139),
    ("Information Synopsis", "Synopsis", 323),
]


def load(path):
    acc = defaultdict(dict)  # (model, method) -> {group: accuracy}
    if not os.path.exists(path):
        return acc
    with open(path) as handle:
        for r in csv.DictReader(handle):
            acc[(r["model"], r["method"])][r["group"]] = float(r["accuracy"])
    return acc


def radar(ax, model, acc, lo, hi):
    angles = np.linspace(0, 2 * np.pi, len(GROUPS), endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(lo, hi)
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{label}\n({n})" for _, label, n in GROUPS], fontsize=5.5)
    ax.tick_params(axis="x", pad=1)
    rings = np.linspace(lo, hi, 4)[1:-1]
    ax.set_yticks(rings)
    ax.set_yticklabels([f"{v:.0f}" for v in rings] if acc else [], fontsize=5)
    ax.set_rlabel_position(270)  # ring values between OCR and Synopsis, away from the polygon edges
    ax.grid(color="#BBBBBB", lw=0.4, ls=":")
    ax.spines["polar"].set_linewidth(0.6)
    for method, (color, marker) in METHODS.items():
        vals = acc.get((model, method))
        if not vals:
            continue
        v = np.array([vals[g] for g, _, _ in GROUPS])
        a, v = np.append(angles, angles[0]), np.append(v, v[0])
        ours = is_ours(method)
        ax.plot(a, v, color=color, marker=marker, ms=2.5, lw=1.3 if ours else 0.8, zorder=3 if ours else 2)
        if ours:
            ax.fill(a, v, color=color, alpha=0.12, zorder=1)


def main():
    use_style()
    acc = load(DATA)
    vals = [x for d in acc.values() for x in d.values()]
    lo, hi = (np.floor(min(vals) / 5) * 5 - 5, np.ceil(max(vals) / 5) * 5) if vals else (0.0, 1.0)

    width, height, size = 1.62, 2.05, 1.0  # figure width/height and radar diameter (in): half of a 3.25 in column
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([(1 - size / width) / 2, 0.40, size / width, size / height], polar=True)
    radar(ax, MODELS[0], acc, lo, hi)
    handles = [Line2D([], [], color=c, marker=m, ms=3, lw=1.3 if is_ours(k) else 0.8,
                      label=k if k == "RAG" else "+ " + k)
               for k, (c, m) in METHODS.items() if (MODELS[0], k) in acc]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=5.5, handlelength=1.6,
               columnspacing=0.8, borderaxespad=0.1)
    print(save(fig, os.path.join(HERE, "videomme_radar")))


if __name__ == "__main__":
    main()
