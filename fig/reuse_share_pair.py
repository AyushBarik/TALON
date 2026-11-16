"""Multi-turn reuse, two panels in ONE column. 300 MT-Video-Bench conversations, 12,288 tokens per turn, turns 1 to 5
(overlap_turns.json for Lembas, baseline_reuse_turns.csv for Top-K).

(a) Share of the tokens a turn needs (allocated frame tokens + question tokens) that an earlier turn of the same
    conversation already used, pooled over the conversations. Top-K and Lembas only: this share is a property of what
    a turn needs, so it is the same with and without incremental allocation.
(b) Tokens prefilled per turn (tokens needed that the KV cache does not hold), mean over the conversations.
    "Lembas w/o incremental allocation" follows from its definition: the whole prompt is prefilled again at every turn, so
    every token a turn needs is new. It is a Lembas variant, drawn solid in dark navy.
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/reuse_share_pair.py
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
TURNS = [1, 2, 3, 4, 5]
SHOWN = ("Top-K", "Lembas")
NO_INC = "Lembas w/o incremental allocation"
NO_INC_COLOR = "#0B2A4A"  # dark navy: a Lembas variant, same hue family, solid like every other line
RENAME = {"Top-K (w/ pruning)": "Top-K", "w/o carried allocation": NO_INC}


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLUMN, COLUMN * 0.6), layout="constrained")
    rows = [dict(r, method="Lembas") for r in json.load(open(os.path.join(HERE, "overlap_turns.json")))]
    rows += list(csv.DictReader(open(os.path.join(HERE, "baseline_reuse_turns.csv"))))
    got, need, n = (defaultdict(lambda: defaultdict(int)) for _ in range(3))
    for r in rows:
        m, t = RENAME.get(r["method"], r["method"]), int(r["turn"])
        got[m][t] += int(r["reused"]); need[m][t] += int(r["need"]); n[m][t] += 1
    handles = {}
    for method, (color, marker) in METHODS.items():
        if method in SHOWN:
            share = [100 * got[method][t] / need[method][t] for t in TURNS]
            print(method, [round(s, 1) for s in share])
            handles[method] = ax.plot(TURNS, share, color=color, marker=marker, ms=2.6, lw=1.0,
                                      zorder=3 if is_ours(method) else 2, clip_on=False)[0]
    for method, (color, marker) in METHODS.items():
        if method in SHOWN:
            bx.plot(TURNS, [(need[method][t] - got[method][t]) / n[method][t] / 1000 for t in TURNS], color=color,
                    marker=marker, ms=2.6, lw=1.0, zorder=3 if is_ours(method) else 2, clip_on=False)
    m = "Lembas"
    handles[NO_INC] = bx.plot(TURNS, [need[m][t] / n[m][t] / 1000 for t in TURNS], color=NO_INC_COLOR,
                              marker="v", ms=2.8, lw=1.0, zorder=2.5, clip_on=False)[0]
    ax.set_xlim(1, 5); ax.set_xticks(TURNS); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Turn"); ax.set_ylabel("Frames or tokens seen\nin earlier turns (%)", labelpad=1)
    bx.set_xlim(1, 5); bx.set_xticks(TURNS); bx.set_ylim(0, 15); bx.set_yticks([0, 5, 10, 15])
    bx.set_yticklabels(["0", "5k", "10k", "15k"])
    bx.set_xlabel("Turn"); bx.set_ylabel("Tokens prefilled", labelpad=1)
    order = [m for m in METHODS if m in handles] + [NO_INC]
    fig.legend([handles[m] for m in order], order, loc="outside upper center", ncol=3, frameon=False, fontsize=5.5,
               columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    panel_labels([ax, bx])
    print(save(fig, os.path.join(HERE, "reuse_share_pair")))


if __name__ == "__main__":
    main()
