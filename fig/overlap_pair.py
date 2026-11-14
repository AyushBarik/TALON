"""Two CDFs in ONE column.

(a) Single turn: share of the 128 candidate frames that receive at least one token, per question (all 6,211 questions
    of MLVU, LVB and Video-MME pooled, Qwen3-VL-8B, 12,288-token budget; fig/frames_cdf.csv, Table 1 runs).
(b) Multi turn: share of the tokens a turn needs (allocated frame tokens + question tokens) that the KV cache already
    holds, for every turn after the first of 300 MT-Video-Bench conversations (1,473 turns; fig/overlap_turns.json,
    an offline replay of TALON's allocations with retrieval relevance at 12,288 tokens per turn).
Every axis starts at its first tick and ends at its last tick.

Regenerate:  uv run --no-project --with matplotlib --with numpy python fig/overlap_pair.py
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from methods import METHODS, is_ours
from orx_figstyle import COLUMN, panel_labels, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
TICKS = [0, 25, 50, 75, 100]


def cdf(ax, values, **style):
    v = np.sort(np.asarray(values, float))
    y = 100 * np.arange(1, len(v) + 1) / len(v)
    return ax.step(np.concatenate([[0], v, [100]]), np.concatenate([[0], y, [100]]), where="post", **style)[0]


def main():
    use_style()
    plt.rcParams.update({"xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.labelsize": 6.5})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(COLUMN, COLUMN * 0.6), layout="constrained")
    by = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(HERE, "frames_cdf.csv"))):
        by[r["method"]].append(100 * int(r["frames"]) / 128)
    handles = {}
    for method, (color, _) in METHODS.items():
        if method in by:
            ours = is_ours(method)
            handles[method] = cdf(ax, by[method], color=color, lw=1.0, zorder=3 if ours else 2)  # one line width for all
    turns = [r for r in json.load(open(os.path.join(HERE, "overlap_turns.json"))) if r["turn"] >= 2]
    color = METHODS["TALON (Ours)"][0]
    cdf(bx, [100 * r["reused"] / r["need"] for r in turns], color=color, lw=1.0)
    for a, xl, yl in ((ax, "Candidate frames\nshown (%)", "CDF of queries (%)"), (bx, "Tokens reused from\nprevious turns (%)", "CDF of turns (%)")):
        a.set_xlim(0, 100); a.set_ylim(0, 100); a.set_xticks(TICKS); a.set_yticks(TICKS)
        a.set_xlabel(xl); a.set_ylabel(yl, labelpad=1)
    order = [m for m in METHODS if m in handles]
    fig.legend([handles[m] for m in order], order, loc="outside upper center", ncol=3, frameon=False, fontsize=5.5,
               columnspacing=0.8, handlelength=1.6, handletextpad=0.4)
    panel_labels([ax, bx])
    print(save(fig, os.path.join(HERE, "overlap_pair")))


if __name__ == "__main__":
    main()
